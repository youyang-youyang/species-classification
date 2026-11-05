"""
通用训练器: DDP + AMP + 梯度累积 + 学习率调度

支持:
  - 多卡 DDP (DistributedDataParallel)
  - BF16 混合精度 (A100 原生支持)
  - 梯度累积 (虚拟大 batch)
  - Warmup + Cosine 学习率调度
  - steps_per_epoch 截断 (大数据集采样)
  - Early Stopping
  - Checkpoint 管理
"""

import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from typing import Optional, Dict, Callable
import json
import math


def species_exposure_summary(counts):
    values = list(counts.values())
    return {
        'window_presentations': sum(values),
        'species_seen': sum(n > 0 for n in values),
        'species_tracked': len(values),
        'exposure_bins': {
            '0': sum(n == 0 for n in values),
            '1-9': sum(1 <= n < 10 for n in values),
            '10-49': sum(10 <= n < 50 for n in values),
            '50-199': sum(50 <= n < 200 for n in values),
            '200+': sum(n >= 200 for n in values),
        },
    }


class Trainer:
    """通用训练器"""

    def __init__(self, model: nn.Module, config,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 scheduler=None,
                 loss_fn: Optional[nn.Module] = None,
                 output_dir: str = './runs',
                 experiment_name: str = 'exp',
                 rank: int = 0,
                 world_size: int = 1,
                 device: torch.device = None):

        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_main = (rank == 0)

        # 模型
        self.model = model.to(self.device)
        if world_size > 1:
            # P/U 关闭时注意力偏置等参数可能不进入计算图；允许逐 batch 改变。
            self.model = DDP(self.model, device_ids=[rank],
                             find_unused_parameters=(getattr(model, 'mode', None) == 'finetune'))

        # 优化器
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(),
            lr=config.pretrain_lr if hasattr(config, 'pretrain_lr') else 5e-4,
            weight_decay=config.pretrain_weight_decay if hasattr(config, 'pretrain_weight_decay') else 0.01,
        )

        # 学习率调度器
        self.scheduler = scheduler

        # 损失函数 (预训练: 内置于模型; 微调: 外部传入)
        self.loss_fn = loss_fn

        # DataLoader
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 混合精度
        self.use_amp = config.precision in ('fp16', 'bf16')
        self.amp_dtype = torch.bfloat16 if config.precision == 'bf16' else torch.float16
        self.scaler = GradScaler(enabled=(config.precision == 'fp16'))

        # 梯度累积
        self.accumulation_steps = config.gradient_accumulation_steps
        self.max_grad_norm = config.max_grad_norm

        # 步数控制
        self.steps_per_epoch = config.steps_per_epoch
        self.val_steps = config.val_steps

        # 输出目录
        self.output_dir = os.path.join(output_dir, experiment_name)
        if self.is_main:
            os.makedirs(self.output_dir, exist_ok=True)

        # 状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = -float('inf')

    def _log(self, msg: str):
        """只在主进程打印"""
        if self.is_main:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def build_cosine_scheduler(self, total_steps: int, warmup_steps: int):
        """构建 Warmup + Cosine 学习率调度器"""
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )

    def train_epoch(self, epoch: int, forward_fn: Callable) -> Dict[str, float]:
        """
        训练一个 epoch

        Args:
            epoch: 当前 epoch
            forward_fn: 前向传播函数, 接受 (model, batch, device) 返回 (loss, metrics)

        Returns:
            avg_metrics: 平均指标
        """
        self.model.train()
        self.current_epoch = epoch

        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)

        metrics_sum = {}
        metrics_count = {}
        n_steps = 0
        self.optimizer.zero_grad()

        epoch_batches = len(self.train_loader)
        if self.steps_per_epoch:
            epoch_batches = min(epoch_batches, self.steps_per_epoch)

        for step_idx, batch in enumerate(self.train_loader):
            # steps_per_epoch 截断
            if self.steps_per_epoch and step_idx >= self.steps_per_epoch:
                break

            # 前向传播 + 损失计算
            with autocast(dtype=self.amp_dtype, enabled=self.use_amp):
                loss, step_metrics = forward_fn(self.model, batch, self.device,
                                                self.loss_fn)

            # 梯度累积
            group_start = (step_idx // self.accumulation_steps) * self.accumulation_steps
            group_size = min(self.accumulation_steps, epoch_batches - group_start)
            scaled_loss = loss / group_size
            self.scaler.scale(scaled_loss).backward()
            if hasattr(self, 'species_exposure'):
                for label in batch['labels']['species'].detach().cpu().tolist():
                    if label >= 0:
                        self.species_exposure[label] = self.species_exposure.get(label, 0) + 1

            if ((step_idx + 1) % self.accumulation_steps == 0
                    or step_idx + 1 == epoch_batches):
                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.max_grad_norm)
                # 更新参数
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                if self.scheduler is not None:
                    self.scheduler.step()

                self.global_step += 1

            # 累积指标
            for k, v in step_metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v
                metrics_count[k] = metrics_count.get(k, 0) + 1
            n_steps += 1

            # 日志
            if self.is_main and (step_idx + 1) % 100 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                avg = {k: v / metrics_count[k] for k, v in metrics_sum.items()}
                metrics_str = ' | '.join(f'{k}: {v:.4f}' for k, v in avg.items())
                self._log(f"Epoch {epoch} Step {step_idx+1} | LR: {lr:.2e} | {metrics_str}")

        avg_metrics = {k: v / metrics_count[k] for k, v in metrics_sum.items()}
        if hasattr(self, 'species_exposure') and self.is_main:
            summary = species_exposure_summary(self.species_exposure)
            self._log(f'Species exposure: {summary}')
            report = dict(epoch=epoch, global_step=self.global_step, counts=self.species_exposure,
                          summary=summary,
                          note='Cumulative inputs after backward, including repeats; not unique windows. '
                               'Epoch report may be newer than the last saved checkpoint.')
            target = os.path.join(self.output_dir, 'species_exposure.json')
            with open(target + '.tmp', 'w') as f:
                json.dump(report, f, indent=2)
            os.replace(target + '.tmp', target)
        return avg_metrics

    @torch.no_grad()
    def validate(self, forward_fn: Callable) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        metrics_sum = {}
        metrics_count = {}
        n_steps = 0

        if self.val_loader is None:
            return {}

        for step_idx, batch in enumerate(self.val_loader):
            if self.val_steps and step_idx >= self.val_steps:
                break

            with autocast(dtype=self.amp_dtype, enabled=self.use_amp):
                _, step_metrics = forward_fn(self.model, batch, self.device,
                                             self.loss_fn)

            for k, v in step_metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v
                metrics_count[k] = metrics_count.get(k, 0) + 1
            n_steps += 1

        avg_metrics = {k: v / metrics_count[k] for k, v in metrics_sum.items()}
        return avg_metrics

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """保存检查点"""
        if not self.is_main:
            return

        model_state = self.model.module.state_dict() if isinstance(
            self.model, DDP) else self.model.state_dict()

        ckpt = {
            'epoch': epoch,
            'model_state': model_state,
            'optimizer_state': self.optimizer.state_dict(),
            'metrics': metrics,
            'global_step': self.global_step,
        }
        if self.scheduler:
            ckpt['scheduler_state'] = self.scheduler.state_dict()
        if hasattr(self, 'species_exposure'):
            ckpt['species_exposure'] = dict(self.species_exposure)
            ckpt['fit_diagnostic'] = getattr(self, 'fit_diagnostic_metadata', None)
        ckpt['best_metric'] = self.best_metric

        # 定期保存
        path = os.path.join(self.output_dir, f'checkpoint_epoch{epoch:03d}.pt')
        torch.save(ckpt, path)
        self._log(f"Saved checkpoint: {path}")

        # 最佳模型
        if is_best:
            best_path = os.path.join(self.output_dir, 'best_model.pt')
            torch.save(ckpt, best_path)
            self._log(f"Saved best model: {best_path}")

    def load_checkpoint(self, ckpt_path: str):
        """加载检查点续训"""
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if hasattr(self, 'species_exposure'):
            if 'species_exposure' not in ckpt:
                raise ValueError('Cannot restore complete exposure history from a checkpoint without counts')
            self.species_exposure = {int(k): int(v) for k, v in ckpt['species_exposure'].items()}

        model = self.model.module if isinstance(self.model, DDP) else self.model
        model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.current_epoch = ckpt.get('epoch', 0)
        self.global_step = ckpt.get('global_step', 0)
        self.best_metric = ckpt.get('best_metric', ckpt.get('metrics', {}).get('best_metric', -float('inf')))

        if self.scheduler and 'scheduler_state' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler_state'])

        self._log(f"Loaded checkpoint: {ckpt_path}, epoch={self.current_epoch}")

    def fit(self, num_epochs: int, forward_fn: Callable,
            metric_key: str = 'mlm_acc',
            save_interval: int = 5,
            early_stop_patience: int = 0,
            resume_from: Optional[str] = None,
            train_eval_loader=None):
        """
        完整训练循环

        Args:
            num_epochs: 总训练 epoch 数
            forward_fn: 前向传播函数
            metric_key: 用于 early stopping 和 best model 的指标名
            save_interval: 每 N epoch 保存一次
            early_stop_patience: 早停耐心值, 0 = 不早停
            resume_from: 续训检查点路径
        """
        start_epoch = 0
        if resume_from:
            self.load_checkpoint(resume_from)
            start_epoch = self.current_epoch + 1

        patience_counter = 0

        for epoch in range(start_epoch, num_epochs):
            self._log(f"{'='*60}")
            self._log(f"Epoch {epoch}/{num_epochs-1}")

            # 训练
            train_metrics = self.train_epoch(epoch, forward_fn)
            self._log(f"Train: {train_metrics}")

            train_eval_metrics = {}
            if train_eval_loader is not None:
                original_loader, original_steps = self.val_loader, self.val_steps
                try:
                    self.val_loader, self.val_steps = train_eval_loader, None
                    train_eval_metrics = self.validate(forward_fn)
                finally:
                    self.val_loader, self.val_steps = original_loader, original_steps
                self._log(f'Train_eval: {train_eval_metrics}')

            # 验证
            val_metrics = self.validate(forward_fn)
            if val_metrics:
                self._log(f"Val:   {val_metrics}")

            # 合并指标
            all_metrics = {**{f'train_{k}': v for k, v in train_metrics.items()},
                           **{f'train_eval_{k}': v for k, v in train_eval_metrics.items()},
                           **{f'val_{k}': v for k, v in val_metrics.items()}}

            # 判断是否为最佳
            current_metric = val_metrics.get(metric_key,
                                             train_metrics.get(metric_key, 0))
            is_best = current_metric > self.best_metric
            if is_best:
                self.best_metric = current_metric
                patience_counter = 0
            else:
                patience_counter += 1

            # 保存
            if (epoch + 1) % save_interval == 0 or is_best:
                self.save_checkpoint(epoch, all_metrics, is_best)

            # Early stopping
            if early_stop_patience > 0 and patience_counter >= early_stop_patience:
                self._log(f"Early stopping at epoch {epoch} "
                          f"(patience={early_stop_patience})")
                break

        self._log(f"Training complete. Best {metric_key}: {self.best_metric:.4f}")
