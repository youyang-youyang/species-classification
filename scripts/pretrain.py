"""
预训练入口脚本

用法:
  # 单卡调试
  python scripts/pretrain.py --config configs/pretrain_bacteria.yaml --gpus 0

  # 4 卡 DDP 训练
  torchrun --nproc_per_node=4 scripts/pretrain.py --config configs/pretrain_bacteria.yaml
"""

import sys
import os
import argparse

# 将项目根目录加入 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from src.config import Config
from src.model.model import build_model
from src.data.dataset import PretrainDataset, collate_pretrain
from src.data.fna_reader import FNAReader
from src.training.trainer import Trainer

import pickle
import numpy as np


def pretrain_forward_fn(model, batch, device, loss_fn=None):
    """
    预训练前向传播回调

    Args:
        model: SpeciesClassifier (pretrain mode)
        batch: dict from PretrainDataset
        device: torch.device
        loss_fn: unused (内置于 model)

    Returns:
        loss: scalar
        metrics: dict
    """
    masked_bases = batch['masked_bases'].to(device)       # [B, L, 21]
    mask_positions = batch['mask_positions'].to(device)    # [B, L]
    targets = batch['targets'].to(device)                  # [B, L, 21]

    # 提取原始模型 (DDP wrapper 下)
    m = model.module if hasattr(model, 'module') else model

    loss, metrics = m.forward_pretrain(
        bases_21=masked_bases,
        mask_positions=mask_positions,
        mlm_targets=targets,
    )

    return loss, metrics


def main():
    parser = argparse.ArgumentParser(description='Species Classifier Pretraining')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    parser.add_argument('--gpus', type=str, default=None,
                        help='Comma-separated GPU IDs (single-GPU mode)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    args = parser.parse_args()

    # 加载配置
    config = Config.from_yaml(args.config)

    # 分布式设置
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    elif args.gpus:
        gpu_id = int(args.gpus.split(',')[0])
        torch.cuda.set_device(gpu_id)
        local_rank = 0

    device = torch.device(f'cuda:{local_rank}')

    # 设置随机种子
    torch.manual_seed(config.seed + local_rank)
    np.random.seed(config.seed + local_rank)

    # 加载窗口索引
    index_path = os.path.join(config.data.index_dir, 'window_index_pretrain.pkl')
    print(f"[Pretrain] Loading window index: {index_path}")
    with open(index_path, 'rb') as f:
        index_data = pickle.load(f)

    train_windows = index_data['train']
    val_windows = index_data.get('val', [])

    print(f"[Pretrain] Train windows: {len(train_windows):,}")
    print(f"[Pretrain] Val windows: {len(val_windows):,}")

    # 创建 FNA Reader
    fna_reader = FNAReader(config.data.fna_file)

    # 创建 Dataset
    train_dataset = PretrainDataset(
        window_index=train_windows,
        fna_reader=fna_reader,
        window_size_bp=config.data.window_size_bp,
        kmer_size=config.model.kmer_size,
        mask_ratio=config.training.mlm_mask_ratio,
        seed=config.seed,
    )

    val_dataset = None
    if val_windows:
        val_dataset = PretrainDataset(
            window_index=val_windows,
            fna_reader=fna_reader,
            window_size_bp=config.data.window_size_bp,
            kmer_size=config.model.kmer_size,
            mask_ratio=config.training.mlm_mask_ratio,
            seed=config.seed + 1000,
        )

    # 创建 DataLoader
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if val_dataset else None
    else:
        train_sampler = RandomSampler(train_dataset)
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size_per_gpu,
        sampler=train_sampler,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=collate_pretrain,
        drop_last=True,
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size_per_gpu,
            sampler=val_sampler,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            collate_fn=collate_pretrain,
        )

    # 创建模型
    model = build_model(config.model, mode='pretrain')

    # 创建训练器
    trainer = Trainer(
        model=model,
        config=config.training,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=config.output_dir,
        experiment_name=config.experiment_name + '_pretrain',
        rank=local_rank,
        world_size=world_size,
        device=device,
    )

    # 构建学习率调度器
    total_steps = (config.training.steps_per_epoch or len(train_loader)) \
                  * config.training.pretrain_epochs \
                  // config.training.gradient_accumulation_steps
    trainer.build_cosine_scheduler(total_steps, config.training.warmup_steps)

    # 开始训练
    trainer.fit(
        num_epochs=config.training.pretrain_epochs,
        forward_fn=pretrain_forward_fn,
        metric_key='mlm_acc',
        save_interval=config.training.save_interval,
        resume_from=args.resume,
    )

    # 保存配置
    if local_rank == 0:
        config.to_yaml(os.path.join(trainer.output_dir, 'config.yaml'))

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
