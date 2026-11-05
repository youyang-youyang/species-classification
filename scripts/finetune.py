"""
微调入口脚本

用法:
  # 单卡调试
  python scripts/finetune.py --config configs/finetune_bacteria.yaml --pretrained runs/pretrain/best_model.pt

  # 4 卡 DDP 训练
  torchrun --nproc_per_node=4 scripts/finetune.py \
      --config configs/finetune_bacteria.yaml \
      --pretrained runs/pretrain/best_model.pt
"""

import sys
import os
import argparse
import json
from array import array
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from src.config import Config
from src.model.model import build_model
from src.model.losses import SpeciesClassificationLoss
from src.data.dataset import FinetuneDataset, collate_finetune
from src.data.fna_reader import FNAReader
from src.data.score_reader import ScoreReader
from src.data.taxonomy import TaxonomyDB
from src.training.trainer import Trainer

import pickle
import numpy as np


class FixedWindowPassSampler:
    """Visit every fixed window once per pass; shuffle separately each epoch."""
    def __init__(self, size, passes, seed):
        if size <= 0 or passes <= 0:
            raise ValueError('Fixed-window size and passes must be positive')
        self.size, self.passes, self.seed, self.epoch = size, passes, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.size * self.passes

    def __iter__(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        for _ in range(self.passes):
            yield from (int(i) for i in rng.permutation(self.size))


def choose_fit_species(counts, seed, per_bin=8, windows_per_species=20):
    rng = np.random.default_rng(seed)
    selected = []
    for low, high in ((100, 1000), (1000, 10000)):
        eligible = sorted(label for label, n in counts['train'].items()
                          if label >= 0 and low <= n < high
                          and n >= windows_per_species
                          and counts['val'].get(label, 0) >= windows_per_species)
        if len(eligible) < per_bin:
            raise ValueError(f'Not enough eligible species in [{low}, {high})')
        selected.extend(int(x) for x in rng.choice(eligible, per_bin, replace=False))
    return set(selected)


class MixedSpeciesSampler:
    """Single-GPU sampling with replacement and a fixed balanced quota per epoch.

    Compact index arrays avoid a Python integer object per training window.
    Only training windows are grouped; taxonomy labels match FinetuneDataset.
    """

    def __init__(self, windows, taxonomy_db, num_samples, balanced_fraction=0.5, seed=42):
        if not windows or num_samples <= 0:
            raise ValueError('Mixed sampling requires training windows and positive sample budget')
        if not 0 <= balanced_fraction <= 1:
            raise ValueError('balanced_fraction must be finite and in [0, 1]')
        self.n_windows = len(windows)
        self.num_samples = int(num_samples)
        self.n_balanced = int(self.num_samples * balanced_fraction + 0.5)
        self.seed = seed
        self.epoch = 0
        groups = {}
        taxid_groups = {}
        typecode = 'I' if len(windows) <= 2**32 else 'Q'
        for index, window in enumerate(windows):
            taxid = int(window[3])
            if taxid not in taxid_groups:
                label = taxonomy_db.encode_lineage(taxid).get('species', -1)
                if label < 0:
                    raise ValueError(f'Training TaxID {taxid} has no valid species label')
                if label not in groups:
                    groups[label] = array(typecode)
                taxid_groups[taxid] = groups[label]
            taxid_groups[taxid].append(index)
        self.species_labels = sorted(groups)
        self.groups = [groups[label] for label in self.species_labels]
        print(f'[Sampler] mixed: {len(self.groups):,} species; '
              f'{self.num_samples:,} windows/epoch = '
              f'{self.num_samples - self.n_balanced:,} uniform-window + '
              f'{self.n_balanced:,} species-balanced; sampling with replacement.', flush=True)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        indices = rng.integers(self.n_windows, size=self.num_samples, dtype=np.int64)
        for position, group_id in enumerate(rng.integers(len(self.groups), size=self.n_balanced)):
            group = self.groups[int(group_id)]
            indices[position] = group[int(rng.integers(len(group)))]
        rng.shuffle(indices)
        return (int(index) for index in indices)


def finetune_forward_fn(model, batch, device, loss_fn=None, guidance_dropout=0.0,
                        disable_phylo=False):
    """
    微调前向传播回调

    对应设计文档完整数据流:
      bases_21 + U + P_vec → TripleGatedEmbedding → Encoder(+B_U) → MS-MaxPool → Classifier
      → L_total = L_CE + λ₁·L_hier + λ₂·L_contrast + λ₃·L_focus
    """
    # 每个前向 micro-batch 抽一次；同一 DDP 步骤所有 rank 使用相同模式。
    # 验证不抽样，固定采用与 predict.py 一致的纯 DNA 输入。
    drop_guidance = not model.training
    if model.training:
        if guidance_dropout >= 1.0:
            drop_guidance = True
        elif guidance_dropout > 0.0:
            draw = torch.rand((), device=device)
            if dist.is_available() and dist.is_initialized():
                dist.broadcast(draw, src=0)
            drop_guidance = draw.item() < guidance_dropout

    bases_21 = batch['bases_21'].to(device)
    uniqueness_scores = None if drop_guidance else batch['uniqueness_scores'].to(device)
    phylo_vecs = (batch['phylo_vecs'].to(device)
                  if not drop_guidance and not disable_phylo and 'phylo_vecs' in batch else None)
    labels = {k: v.to(device) for k, v in batch['labels'].items()}

    # 必须经过 DDP.forward，不能绕过包装器直接调用 model.module。
    logits, Z, loss, metrics = model(
        bases_21=bases_21,
        uniqueness_scores=uniqueness_scores,
        phylo_vecs=phylo_vecs,
        labels=labels,
        loss_fn=loss_fn,
    )

    # 计算各级准确率
    with torch.no_grad():
        for level, level_logits in logits.items():
            if level in labels:
                level_labels = labels[level]
                valid = level_labels >= 0
                if valid.sum() > 0:
                    preds = level_logits[valid].argmax(dim=-1)
                    acc = (preds == level_labels[valid]).float().mean().item()
                    metrics[f'{level}_acc'] = acc

    metrics['guidance_drop_rate'] = float(drop_guidance)
    mode = 'dna_only' if drop_guidance else 'guided'
    metrics[f'{mode}_L_total'] = metrics['L_total']
    if 'species_acc' in metrics:
        metrics[f'{mode}_species_acc'] = metrics['species_acc']

    return loss, metrics


def main():
    parser = argparse.ArgumentParser(description='Species Classifier Finetuning')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained checkpoint')
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--classification-loss-only', action='store_true',
                        help='Species CE plus ungated upper-level CE; no contrast/focus loss')
    parser.add_argument('--guidance-dropout', type=float, default=None,
                        help='Override joint P/U dropout probability per training batch (0..1)')
    parser.add_argument('--disable-phylo', action='store_true',
                        help='Always disable P; guidance dropout then applies only to U')
    parser.add_argument('--experiment-name', default=None,
                        help='Separate output name for the new finetuning experiment')
    parser.add_argument('--species-balanced-fraction', type=float, default=None,
                        help='Fraction drawn uniformly by species (0 keeps original sampler; single GPU)')
    parser.add_argument('--fit-diagnostic', action='store_true',
                        help='Single-GPU fixed 16-species fitting diagnostic, 320 train/320 val windows')
    parser.add_argument('--steps-per-epoch', type=int, default=None,
                        help='Override training micro-batches per epoch')
    parser.add_argument('--track-species-exposure', action='store_true',
                        help='Count actual species inputs after backward; single GPU only')
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.classification_loss_only:
        if args.resume or not args.experiment_name:
            parser.error('Loss control requires a new --experiment-name, without --resume')
        config.training.classification_loss_only = True
    if args.steps_per_epoch is not None:
        if args.steps_per_epoch <= 0 or args.fit_diagnostic:
            parser.error('--steps-per-epoch must be positive and cannot override fit diagnostic')
        config.training.steps_per_epoch = args.steps_per_epoch
    if args.track_species_exposure:
        config.training.track_species_exposure = True
    if (config.training.steps_per_epoch and
            config.training.steps_per_epoch % config.training.gradient_accumulation_steps):
        parser.error('steps_per_epoch must be divisible by gradient_accumulation_steps')
    if args.species_balanced_fraction is not None:
        config.training.species_balanced_fraction = args.species_balanced_fraction
    if not 0 <= config.training.species_balanced_fraction <= 1:
        raise ValueError('species_balanced_fraction must be finite and in [0, 1]')
    if args.disable_phylo:
        config.training.disable_phylo = True
    if args.guidance_dropout is not None:
        config.training.guidance_dropout = args.guidance_dropout
    if not 0.0 <= config.training.guidance_dropout <= 1.0:
        raise ValueError('guidance_dropout must be finite and between 0 and 1')
    if config.data.dna_only:
        if args.guidance_dropout is not None and args.guidance_dropout != 1.0:
            parser.error('DNA-only data cannot enable P/U guidance')
        config.training.guidance_dropout = 1.0
        config.training.disable_phylo = True
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    fit_dir = os.path.join(config.output_dir, config.experiment_name + '_finetune')
    if args.classification_loss_only and os.path.exists(fit_dir):
        parser.error('Loss control output exists; use a new --experiment-name')
    if args.steps_per_epoch is not None and not args.resume and os.path.exists(fit_dir):
        parser.error('Output directory exists; use a new --experiment-name for budget experiments')
    if args.fit_diagnostic:
        if not args.experiment_name or args.resume:
            parser.error('Fit diagnostic requires a new --experiment-name and does not support resume')
        if os.path.exists(fit_dir):
            parser.error(f'Fit diagnostic output already exists: {fit_dir}; use a new experiment name')
        config.training.guidance_dropout = 1.0
        config.training.disable_phylo = True
        config.training.species_balanced_fraction = 0.0
        config.training.finetune_epochs = 50
        config.training.warmup_steps = 100
        config.training.val_steps = None
        config.training.early_stop_patience = 0
        config.training.save_interval = 5
        config.data.num_workers = 0

    # 分布式设置
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if config.training.track_species_exposure and world_size != 1:
        parser.error('Species exposure tracking currently supports single GPU only')
    if args.fit_diagnostic and world_size != 1:
        parser.error('Fit diagnostic currently supports single GPU only')
    if config.training.species_balanced_fraction > 0:
        if world_size != 1:
            raise ValueError('Mixed species sampling currently supports single-GPU runs only')
        if not config.training.steps_per_epoch or config.training.steps_per_epoch <= 0:
            raise ValueError('Mixed species sampling requires positive steps_per_epoch')

    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    elif args.gpus:
        gpu_id = int(args.gpus.split(',')[0])
        torch.cuda.set_device(gpu_id)
        local_rank = 0

    device = torch.device(f'cuda:{local_rank}')
    torch.manual_seed(config.seed + local_rank)
    np.random.seed(config.seed + local_rank)
    if local_rank == 0:
        guidance_mode = 'U-only dropout (P disabled)' if config.training.disable_phylo else 'Joint P/U dropout'
        print(f'[Finetune] {guidance_mode}={config.training.guidance_dropout}; '
              'validation and best-model selection use DNA only.', flush=True)

    # 加载分类数据库
    taxonomy_path = os.path.join(config.data.index_dir, 'taxonomy_db.pkl')
    taxonomy_db = TaxonomyDB.load(taxonomy_path)

    # 加载窗口索引
    index_path = os.path.join(config.data.index_dir, 'window_index_finetune.pkl')
    print(f"[Finetune] Loading window index: {index_path}")
    with open(index_path, 'rb') as f:
        index_data = pickle.load(f)

    train_windows = index_data['train']
    val_windows = index_data.get('val', [])
    print(f"[Finetune] Train: {len(train_windows):,} | Val: {len(val_windows):,}")
    fit_metadata = None
    if args.fit_diagnostic:
        from scripts.evaluate import count_species_windows, select_evaluation_windows
        os.makedirs(fit_dir, exist_ok=False)
        counts = count_species_windows(index_data, taxonomy_db)
        selected = choose_fit_species(counts, config.seed)
        train_windows, train_report = select_evaluation_windows(
            train_windows, taxonomy_db, 'species_balanced', 20, 0, config.seed,
            os.path.join(fit_dir, 'fit_train_manifest.json'), 'train', selected)
        val_windows, val_report = select_evaluation_windows(
            val_windows, taxonomy_db, 'species_balanced', 20, 0, config.seed,
            os.path.join(fit_dir, 'fit_val_manifest.json'), 'val', selected)
        if {w[0].accession for w in train_windows} & {w[0].accession for w in val_windows}:
            raise ValueError('Selected train and val accessions overlap; stop diagnostic')
        batch_size = config.training.batch_size_per_gpu
        if (len(train_windows) % (batch_size * config.training.gradient_accumulation_steps)
                or len(val_windows) % batch_size):
            raise ValueError('Fixed subset must divide evenly into batches and accumulation groups')
        config.training.steps_per_epoch = len(train_windows) * 5 // batch_size
        fit_metadata = dict(mode='fixed_window_fit', seed=config.seed, passes_per_epoch=5,
                            train=train_report, val=val_report,
                            species_names={str(s): taxonomy_db.idx2label['species'][s] for s in selected},
                            note='Diagnostic only; full classifier and original losses retained. No test sampling.')
        with open(os.path.join(fit_dir, 'fit_diagnostic.json'), 'w', encoding='utf-8') as f:
            json.dump(fit_metadata, f, indent=2)
        print(f'[Fit diagnostic] {len(selected)} species; {len(train_windows)} fixed train / '
              f'{len(val_windows)} fixed val windows; 5 complete train passes per epoch.', flush=True)
        del index_data

    # 加载 GTDB 进化向量查找表 (P_vec，三重门控第二路)
    phylo_dict = None
    phylo_path = getattr(config.data, 'phylo_file', '')
    if config.training.disable_phylo or config.training.guidance_dropout == 1.0:
        print('[Finetune] P disabled for all batches; skipping phylo database loading.')
    elif phylo_path and os.path.exists(phylo_path):
        print(f"[Finetune] 加载 GTDB 进化向量库 (Triple-Gated): {phylo_path}")
        phylo_dict = torch.load(phylo_path, map_location='cpu', weights_only=False)
        print(f"  已成功读取 {len(phylo_dict):,} 条菌株进化向量。")
    else:
        print("[Finetune] Warning: 未指定或未找到 phylo_file，三重门控进化分支以全 0 向量兜底。")

    # 创建 Reader
    fna_reader = FNAReader(config.data.fna_file)
    score_reader = None if config.data.dna_only else ScoreReader(
        config.data.scores_file,
        config.data.scores_bin_file,
    )

    # 创建 Dataset
    train_dataset = FinetuneDataset(
        window_index=train_windows,
        fna_reader=fna_reader,
        score_reader=score_reader,
        taxonomy_db=taxonomy_db,
        window_size_bp=config.data.window_size_bp,
        kmer_size=config.model.kmer_size,
        phylo_dict=phylo_dict,
    )

    val_dataset = None
    if val_windows:
        val_dataset = FinetuneDataset(
            window_index=val_windows,
            fna_reader=fna_reader,
            score_reader=score_reader,
            taxonomy_db=taxonomy_db,
            window_size_bp=config.data.window_size_bp,
            kmer_size=config.model.kmer_size,
            phylo_dict=phylo_dict,
        )

    # DataLoader
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if val_dataset else None
    else:
        if args.fit_diagnostic:
            train_sampler = FixedWindowPassSampler(len(train_dataset), 5, config.seed)
        elif config.training.species_balanced_fraction > 0:
            print('[Sampler] Building species groups from training index...', flush=True)
            train_sampler = MixedSpeciesSampler(
                train_windows, taxonomy_db,
                config.training.steps_per_epoch * config.training.batch_size_per_gpu,
                config.training.species_balanced_fraction, config.seed)
        else:
            train_sampler = RandomSampler(train_dataset)
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size_per_gpu,
        sampler=train_sampler,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=collate_finetune,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size_per_gpu,
        sampler=val_sampler,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=collate_finetune,
    ) if val_dataset else None
    train_eval_loader = DataLoader(
        train_dataset, batch_size=config.training.batch_size_per_gpu,
        num_workers=0, collate_fn=collate_finetune, shuffle=False,
    ) if args.fit_diagnostic else None

    # 创建模型 (微调模式, 加载预训练权重)
    model = build_model(
        config=config.model,
        mode='finetune',
        num_classes_per_level=taxonomy_db.num_classes,
        pretrain_ckpt=args.pretrained,
    )

    # 创建损失函数 (设计文档 §4.1.5)
    loss_fn = SpeciesClassificationLoss(
        classification_only=config.training.classification_loss_only,
        lambda_hier=config.training.lambda_hier,
        lambda_contrast=config.training.lambda_contrast,
        lambda_focus=config.training.lambda_focus,
        delta_focus=config.training.delta_focus,
        tau_contrast=config.training.tau_contrast,
    )
    if local_rank == 0 and config.training.classification_loss_only:
        print('[Loss] Ordinary CE: species weight=1; upper-level weights=lambda_hier * '
              'level_weights; no correctness mask, contrastive or focus loss.', flush=True)

    # 优化器: 差异化学习率
    # 预训练层 (encoder) 用较小 LR, 新增层 (pooling, classifier) 用较大 LR
    pretrained_params = []
    new_params = []
    for name, param in model.named_parameters():
        if any(prefix in name for prefix in ['encoder.', 'embedding.base_embedding.',
                                              'pos_encoding.']):
            pretrained_params.append(param)
        else:
            new_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': pretrained_params, 'lr': config.training.finetune_lr},
        {'params': new_params, 'lr': config.training.finetune_lr * 10},
    ], weight_decay=config.training.finetune_weight_decay)

    # 训练器
    trainer = Trainer(
        model=model,
        config=config.training,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        output_dir=config.output_dir,
        experiment_name=config.experiment_name + '_finetune',
        rank=local_rank,
        world_size=world_size,
        device=device,
    )

    # 学习率调度
    epoch_batches = min(config.training.steps_per_epoch or len(train_loader), len(train_loader))
    accumulation = config.training.gradient_accumulation_steps
    total_steps = ((epoch_batches + accumulation - 1) // accumulation) * config.training.finetune_epochs
    trainer.build_cosine_scheduler(total_steps, config.training.warmup_steps)
    print(f'[Budget] steps/epoch={config.training.steps_per_epoch}; '
          f'batch/GPU={config.training.batch_size_per_gpu}; '
          f'accumulation={config.training.gradient_accumulation_steps}; '
          f'epochs<={config.training.finetune_epochs}; optimizer updates<={total_steps}; '
          f'warmup updates={config.training.warmup_steps}', flush=True)
    if args.fit_diagnostic or config.training.track_species_exposure:
        trainer.species_exposure = {int(s): 0 for s in getattr(train_sampler, 'species_labels', [])}
        trainer.fit_diagnostic_metadata = fit_metadata
        with open(os.path.join(trainer.output_dir, 'species_label_names.json'), 'w', encoding='utf-8') as f:
            json.dump(taxonomy_db.idx2label['species'], f, ensure_ascii=False)

    if local_rank == 0:
        config.to_yaml(os.path.join(trainer.output_dir, 'config.yaml'))

    # 训练
    trainer.fit(
        num_epochs=config.training.finetune_epochs,
        forward_fn=partial(finetune_forward_fn,
                           guidance_dropout=config.training.guidance_dropout,
                           disable_phylo=config.training.disable_phylo),
        metric_key='species_acc',
        save_interval=config.training.save_interval,
        early_stop_patience=config.training.early_stop_patience,
        resume_from=args.resume,
        train_eval_loader=train_eval_loader,
    )

    if local_rank == 0:
        config.to_yaml(os.path.join(trainer.output_dir, 'config.yaml'))

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
