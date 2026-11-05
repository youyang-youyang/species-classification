"""
预训练模型规范评测脚本 (evaluate_pretrain.py)

按照业界大模型评测范式，对预训练模型进行三大维度的规范评测：
1. 【准确率与泛化】: 在未泄露的测试集 (Test Split) 上评估 MLM 准确率与 Loss。
2. 【计算吞吐与效率】: 测试吞吐量 (seq/s, bp/s) 以及显存占用 (VRAM Profile)。
3. 【零样本表征诊断】: 计算表征空间余弦相似度分布及有效秩 (RankMe)，评估表征退化与多样性。

用法:
  python scripts/evaluate_pretrain.py \
      --config configs/pretrain_bacteria.yaml \
      --checkpoint runs/bacteria_species_cls_v1_pretrain/best_model.pt \
      --samples 20000 \
      --batch-size 16
"""

import sys
import os
import time
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.model.model import build_model
from src.data.dataset import PretrainDataset, collate_pretrain
from src.data.fna_reader import FNAReader


def compute_rankme(embeddings: torch.Tensor) -> float:
    """
    计算表征的有效秩 (RankMe) - 业界用于评估表征坍塌 (Representation Collapse) 的标准无监督指标。
    RankMe = exp( - sum (p_i * ln(p_i)) )，其中 p_i 是奇异值归一化后的概率分布。
    """
    if embeddings.size(0) < 2:
        return 0.0
    emb = embeddings.double()
    # 奇异值分解
    _, S, _ = torch.linalg.svd(emb, full_matrices=False)
    S_sum = S.sum()
    if S_sum == 0:
        return 0.0
    p = S / S_sum
    # 避免 log(0)
    entropy = -torch.sum(p * torch.log(p + 1e-15))
    rankme = torch.exp(entropy).item()
    return rankme


@torch.no_grad()
def evaluate_mlm_accuracy(model, dataloader, device, max_batches=None):
    """评测 MLM 泛化准确率和 Loss"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_masked_tokens = 0
    batches_evaluated = 0

    desc = "Evaluating MLM"
    for batch in tqdm(dataloader, desc=desc, total=max_batches):
        masked_bases = batch['masked_bases'].to(device)
        mask_positions = batch['mask_positions'].to(device)
        targets = batch['targets'].to(device)

        m = model.module if hasattr(model, 'module') else model
        # 前向计算
        x = m.embedding(masked_bases)
        x = m.pos_encoding(x)
        H = m.encoder(x, uniqueness_scores=None)
        mlm_logits = m.mlm_head(H, mask_positions)
        targets_flat = targets[mask_positions].reshape(-1).long()

        valid = targets_flat < 4  # 排除 N 碱基
        if valid.sum() == 0:
            continue

        logits_flat = mlm_logits.reshape(-1, 5)
        loss = nn.functional.cross_entropy(logits_flat[valid], targets_flat[valid])

        total_loss += loss.item()
        preds = logits_flat[valid].argmax(dim=-1)
        total_correct += (preds == targets_flat[valid]).sum().item()
        total_masked_tokens += valid.sum().item()

        batches_evaluated += 1
        if max_batches and batches_evaluated >= max_batches:
            break

    avg_loss = total_loss / batches_evaluated if batches_evaluated > 0 else 0.0
    avg_acc = total_correct / total_masked_tokens if total_masked_tokens > 0 else 0.0

    return avg_loss, avg_acc


@torch.no_grad()
def benchmark_efficiency(model, dataloader, device, warmup_steps=10, benchmark_steps=20):
    """评测硬件效率、吞吐率和显存占用"""
    model.eval()
    
    # 模拟数据
    batch = next(iter(dataloader))
    masked_bases = batch['masked_bases'].to(device)
    mask_positions = batch['mask_positions'].to(device)
    targets = batch['targets'].to(device)

    # 显存清理
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        start_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)

    # Warmup
    for _ in range(warmup_steps):
        m = model.module if hasattr(model, 'module') else model
        x = m.embedding(masked_bases)
        x = m.pos_encoding(x)
        H = m.encoder(x, uniqueness_scores=None)
        _ = m.mlm_head(H, mask_positions)

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

    # 实际计时
    start_time = time.perf_counter()
    for _ in range(benchmark_steps):
        m = model.module if hasattr(model, 'module') else model
        x = m.embedding(masked_bases)
        x = m.pos_encoding(x)
        H = m.encoder(x, uniqueness_scores=None)
        _ = m.mlm_head(H, mask_positions)

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    end_time = time.perf_counter()

    # 指标计算
    total_time = end_time - start_time
    avg_step_time_ms = (total_time / benchmark_steps) * 1000
    
    batch_size = masked_bases.size(0)
    seq_len = masked_bases.size(1)
    
    throughput_seq = (batch_size * benchmark_steps) / total_time
    throughput_bp = (batch_size * seq_len * benchmark_steps) / total_time

    peak_memory = 0.0
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return {
        'avg_step_time_ms': avg_step_time_ms,
        'throughput_seq_per_sec': throughput_seq,
        'throughput_bp_per_sec': throughput_bp,
        'peak_vram_mb': peak_memory,
        'batch_size': batch_size,
        'seq_len': seq_len
    }


@torch.no_grad()
def evaluate_representation_geometry(model, dataloader, device, num_batches=10):
    """零样本表征几何诊断：计算表示秩 (RankMe) 与余弦相似度分布"""
    model.eval()
    all_embeddings = []

    batches_processed = 0
    for batch in dataloader:
        masked_bases = batch['masked_bases'].to(device)
        m = model.module if hasattr(model, 'module') else model
        
        # 提取表征 (Encoder 最后一层输出的序列平均池化)
        x = m.embedding(masked_bases)
        x = m.pos_encoding(x)
        H = m.encoder(x, uniqueness_scores=None) # [B, L, D]
        seq_emb = H.mean(dim=1)  # [B, D]
        
        all_embeddings.append(seq_emb.cpu())
        batches_processed += 1
        if batches_processed >= num_batches:
            break

    embeddings = torch.cat(all_embeddings, dim=0) # [N, D]
    
    # 1. 计算 RankMe
    rankme_score = compute_rankme(embeddings)
    
    # 2. 计算两两余弦相似度
    emb_norm = nn.functional.normalize(embeddings, p=2, dim=-1)
    cos_sim = torch.matmul(emb_norm, emb_norm.T) # [N, N]
    
    # 提取上三角元素 (排除对角线自身相似度)
    n = cos_sim.size(0)
    indices = torch.triu_indices(n, n, offset=1)
    flat_sims = cos_sim[indices[0], indices[1]].numpy()
    
    cos_mean = np.mean(flat_sims)
    cos_std = np.std(flat_sims)
    cos_min = np.min(flat_sims)
    cos_max = np.max(flat_sims)
    
    return {
        'rankme': rankme_score,
        'num_samples': n,
        'embedding_dim': embeddings.size(1),
        'cosine_sim_mean': cos_mean,
        'cosine_sim_std': cos_std,
        'cosine_sim_min': cos_min,
        'cosine_sim_max': cos_max
    }


def main():
    parser = argparse.ArgumentParser(description='Genomic LM Pretrain Standard Benchmarking')
    parser.add_argument('--config', type=str, required=True, help='Path to pretrain YAML config')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to pretrained checkpoint')
    parser.add_argument('--gpus', type=str, default='0', help='GPU ID to use')
    parser.add_argument('--samples', type=int, default=20000, help='Max test samples to evaluate for MLM')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for evaluation')
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    device = torch.device(f'cuda:{args.gpus}' if torch.cuda.is_available() else 'cpu')

    print(f"\n========================================================")
    print(f"📖 Loading Config & Index Data...")
    print(f"========================================================")
    
    # 加载窗口索引
    index_path = os.path.join(config.data.index_dir, 'window_index_pretrain.pkl')
    with open(index_path, 'rb') as f:
        index_data = pickle.load(f)
    
    test_windows = index_data.get('test', [])
    if not test_windows:
        print("[Warning] No 'test' split found in index. Using 'val' split instead.")
        test_windows = index_data.get('val', [])
        
    print(f"Loaded {len(test_windows):,} test windows.")

    fna_reader = FNAReader(config.data.fna_file)
    dataset = PretrainDataset(
        window_index=test_windows,
        fna_reader=fna_reader,
        window_size_bp=config.data.window_size_bp,
        kmer_size=config.model.kmer_size,
        mask_ratio=config.training.mlm_mask_ratio,
        seed=config.seed + 2026,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(dataset),
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=collate_pretrain,
    )

    # 模型加载
    print(f"\n⚙️ Building Model and Loading Checkpoint...")
    model = build_model(config.model, mode='pretrain')
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state', checkpoint)
    # 移除 DDP 前缀
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    print(f"Successfully loaded checkpoint: {args.checkpoint}")

    # ========================================================
    # 评测 1: 硬件吞吐量和效率 (Efficiency)
    # ========================================================
    print(f"\n========================================================")
    print(f"📊 1. Benchmarking Hardware Efficiency...")
    print(f"========================================================")
    eff_results = benchmark_efficiency(model, dataloader, device)
    print(f"  - Batch Size:          {eff_results['batch_size']}")
    print(f"  - Sequence Length:     {eff_results['seq_len']} bp")
    print(f"  - Avg Step Latency:    {eff_results['avg_step_time_ms']:.2f} ms")
    print(f"  - Throughput (Seqs/s): {eff_results['throughput_seq_per_sec']:.2f} seq/s")
    print(f"  - Throughput (Bps/s):  {eff_results['throughput_bp_per_sec'] / 1e6:.2f} Mbp/s")
    if torch.cuda.is_available():
        print(f"  - Peak VRAM Usage:     {eff_results['peak_vram_mb']:.1f} MB")
    else:
        print(f"  - Peak VRAM Usage:     N/A (CPU Mode)")

    # ========================================================
    # 评测 2: 零样本表征几何诊断 (Representation Geometry)
    # ========================================================
    print(f"\n========================================================")
    print(f"📊 2. Diagnostics of Representational Geometry...")
    print(f"========================================================")
    # 提取大约 1000 个样本评估空间分布
    rep_results = evaluate_representation_geometry(model, dataloader, device, num_batches=1000 // args.batch_size)
    print(f"  - Evaluated Samples:   {rep_results['num_samples']}")
    print(f"  - Embedding Dimension: {rep_results['embedding_dim']}")
    print(f"  - RankMe Score:        {rep_results['rankme']:.2f} / {min(rep_results['num_samples'], rep_results['embedding_dim'])}")
    print(f"    (说明: 越接近维数 {rep_results['embedding_dim']} 表征的多样性与空间利用率越高，防坍塌能力越好)")
    print(f"  - Cosine Similarity:  Mean: {rep_results['cosine_sim_mean']:.4f} | Std: {rep_results['cosine_sim_std']:.4f} | Range: [{rep_results['cosine_sim_min']:.4f}, {rep_results['cosine_sim_max']:.4f}]")
    print(f"    (说明: 平均相似度越适中说明表征区分度越强；如果接近 1.0 说明发生了表征坍塌)")

    # ========================================================
    # 评测 3: 测试集 MLM 泛化准确率 (Generalization Accuracy)
    # ========================================================
    print(f"\n========================================================")
    print(f"📊 3. Evaluating MLM Accuracy on Test Set...")
    print(f"========================================================")
    max_batches = args.samples // args.batch_size
    test_loss, test_acc = evaluate_mlm_accuracy(model, dataloader, device, max_batches=max_batches)
    print(f"\n[Final Results on Test Set]")
    print(f"  - Evaluated Samples:   {min(args.samples, len(test_windows)):,}")
    print(f"  - Test MLM Loss:       {test_loss:.5f}")
    print(f"  - Test MLM Accuracy:   {test_acc * 100:.2f}%")
    print(f"========================================================\n")


if __name__ == '__main__':
    main()
