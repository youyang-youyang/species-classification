"""
预训练 Dataset: 纯序列 MLM

对应设计文档 §5.1 阶段一:
  在大规模无标签基因组上进行掩码语言建模 (MLM)，
  初始化 W_emb 和 Encoder 参数。

数据流:
  FNA 文件 → 读取碱基序列 → 21-mer tokenize → MLM mask → 输出

输出格式:
  masked_bases:   [L, 21]  被 mask 后的 21-mer 碱基编码
  mask_positions: [L]      bool, True = 被 mask 的位置
  targets:        [L, 21]  原始碱基 (loss 目标)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Dict, Any
import pickle
import os

from .tokenizer import KmerTokenizer
from .fna_reader import FNAReader, SequenceInfo


class PretrainDataset(Dataset):
    """
    预训练 Dataset

    核心设计:
    1. 每个样本 = 一个固定大小的碱基窗口 (如 4096bp)
    2. 先在碱基层面随机 mask 15% 的碱基，再滑窗 tokenize 为 21-mer tokens
    3. 同时对原始序列 tokenize 作为 loss targets
    4. 无唯一性评分, 无物种标签

    碱基级 Mask 改进原理（对比旧的 Token-level Mask）:
      旧方案: tokenize → mask token → 邻居 token 含 20/21 相同碱基 → 抄写捷径
      新方案: mask base → tokenize → 被 mask 碱基在所有相邻 token 中同时为 N
              → 邻居 token 无法提供被 mask 碱基的信息 → 捷径彻底消除
              → 迫使模型依赖更长程的序列上下文学习真实 DNA 分布规律
    """

    def __init__(self, window_index: List[Tuple],
                 fna_reader: FNAReader,
                 window_size_bp: int = 4096,
                 kmer_size: int = 21,
                 mask_ratio: float = 0.15,
                 seed: int = 42):
        """
        Args:
            window_index: [(SequenceInfo, window_start_bp), ...] 窗口列表
            fna_reader: FNA 文件读取器
            window_size_bp: 窗口大小 (碱基数)
            kmer_size: k-mer 大小
            mask_ratio: MLM mask 比例
            seed: 随机种子
        """
        self.window_index = window_index
        self.fna_reader = fna_reader
        self.window_size_bp = window_size_bp
        self.kmer_size = kmer_size
        self.mask_ratio = mask_ratio
        self.tokenizer = KmerTokenizer(kmer_size)
        self.target_token_len = window_size_bp - kmer_size + 1  # 4096-21+1=4076

        # 每个 worker 使用不同的随机种子
        self.base_seed = seed

    def __len__(self):
        return len(self.window_index)

    def __getitem__(self, idx: int):
        """
        Returns:
            masked_bases:   [L, 21] int64  碱基级 mask 后再 tokenize 的 21-mer 编码
            mask_positions: [L] bool        True = 中心碱基被 mask 的 Token 位置
            targets:        [L, 21] int64   原始碱基 (loss 目标)
        """
        seq_info, window_start = self.window_index[idx]

        # 1. 读取碱基序列
        sequence = self.fna_reader.read_bases(seq_info, window_start,
                                              self.window_size_bp)

        # 2. 碱基级 MLM masking（先在碱基字符串上 mask，再 tokenize）
        #    修复 stride=1 滑窗重叠导致的捷径学习问题:
        #    被 mask 的碱基在所有包含它的 21 个相邻 Token 中同时为 N，
        #    邻居 Token 无法提供该碱基信息，迫使模型学习真实序列上下文规律。
        rng = np.random.default_rng(self.base_seed + idx)
        masked_sequence, mask_positions, _ = KmerTokenizer.apply_base_level_mlm_mask(
            sequence,
            kmer_size=self.kmer_size,
            target_token_len=self.target_token_len,
            mask_ratio=self.mask_ratio,
            rng=rng,
        )

        # 3. Tokenize 被 mask 后的序列 → 模型输入
        masked_bases, pad_mask = self.tokenizer.tokenize_with_padding(
            masked_sequence, self.target_token_len
        )

        # 4. Tokenize 原始序列 → loss 目标（与 masked_bases 仅在中心碱基位置有差异）
        targets, _ = self.tokenizer.tokenize_with_padding(
            sequence, self.target_token_len
        )

        # 5. 只在有效位置计算 mask（不 mask padding 区域）
        mask_positions = mask_positions & pad_mask

        return {
            'masked_bases': torch.tensor(masked_bases, dtype=torch.long),
            'mask_positions': torch.tensor(mask_positions, dtype=torch.bool),
            'targets': torch.tensor(targets, dtype=torch.long),
        }


class FinetuneDataset(Dataset):
    """
    微调 Dataset: 碱基序列 + 唯一性评分 + 物种标签

    对应设计文档 §5.1 阶段二:
      加载预训练权重，解冻所有参数，
      使用 L_total 在物种分类数据集上进行端到端训练。

    输出格式:
      bases_21:           [L, 21]  21-mer 碱基编码
      uniqueness_scores:  [L]      唯一性评分 (0~1)
      labels:             dict     {level: int} 各级分类标签
    """

    def __init__(self, window_index: List[Tuple],
                 fna_reader: FNAReader,
                 score_reader,
                 taxonomy_db,
                 window_size_bp: int = 4096,
                 kmer_size: int = 21,
                 phylo_dict: Optional[Dict[str, torch.Tensor]] = None,
                 d_phylo: int = 64):
        """
        Args:
            window_index: [(SequenceInfo, window_start_bp, score_offset, taxid), ...]
            fna_reader: FNA 文件读取器
            score_reader: 唯一性评分读取器
            taxonomy_db: 分类数据库
            window_size_bp: 窗口大小
            kmer_size: k-mer 大小
            phylo_dict: {accession: Tensor([64])} GTDB 庞加莱双曲进化坐标字典
            d_phylo: 进化向量维度 (默认 64)
        """
        self.window_index = window_index
        self.fna_reader = fna_reader
        self.score_reader = score_reader
        self.taxonomy_db = taxonomy_db
        self.window_size_bp = window_size_bp
        self.kmer_size = kmer_size
        self.tokenizer = KmerTokenizer(kmer_size)
        self.target_token_len = window_size_bp - kmer_size + 1
        self.phylo_dict = phylo_dict
        self.d_phylo = d_phylo

        # 默认未知进化向量 (全 0 或 'UNKNOWN')
        if phylo_dict is not None and 'UNKNOWN' in phylo_dict:
            self.default_phylo = phylo_dict['UNKNOWN']
            if not isinstance(self.default_phylo, torch.Tensor):
                self.default_phylo = torch.tensor(self.default_phylo, dtype=torch.float32)
        else:
            self.default_phylo = torch.zeros(d_phylo, dtype=torch.float32)

    def __len__(self):
        return len(self.window_index)

    def __getitem__(self, idx: int):
        """
        Returns:
            bases_21: [L, 21] int64
            uniqueness_scores: [L] float32
            labels: dict {level: int64}
            phylo_vecs: [d_phylo] float32 (若提供了 phylo_dict)
        """
        seq_info, window_start, score_offset, taxid = self.window_index[idx]

        # 1. 读取碱基序列
        sequence = self.fna_reader.read_bases(seq_info, window_start,
                                              self.window_size_bp)

        # 2. Tokenize
        bases_21, pad_mask = self.tokenizer.tokenize_with_padding(
            sequence, self.target_token_len
        )

        # 3. 读取唯一性评分
        if self.score_reader is None:
            if score_offset != -1:
                raise ValueError('DNA-only index must explicitly mark absent scores with -1')
            scores = None
        else:
            if score_offset < 0:
                raise ValueError('Scores absent; use the generated DNA-only data configuration')
            scores = self.score_reader.read_scores_binary(
                score_offset + window_start, self.target_token_len)

        # 4. 分类标签
        labels = self.taxonomy_db.encode_lineage(taxid)

        # 转为 tensor
        labels_tensor = {
            level: torch.tensor(label_val, dtype=torch.long)
            for level, label_val in labels.items()
        }

        item = {
            'bases_21': torch.tensor(bases_21, dtype=torch.long),
            'uniqueness_scores': torch.tensor(scores, dtype=torch.float) if scores is not None else None,
            'labels': labels_tensor,
            'pad_mask': torch.tensor(pad_mask, dtype=torch.bool),
        }

        # 5. 读取进化向量 (P_vec，三重门控第二路)
        if self.phylo_dict is not None:
            p_vec = self.phylo_dict.get(seq_info.accession, self.default_phylo)
            if not isinstance(p_vec, torch.Tensor):
                p_vec = torch.tensor(p_vec, dtype=torch.float32)
            item['phylo_vecs'] = p_vec.float()

        return item


def collate_pretrain(batch):
    """预训练 DataLoader 的 collate 函数"""
    return {
        'masked_bases': torch.stack([b['masked_bases'] for b in batch]),
        'mask_positions': torch.stack([b['mask_positions'] for b in batch]),
        'targets': torch.stack([b['targets'] for b in batch]),
    }


def collate_finetune(batch):
    """微调 DataLoader 的 collate 函数"""
    # 碱基和评分
    bases_21 = torch.stack([b['bases_21'] for b in batch])
    scores = [b['uniqueness_scores'] for b in batch]
    if all(s is None for s in scores):
        uniqueness_scores = None
    elif any(s is None for s in scores):
        raise ValueError('Cannot mix scored and unscored windows in a batch')
    else:
        uniqueness_scores = torch.stack(scores)

    # 标签: 每个 level 单独 stack
    levels = batch[0]['labels'].keys()
    labels = {
        level: torch.stack([b['labels'][level] for b in batch])
        for level in levels
    }

    result = {
        'bases_21': bases_21,
        'uniqueness_scores': uniqueness_scores,
        'labels': labels,
    }

    if 'phylo_vecs' in batch[0]:
        result['phylo_vecs'] = torch.stack([b['phylo_vecs'] for b in batch])

    return result
