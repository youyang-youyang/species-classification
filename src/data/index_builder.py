"""
窗口索引构建器

参考 KmerFormer index_builder.py 设计:
1. 扫描 FNA 文件 → 记录每条序列的 accession、字节偏移、碱基数
2. 扫描 scores 文件 → 记录每条序列的评分偏移
3. 将序列切分为固定大小的窗口 (4096bp, stride=2048)
4. 以 accession 为粒度进行 Stratified 划分 (train/val/test)

关键约束 (参考 KmerFormer §11 Q3):
  build_label_encoders() 必须在 build_index() 之前调用!
  否则所有 taxon_labels 会是 -1。
"""

import os
import pickle
import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm

from .fna_reader import SequenceInfo
from .taxonomy import TaxonomyDB


class IndexBuilder:
    """窗口索引构建器"""

    @staticmethod
    def species_group_assignment(accession_species, accession_groups=None,
                                 val_ratio=0.1, test_ratio=0.1, seed=42,
                                 two_group_policy='balanced'):
        """按编码后的 species 分层、按不可拆分来源组分配；与窗口顺序无关。

        提供来源映射时要求覆盖全部 accession，且来源组不能混入多个物种。
        不提供映射时仅保证 accession 隔离，不宣称 assembly/菌株隔离。
        """
        import math
        import random
        if (not math.isfinite(val_ratio) or not math.isfinite(test_ratio)
                or val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1):
            raise ValueError('Ratios must be finite, nonnegative and sum to < 1')
        if two_group_policy not in ('balanced', 'val', 'test'):
            raise ValueError('Unknown two-group policy')
        groups = defaultdict(list)
        group_species = {}
        for acc in sorted(accession_species):
            species = accession_species[acc]
            if species < 0:
                raise ValueError(f'Invalid species label: {acc}')
            group = acc if accession_groups is None else accession_groups.get(acc)
            if not group:
                raise ValueError(f'Missing source group: {acc}')
            if group in group_species and group_species[group] != species:
                raise ValueError(f'Source group has conflicting species: {group}')
            group_species[group] = species
            groups[group].append(acc)
        by_species = defaultdict(list)
        for group, species in group_species.items():
            by_species[species].append(group)
        rng = random.Random(seed)
        assignment = {}
        for species in sorted(by_species):
            candidates = sorted(by_species[species])
            rng.shuffle(candidates)
            n = len(candidates)
            nv = nt = 0
            if n == 2:
                if val_ratio > 0 and test_ratio > 0:
                    policy = two_group_policy
                    if policy == 'balanced':
                        policy = 'val' if rng.random() < val_ratio / (val_ratio + test_ratio) else 'test'
                    nv, nt = (1, 0) if policy == 'val' else (0, 1)
                elif val_ratio > 0:
                    nv = 1
                elif test_ratio > 0:
                    nt = 1
            elif n >= 3:
                nv = max(1, round(n * val_ratio)) if val_ratio > 0 else 0
                nt = max(1, round(n * test_ratio)) if test_ratio > 0 else 0
                while nv + nt > n - 1:
                    if nt > 1:
                        nt -= 1
                    elif nv > 1:
                        nv -= 1
                    else:
                        raise ValueError('Unable to reserve a training group')
            for i, group in enumerate(candidates):
                split = 'val' if i < nv else ('test' if i < nv + nt else 'train')
                for acc in groups[group]:
                    assignment[acc] = split
        return assignment

    def __init__(self, fna_file: str, scores_file: str,
                 taxonomy_db: TaxonomyDB,
                 window_size_bp: int = 4096,
                 window_stride_bp: int = 2048,
                 min_seq_len: int = 4096):
        """
        Args:
            fna_file: FNA 基因组文件路径
            scores_file: 唯一性评分文件路径
            taxonomy_db: 分类数据库 (已调用 build_label_encoders)
            window_size_bp: 窗口大小 (碱基数)
            window_stride_bp: 窗口步长
            min_seq_len: 最小序列长度 (短于此的序列被跳过)
        """
        self.fna_file = fna_file
        self.scores_file = scores_file
        self.taxonomy_db = taxonomy_db
        self.window_size_bp = window_size_bp
        self.window_stride_bp = window_stride_bp
        self.min_seq_len = min_seq_len

    def scan_fna(self) -> List[SequenceInfo]:
        """
        扫描 FNA 文件, 提取每条序列的元数据

        Returns:
            sequences: [SequenceInfo, ...] 所有序列的元信息
        """
        print(f"[IndexBuilder] Scanning FNA: {self.fna_file}")
        sequences = []

        with open(self.fna_file, 'rb') as f:
            current_acc = None
            current_desc = None
            seq_start = 0
            line_len_bytes = 0
            line_len_bases = 0
            num_bases = 0
            first_data_line = True

            pbar = tqdm(desc="Scanning FNA", unit=" lines")
            while True:
                line_bytes = f.readline()
                if not line_bytes:
                    break
                pbar.update(1)

                if line_bytes.startswith(b'>'):
                    # 保存上一条序列
                    if current_acc is not None and num_bases >= self.min_seq_len:
                        sequences.append(SequenceInfo(
                            accession=current_acc,
                            description=current_desc,
                            fna_file=self.fna_file,
                            fna_seq_start=seq_start,
                            line_len_bytes=line_len_bytes,
                            line_len_bases=line_len_bases,
                            num_bases=num_bases,
                            is_variable_length=False,
                        ))

                    # 新序列
                    line_str = line_bytes.decode('latin1').strip()
                    current_desc = line_str
                    current_acc = line_str[1:].split()[0]
                    seq_start = f.tell()  # 下一行的位置就是序列数据开始
                    num_bases = 0
                    first_data_line = True
                else:
                    content = line_bytes.rstrip(b'\r\n')
                    if first_data_line and content:
                        line_len_bases = len(content)
                        line_len_bytes = len(line_bytes)  # 含换行符
                        first_data_line = False
                    num_bases += len(content)

            pbar.close()

            # 最后一条序列
            if current_acc is not None and num_bases >= self.min_seq_len:
                sequences.append(SequenceInfo(
                    accession=current_acc,
                    description=current_desc,
                    fna_file=self.fna_file,
                    fna_seq_start=seq_start,
                    line_len_bytes=line_len_bytes,
                    line_len_bases=line_len_bases,
                    num_bases=num_bases,
                    is_variable_length=False,
                ))

        print(f"[IndexBuilder] Found {len(sequences):,} sequences "
              f"(>= {self.min_seq_len}bp)")
        return sequences

    def scan_scores(self) -> Dict[str, Tuple[int, int]]:
        """
        扫描 scores 文件, 记录每条序列的评分偏移

        Returns:
            score_offsets: {accession: (file_byte_offset, num_scores)}
        """
        print(f"[IndexBuilder] Scanning scores: {self.scores_file}")
        offsets = {}

        if not self.scores_file or not os.path.exists(self.scores_file):
            print(f"[IndexBuilder] Warning: scores file '{self.scores_file}' not found. Skipping score scanning.")
            return offsets

        with open(self.scores_file, 'rb') as f:
            current_acc = None
            data_start = 0
            num_scores = 0

            pbar = tqdm(desc="Scanning scores", unit=" lines")
            while True:
                line_bytes = f.readline()
                if not line_bytes:
                    break
                pbar.update(1)

                if line_bytes.startswith(b'>'):
                    if current_acc is not None:
                        offsets[current_acc] = (data_start, num_scores)

                    line_str = line_bytes.decode('latin1').strip()
                    current_acc = line_str[1:].split()[0]
                    data_start = f.tell()
                    num_scores = 0
                else:
                    line_str = line_bytes.decode('latin1').strip()
                    values = line_str.rstrip(',').split(',')
                    num_scores += len([v for v in values if v.strip()])

            pbar.close()

            if current_acc is not None:
                offsets[current_acc] = (data_start, num_scores)

        print(f"[IndexBuilder] Found scores for {len(offsets):,} sequences")
        return offsets

    def extract_taxid_from_accession(self, accession: str,
                                     acc_to_taxid: Dict[str, int]) -> int:
        """从 accession 查找 TaxID"""
        return acc_to_taxid.get(accession, -1)

    def build_windows(self, sequences: List[SequenceInfo],
                      score_offsets: Dict[str, Tuple[int, int]],
                      acc_to_taxid: Dict[str, int],
                      mode: str = 'pretrain'
                      ) -> List[Tuple]:
        """
        将序列切分为训练窗口

        Args:
            sequences: 序列元信息列表
            score_offsets: {accession: (offset, num_scores)}
            acc_to_taxid: {accession: taxid} 映射
            mode: 'pretrain' 或 'finetune'

        Returns:
            windows: 窗口列表
              pretrain: [(SequenceInfo, window_start_bp), ...]
              finetune: [(SequenceInfo, window_start_bp, score_offset, taxid), ...]
        """
        print(f"[IndexBuilder] Building {mode} windows...")
        windows = []
        skipped_no_score = 0
        skipped_no_taxid = 0

        for seq_info in tqdm(sequences, desc="Building windows"):
            acc = seq_info.accession
            n_bases = seq_info.num_bases

            # 检查 scores
            score_offset = 0
            if mode == 'finetune':
                if acc not in score_offsets:
                    skipped_no_score += 1
                    continue
                val = score_offsets[acc]
                score_offset = val[0] if isinstance(val, (tuple, list)) else val

            # 检查 taxid (微调时需要)
            taxid = -1
            if mode == 'finetune':
                taxid = self.extract_taxid_from_accession(acc, acc_to_taxid)
                if taxid < 0:
                    skipped_no_taxid += 1
                    continue
                # 验证标签是否有效
                labels = self.taxonomy_db.encode_lineage(taxid)
                if labels.get('species', -1) < 0:
                    skipped_no_taxid += 1
                    continue

            # 滑窗切分
            for start in range(0, n_bases - self.window_size_bp + 1,
                               self.window_stride_bp):
                if mode == 'pretrain':
                    windows.append((seq_info, start))
                else:
                    windows.append((seq_info, start, score_offset, taxid))

        print(f"[IndexBuilder] Built {len(windows):,} windows")
        if skipped_no_score:
            print(f"  Skipped (no scores): {skipped_no_score}")
        if skipped_no_taxid:
            print(f"  Skipped (no taxid): {skipped_no_taxid}")

        return windows

    @staticmethod
    def stratified_split(windows: List[Tuple],
                         train_ratio: float = 0.85,
                         val_ratio: float = 0.10,
                         test_ratio: float = 0.05,
                         seed: int = 42) -> Dict[str, List]:
        """
        以 accession 为粒度的 Stratified 划分

        参考 KmerFormer resplit_window_index_stratified.py:
        保证同一条序列的所有窗口在同一个 split 中,
        避免训练/测试数据泄露。

        Returns:
            splits: {'train': [...], 'val': [...], 'test': [...]}
        """
        rng = np.random.default_rng(seed)

        # 按 accession 分组
        acc_groups = defaultdict(list)
        for w in windows:
            acc = w[0].accession
            acc_groups[acc].append(w)

        # 随机打乱 accession 列表
        accessions = list(acc_groups.keys())
        rng.shuffle(accessions)

        n = len(accessions)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_accs = set(accessions[:n_train])
        val_accs = set(accessions[n_train:n_train + n_val])
        test_accs = set(accessions[n_train + n_val:])

        splits = {
            'train': [w for acc in train_accs for w in acc_groups[acc]],
            'val': [w for acc in val_accs for w in acc_groups[acc]],
            'test': [w for acc in test_accs for w in acc_groups[acc]],
        }

        print(f"[IndexBuilder] Split results:")
        print(f"  Train: {len(splits['train']):,} windows ({len(train_accs)} accessions)")
        print(f"  Val:   {len(splits['val']):,} windows ({len(val_accs)} accessions)")
        print(f"  Test:  {len(splits['test']):,} windows ({len(test_accs)} accessions)")

        return splits
