"""
21-mer 无碰撞编码器

将 DNA 碱基序列转换为 21-mer token 序列:
  输入: "ATCGATCG..." (长度 N 的碱基字符串)
  输出: bases_21 array, shape [L, 21]
         L = N - 20 (滑窗步长=1)
         每个元素是碱基编码 (A=0, T=1, C=2, G=3, N=4)
"""

import numpy as np
from typing import Tuple


# 碱基编码映射
BASE_TO_INT = {
    'A': 0, 'a': 0,
    'T': 1, 't': 1,
    'C': 2, 'c': 2,
    'G': 3, 'g': 3,
    'N': 4, 'n': 4,
}

# 预计算的查找表 (ASCII → int), 未知字符映射为 N=4
ASCII_LUT = np.full(256, 4, dtype=np.uint8)
for char, idx in BASE_TO_INT.items():
    ASCII_LUT[ord(char)] = idx


class KmerTokenizer:
    """
    21-mer 无碰撞分词器

    核心设计:
    1. 每个碱基独立编码为整数 (A=0, T=1, C=2, G=3, N=4)
    2. 滑动窗口提取 21-mer: 每个位置产生一个长度 21 的碱基编码数组
    3. 与唯一性评分天然 1:1 对齐, 无需任何 pooling 对齐操作

    与 BPE 的关键区别:
    - BPE: 变长 token, 唯一性评分需要 max pooling 对齐 → 信息损失
    - 21-mer: 固定长度, 与唯一性评分天然对齐 → 零损失
    """

    def __init__(self, kmer_size: int = 21):
        self.kmer_size = kmer_size

    def encode_sequence(self, sequence: str) -> np.ndarray:
        """
        将碱基字符串编码为整数数组

        Args:
            sequence: DNA 碱基序列字符串, 如 "ATCGATCG..."

        Returns:
            encoded: [N] np.uint8, 碱基编码
        """
        # 高性能: 通过 ASCII 查找表批量转换
        seq_bytes = np.frombuffer(sequence.encode('ascii'), dtype=np.uint8)
        encoded = ASCII_LUT[seq_bytes]
        return encoded

    def tokenize(self, sequence: str) -> np.ndarray:
        """
        将碱基序列转换为 21-mer token 序列

        Args:
            sequence: DNA 碱基序列, 长度 N

        Returns:
            bases_21: [L, 21] np.uint8
                L = N - kmer_size + 1
                每行是一条 21-mer 的碱基编码
        """
        encoded = self.encode_sequence(sequence)
        N = len(encoded)

        if N < self.kmer_size:
            raise ValueError(
                f"Sequence length ({N}) < kmer_size ({self.kmer_size})")

        # 滑动窗口视图 (零拷贝, 无额外内存)
        L = N - self.kmer_size + 1
        bases_21 = np.lib.stride_tricks.sliding_window_view(
            encoded, self.kmer_size
        )  # [L, 21]

        # 返回连续内存的副本 (滑动窗口视图不拥有内存)
        return bases_21.copy()

    def tokenize_with_padding(self, sequence: str,
                              target_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tokenize 并 pad 到目标长度

        Args:
            sequence: DNA 碱基序列
            target_length: 目标 token 数 (L)

        Returns:
            bases_21: [target_length, 21] 碱基编码 (padding 位置填 4=N)
            pad_mask: [target_length] bool, True = 有效 token
        """
        bases_21 = self.tokenize(sequence)
        L = len(bases_21)

        pad_mask = np.zeros(target_length, dtype=bool)
        pad_mask[:L] = True

        if L < target_length:
            # Padding: 用 N (=4) 填充
            padded = np.full((target_length, self.kmer_size), 4, dtype=np.uint8)
            padded[:L] = bases_21
            bases_21 = padded
        elif L > target_length:
            # 截断
            bases_21 = bases_21[:target_length]
            pad_mask[:] = True

        return bases_21, pad_mask

    @staticmethod
    def apply_mlm_mask(bases_21: np.ndarray, mask_ratio: float = 0.15,
                       rng: np.random.Generator = None
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        对 21-mer token 序列应用 MLM masking

        策略 (与 BERT 一致):
        - 15% 的 token 被选中
        - 其中 80% 替换为 [MASK] (碱基全设为 N=4)
        - 10% 替换为随机碱基
        - 10% 保持不变

        Args:
            bases_21: [L, 21] 原始碱基编码
            mask_ratio: mask 比例
            rng: 随机数生成器

        Returns:
            masked_bases: [L, 21] mask 后的碱基编码
            mask_positions: [L] bool, True = 被 mask 的位置
            targets: [L, 21] 原始碱基 (用于 loss 计算)
        """
        if rng is None:
            rng = np.random.default_rng()

        L, K = bases_21.shape
        targets = bases_21.copy()
        masked = bases_21.copy()

        # 选择要 mask 的位置 (15%)
        mask_positions = rng.random(L) < mask_ratio

        n_masked = mask_positions.sum()
        if n_masked == 0:
            return masked, mask_positions, targets

        masked_indices = np.where(mask_positions)[0]

        # 对每个 mask 位置决定操作
        ops = rng.random(n_masked)
        for idx, op in zip(masked_indices, ops):
            if op < 0.8:
                # 80%: 替换为 [MASK] (全 N)
                masked[idx] = 4
            elif op < 0.9:
                # 10%: 替换为随机碱基 (0-3)
                masked[idx] = rng.integers(0, 4, size=K).astype(np.uint8)
            # 10%: 保持不变 (什么都不做)

        return masked, mask_positions, targets

    @staticmethod
    def apply_base_level_mlm_mask(
            sequence: str,
            kmer_size: int = 21,
            target_token_len: int = 4076,
            mask_ratio: float = 0.15,
            rng: np.random.Generator = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        碱基级别 MLM Masking —— 修复 stride=1 滑窗重叠导致的捷径学习问题

        与 Token-level Masking (apply_mlm_mask) 的核心区别:
          旧方案: T[i] 全部 21 个位置置 N
                  → 邻居 T[i±1] 各含 T[i] 的 20/21 个原始碱基
                  → 模型从邻居 token 直接"抄写"答案 (Shortcut Learning)

          新方案: 先在原始碱基字符串上按位置随机置 N，再滑窗 tokenize
                  → 被 mask 的碱基 j 在所有含它的 21 个相邻 token 中同时为 N
                  → 任何邻居 token 均无法提供该碱基信息 → 捷径彻底消除
                  → 模型被迫依赖更长程的序列上下文完成预测

        Token mask_positions 定义:
          Token i 对应的 21-mer 中心碱基绝对位置为 i + kmer_size//2 (= i + 10)
          若该中心碱基被选中 mask，则 token i 被标记为 masked token，
          保持约 mask_ratio 比例的 token 级 mask 率（与旧方案一致）。

        损失计算说明:
          每个 masked token 的 21-mer 中有 1 个 N（中心位置），
          其余 20 个位置在 masked_bases 输入中仍然可见。
          MLMLoss 对全部 21 个位置计算 CE loss；
          驱动模型学习 DNA 真实上下文的梯度主要来自中心 N 位置，
          其余 20 个可见位置的损失收敛极快，贡献趋近于零。

        Args:
            sequence:         原始 DNA 碱基序列字符串 (长度为 window_size_bp)
            kmer_size:        k-mer 大小 (默认 21)
            target_token_len: tokenize 后的目标 Token 数 (= window_size_bp - kmer_size + 1)
            mask_ratio:       碱基级别 mask 比例 (默认 0.15)
            rng:              numpy 随机数生成器 (None 时自动创建)

        Returns:
            masked_sequence:      [N] np.uint8, 碱基级 mask 后再 encode 的碱基序列
                                  (与 tokenize_with_padding 配合使用: 先 mask 再 tokenize)
            token_mask_positions: [target_token_len] bool
                                  True = 该 Token 的中心碱基被 mask
            base_masked:          [N] bool, 原始碱基字符串中各位置是否被 mask
                                  (用于调试/分析)
        """
        if rng is None:
            rng = np.random.default_rng()

        n = len(sequence)
        seq_bytes = bytearray(sequence.encode('ascii'))

        # 追踪哪些碱基被选中 mask（用于推导 token 级 mask_positions）
        base_masked = np.zeros(n, dtype=bool)

        # 随机选择 mask_ratio 比例的碱基位置（不重复采样）
        n_to_mask = max(1, int(n * mask_ratio))
        selected = rng.choice(n, size=n_to_mask, replace=False)

        # 对每个选中碱基应用 80/10/10 策略（与 BERT 一致）
        _BASES = [ord('A'), ord('T'), ord('C'), ord('G')]
        ops = rng.random(n_to_mask)
        for j, op in zip(selected, ops):
            if op < 0.8:
                # 80%: 替换为 [MASK] = N
                seq_bytes[j] = ord('N')
            elif op < 0.9:
                # 10%: 替换为随机碱基 (A/T/C/G)
                seq_bytes[j] = _BASES[rng.integers(0, 4)]
            # else 10%: 保持不变（但仍标记为 mask 位置）
            base_masked[j] = True

        masked_sequence = seq_bytes.decode('ascii')

        # 将碱基级 mask 映射为 token 级 mask_positions（以中心碱基为准）
        # token i 的中心碱基在原始序列中的绝对位置为 i + center_offset
        center_offset = kmer_size // 2  # kmer_size=21 时为 10
        n_valid = min(target_token_len, n - kmer_size + 1)
        token_mask_positions = np.zeros(target_token_len, dtype=bool)
        if n_valid > 0:
            center_indices = np.arange(n_valid) + center_offset
            token_mask_positions[:n_valid] = base_masked[center_indices]

        return masked_sequence, token_mask_positions, base_masked
