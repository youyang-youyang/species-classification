"""
第三步：唯一性加权多尺度最大响应池化（Uniqueness-Weighted MS-MaxPool）

对应设计文档 §3:

  1. 唯一性调制:    Score(H_i, U_i) = H_i + γ · U_i
  2. 并行多尺度池化:
     - 细粒度 (SNPs):  F_3  = MaxPool_{k=3,s=1}(Score)
     - 中粒度 (Motifs): F_10 = MaxPool_{k=10,s=2}(Score)
     - 粗粒度 (Genes):  F_50 = MaxPool_{k=50,s=10}(Score)
     - 全局注意力:      F_attn = AttentionPool(H_out)  (不受唯一性调制)
  3. 特征聚合:
     - 展平 → 拼接 → 投影
     - Z = GELU(F_concat · W_z + b_z)   → 基因组指纹

目的: 从冗长的序列中提取最具判别力的"基因组指纹"，
确保唯一性高的特征在池化过程中不被平均池化淹没。

变量速查表 (§3.2):
  Score:    [B, L, D]     调制后的特征
  γ:        Scalar        唯一性奖励系数
  F_k:      Varied        各尺度池化结果
  F_concat: [B, D_total]  拼接后的总特征
  Z:        [B, D']       基因组指纹，模型的终极特征表示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AttentionPool(nn.Module):
    """
    注意力池化 - 设计文档 §3.1 中的全局注意力路径

    不受唯一性调制影响，保留全局语义。
    使用可学习的 query 向量对序列做注意力加权求和。

    Input:  H [B, L, D]
    Output: F_attn [B, D]
    """

    def __init__(self, d_model: int = 768):
        super().__init__()
        # 可学习的 attention query
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.key_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** 0.5

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: [B, L, D] 原始编码器输出 (不经唯一性调制)
        Returns:
            F_attn: [B, D] 全局注意力池化特征
        """
        B, L, D = H.shape

        # query: [1, 1, D] → broadcast → [B, 1, D]
        query = self.query.expand(B, -1, -1)
        key = self.key_proj(H)  # [B, L, D]

        # 注意力分数: [B, 1, D] × [B, D, L] → [B, 1, L]
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, 1, L]

        # 加权求和: [B, 1, L] × [B, L, D] → [B, 1, D] → [B, D]
        F_attn = torch.matmul(attn_weights, H).squeeze(1)

        return F_attn


class MultiScaleMaxPool(nn.Module):
    """
    多尺度最大响应池化 - 设计文档 §3.1 中的并行池化路径

    对 Score = H + γ·U 执行不同尺度的 MaxPool:
    - k=3, s=1:  捕捉 SNP 级别的细粒度变异
    - k=10, s=2: 捕捉 Motif 级别的中粒度模式
    - k=50, s=10: 捕捉 Gene 级别的粗粒度结构
    """

    def __init__(self, pool_kernels: Tuple[int, ...] = (3, 10, 50),
                 pool_strides: Tuple[int, ...] = (1, 2, 10)):
        super().__init__()
        self.pools = nn.ModuleList([
            nn.MaxPool1d(kernel_size=k, stride=s, padding=k // 2)
            for k, s in zip(pool_kernels, pool_strides)
        ])

    def forward(self, score: torch.Tensor) -> list:
        """
        Args:
            score: [B, L, D] 唯一性调制后的特征

        Returns:
            pooled_features: list of [B, D] tensors (每个尺度一个)
        """
        # Conv1d/MaxPool1d 需要 [B, D, L] 格式
        score_t = score.transpose(1, 2)  # [B, D, L]

        pooled = []
        for pool in self.pools:
            f = pool(score_t)             # [B, D, L_k]
            # 取全局最大值: [B, D, L_k] → [B, D]
            f_max = f.max(dim=-1).values  # [B, D]
            pooled.append(f_max)

        return pooled


class UniquenessMSMaxPool(nn.Module):
    """
    唯一性加权多尺度最大响应池化 - 设计文档 §3 完整实现

    核心公式:
      Score(H_i, U_i) = H_i + γ · U_i                    (唯一性调制)
      F_k = MaxPool_k(Score)                               (多尺度池化)
      F_attn = AttentionPool(H)                            (全局语义)
      F_concat = Concat(Flat(F_3), ..., Flat(F_attn))      (特征聚合)
      Z = GELU(F_concat · W_z + b_z)                      (投影为基因组指纹)

    Input:
      H: [B, L, D]     编码器输出
      U: [B, L]        唯一性评分 (微调时), None (预训练时)

    Output:
      Z: [B, D']       基因组指纹 (D' 通常 = D)
    """

    def __init__(self, d_model: int = 768, d_fingerprint: int = 768,
                 gamma: float = 2.0,
                 pool_kernels: Tuple[int, ...] = (3, 10, 50),
                 pool_strides: Tuple[int, ...] = (1, 2, 10)):
        super().__init__()

        self.gamma = nn.Parameter(torch.tensor(gamma))  # γ: 可学习的奖励系数
        self.d_model = d_model

        # 多尺度 MaxPool (§3.1 步骤2)
        self.ms_maxpool = MultiScaleMaxPool(pool_kernels, pool_strides)

        # 注意力池化 (§3.1 步骤2, 全局语义路径)
        self.attn_pool = AttentionPool(d_model)

        # 特征投影 (§3.1 步骤3)
        # 总特征维度: len(pool_kernels) × D + D (注意力池化)
        n_paths = len(pool_kernels) + 1  # +1 for attention pool
        total_dim = n_paths * d_model
        self.projection = nn.Sequential(
            nn.Linear(total_dim, d_fingerprint),
            nn.GELU(),
            nn.LayerNorm(d_fingerprint),
        )

    def forward(self, H: torch.Tensor,
                uniqueness_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            H: [B, L, D] 编码器输出
            uniqueness_scores: [B, L] 唯一性评分 (0~1)。None 时不使用调制

        Returns:
            Z: [B, D'] 基因组指纹
        """
        # === Step 1: 唯一性调制得分 (§3.1 步骤1) ===
        # Score(H_i, U_i) = H_i + γ · U_i
        if uniqueness_scores is not None:
            Score = H + self.gamma * uniqueness_scores.unsqueeze(-1)  # [B, L, D]
        else:
            Score = H  # 预训练时不调制

        # === Step 2: 并行多尺度池化 (§3.1 步骤2) ===
        # 多尺度 MaxPool (受唯一性调制)
        pooled_features = self.ms_maxpool(Score)  # list of [B, D]

        # 注意力池化 (不受唯一性调制，保留全局语义)
        F_attn = self.attn_pool(H)  # [B, D]
        pooled_features.append(F_attn)

        # === Step 3: 特征聚合与投影 (§3.1 步骤3) ===
        # 拼接: [B, n_paths × D]
        F_concat = torch.cat(pooled_features, dim=-1)  # [B, 4D]

        # 投影为基因组指纹: [B, 4D] → [B, D']
        Z = self.projection(F_concat)  # [B, D']

        return Z
