"""
第二步（后半）：带唯一性偏置的稀疏注意力（Uniqueness-Biased Sparse Attention）

对应设计文档 §2.1.2-§2.1.5:

  QKV 投影:    Q = H_cnn·W_Q,  K = H_cnn·W_K,  V = H_cnn·W_V
  唯一性偏置:   B_U[i,j] = β · ReLU(W_b · (U_i · U_j))
  注意力:      Attention(Q,K,V) = softmax(QK^T/√d_k + M + B_U) · V
  残差:        H_out = LayerNorm(H_cnn + H_attn)

其中:
  M = 滑动窗口掩码 (Modified Longformer)
  B_U = 唯一性偏置矩阵 (核心创新点)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class SparseAttentionWithUniquenessBias(nn.Module):
    """
    带唯一性偏置的稀疏注意力 - 设计文档 §2.1

    三重创新:
    1. 滑动窗口掩码 M: 限制注意力范围，降低 O(L²) 到 O(L·w)
    2. 唯一性偏置 B_U: 拉近高唯一性位点间的距离 (仅微调时使用)
    3. 多头注意力: H 个独立注意力头

    变量速查表 (设计文档 §2.2):
      H_cnn:  [B, L, D]     CNN 提取的局部特征
      Q,K,V:  [B, H, L, d_k] 多头注意力矩阵
      B_U:    [B, H, L, L]  唯一性偏置矩阵
      β:      Scalar        偏置强度系数
      M:      [L, L]        注意力掩码
      H_out:  [B, L, D]     Encoder 输出的隐状态
    """

    def __init__(self, d_model: int = 768, n_heads: int = 12,
                 window_size: int = 256, dropout: float = 0.1,
                 beta_init: float = 0.5, w_b_init: float = 1.0):
        super().__init__()

        assert d_model % n_heads == 0, f"d_model({d_model}) must be divisible by n_heads({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度
        self.window_size = window_size
        self.scale = math.sqrt(self.d_k)

        # QKV 投影 (设计文档 §2.1.2)
        # W_Q, W_K, W_V ∈ R^{D×D}
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        # 唯一性偏置参数 (设计文档 §2.1.3)
        # β: 偏置强度系数, W_b: 缩放系数
        self.beta = nn.Parameter(torch.tensor(beta_init))
        self.W_b = nn.Parameter(torch.tensor(w_b_init))

        self.attn_dropout = nn.Dropout(dropout)

        # 缓存滑动窗口掩码
        self._cached_mask = None
        self._cached_mask_size = 0

    def _build_sliding_window_mask(self, L: int, device: torch.device) -> torch.Tensor:
        """
        构建滑动窗口掩码 M

        M[i,j] = 0    if |i-j| <= window_size // 2  (窗口内)
        M[i,j] = -inf  otherwise                     (窗口外)

        对应设计文档 §2.1.4: M 是因果掩码或滑动窗口掩码，限制非局部注意力
        """
        if self._cached_mask_size == L and self._cached_mask is not None:
            return self._cached_mask.to(device)

        half_w = self.window_size // 2
        # 创建位置差矩阵
        positions = torch.arange(L, device=device)
        dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()  # [L, L]

        # 窗口外的位置设为 -inf
        mask = torch.zeros(L, L, device=device)
        mask[dist > half_w] = float('-inf')

        self._cached_mask = mask
        self._cached_mask_size = L
        return mask

    def compute_uniqueness_bias(self, uniqueness_scores: Optional[torch.Tensor],
                                L: int) -> Optional[torch.Tensor]:
        """
        计算唯一性偏置矩阵 B_U

        对应设计文档 §2.1.3:
          B_U[i,j] = β · ReLU(W_b · (U_i · U_j))

        含义: U_i·U_j 是位置 i 和 j 的唯一性分数的乘积。
        当两个位置都有高唯一性时 (如特有 SNP)，B_U 值大，
        注意力分数被加强，模型更关注这些位点间的关系。

        Args:
            uniqueness_scores: [B, L] 唯一性评分, 0~1。None 表示不使用
            L: 序列长度

        Returns:
            B_U: [B, 1, L, L] 或 None (broadcast to all heads)
        """
        if uniqueness_scores is None:
            return None

        # U_i · U_j 的外积: [B, L] × [B, L] → [B, L, L]
        U_outer = uniqueness_scores.unsqueeze(-1) * uniqueness_scores.unsqueeze(-2)

        # β · ReLU(W_b · (U_i · U_j))
        B_U = self.beta * F.relu(self.W_b * U_outer)  # [B, L, L]

        # 扩展维度以匹配多头: [B, 1, L, L] (broadcast 到所有头)
        return B_U.unsqueeze(1)

    def forward(self, x: torch.Tensor,
                uniqueness_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] 输入特征 (来自 CNN Stem)
            uniqueness_scores: [B, L] 唯一性评分。None 时不使用偏置 (预训练模式)

        Returns:
            output: [B, L, D] 注意力输出
        """
        B, L, D = x.shape

        # === Step 1: QKV 投影与多头切分 (§2.1.2) ===
        Q = self.W_Q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, L, d_k]
        K = self.W_K(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, L, d_k]
        V = self.W_V(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, L, d_k]

        # === Step 2: 注意力分数 QK^T/√d_k ===
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, L, L]

        # === Step 3: 加入滑动窗口掩码 M (§2.1.4) ===
        mask = self._build_sliding_window_mask(L, x.device)  # [L, L]
        scores = scores + mask.unsqueeze(0).unsqueeze(0)  # broadcast [1, 1, L, L]

        # === Step 4: 加入唯一性偏置 B_U (§2.1.3, 仅微调时) ===
        B_U = self.compute_uniqueness_bias(uniqueness_scores, L)
        if B_U is not None:
            # 只在滑动窗口内应用偏置 (窗口外已被 mask 为 -inf)
            scores = scores + B_U

        # === Step 5: Softmax + 加权求和 ===
        attn_weights = F.softmax(scores, dim=-1)  # [B, H, L, L]
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, V)  # [B, H, L, d_k]

        # === Step 6: 多头合并 + 输出投影 ===
        output = output.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, D]
        output = self.W_O(output)

        return output


class TransformerBlock(nn.Module):
    """
    Pre-LN Transformer Block

    对应设计文档 §2.1.5:
      H_out = LayerNorm(H_cnn + H_attn)

    使用 Pre-LN 变体以提高训练稳定性:
      h = x + Attn(LN(x))
      h = h + FFN(LN(h))
    """

    def __init__(self, d_model: int = 768, n_heads: int = 12,
                 d_ff: int = 3072, window_size: int = 256,
                 dropout: float = 0.1,
                 beta_init: float = 0.5, w_b_init: float = 1.0):
        super().__init__()

        # Pre-LN: 在注意力之前 normalize
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SparseAttentionWithUniquenessBias(
            d_model=d_model,
            n_heads=n_heads,
            window_size=window_size,
            dropout=dropout,
            beta_init=beta_init,
            w_b_init=w_b_init,
        )

        # Pre-LN: 在 FFN 之前 normalize
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor,
                uniqueness_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]
            uniqueness_scores: [B, L] or None

        Returns:
            [B, L, D]
        """
        # Pre-LN Attention + Residual
        h = self.norm1(x)
        h = self.attn(h, uniqueness_scores)
        x = x + h

        # Pre-LN FFN + Residual
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h

        return x
