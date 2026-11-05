"""
第二步（前半）：空洞卷积前端（Dilated CNN Stem）

对应设计文档 §2.1.1:
  H_cnn^(l) = GELU(LayerNorm(Conv1D_dilated^(l)(X_final)))

堆叠多层 1D 卷积，膨胀率指数增加（1, 2, 4），
用于捕捉局部微小变异（SNPs/Indels），增强对变异的敏感度。

感受野计算:
  Layer 1 (d=1, k=3): 感受野 = 3
  Layer 2 (d=2, k=3): 感受野 = 3 + 2×2 = 7
  Layer 3 (d=4, k=3): 感受野 = 7 + 2×4 = 15
  总感受野 = 15 个 token (对应 15 + 20 = 35 个碱基位置)
"""

import torch
import torch.nn as nn


class DilatedConvBlock(nn.Module):
    """单个空洞卷积块: Conv1D → LayerNorm → GELU + 残差连接"""

    def __init__(self, d_model: int, kernel_size: int, dilation: int,
                 dropout: float = 0.1):
        super().__init__()
        # 因果 padding: 保持序列长度不变
        padding = dilation * (kernel_size - 1) // 2

        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]
        Returns:
            [B, L, D]  残差连接后的输出
        """
        residual = x

        # Conv1d 需要 [B, D, L] 格式
        h = x.transpose(1, 2)          # [B, D, L]
        h = self.conv(h)               # [B, D, L]
        h = h.transpose(1, 2)          # [B, L, D]

        # LayerNorm + GELU (对应设计文档: GELU(LayerNorm(...)))
        h = self.norm(h)
        h = self.activation(h)
        h = self.dropout(h)

        # 残差连接
        return h + residual


class DilatedCNNStem(nn.Module):
    """
    空洞卷积前端 - 设计文档 §2.1.1

    堆叠多层空洞卷积，膨胀率指数增加:
    - Layer 1: dilation=1, 捕捉相邻碱基模式
    - Layer 2: dilation=2, 捕捉间隔碱基模式
    - Layer 3: dilation=4, 捕捉远距离碱基模式

    目的: 在 Transformer 之前提取多尺度局部特征，
    增强对 SNPs/Indels 等微小变异的敏感度。

    Input:  X_final [B, L, D]  (来自第一步的门控嵌入)
    Output: H_cnn [B, L, D]    (局部特征)
    """

    def __init__(self, d_model: int = 768, kernel_size: int = 3,
                 dilations: tuple = (1, 2, 4), dropout: float = 0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            DilatedConvBlock(d_model, kernel_size, d, dropout)
            for d in dilations
        ])

        # 最终的 LayerNorm
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] 嵌入序列
        Returns:
            H_cnn: [B, L, D] CNN 提取的局部特征
        """
        h = x
        for layer in self.layers:
            h = layer(h)
        return self.final_norm(h)
