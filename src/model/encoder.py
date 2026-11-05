"""
第二步完整编码器：混合尺度稀疏编码（Hybrid Encoding）

对应设计文档 §2:
  X_final → DilatedCNNStem → [TransformerBlock × N] → H_out

CNN 前端捕捉局部特征 (SNPs/Indels)
Transformer 建立长程依赖 (跨基因区域的关联)
唯一性偏置 B_U 强化独有基因内部的连续性
"""

import torch
import torch.nn as nn
from typing import Optional

from .cnn_stem import DilatedCNNStem
from .sparse_attention import TransformerBlock


class HybridEncoder(nn.Module):
    """
    混合尺度稀疏编码器 - 设计文档 §2 完整实现

    架构: CNN Stem → Transformer × N_layers

    变量速查表 (§2.2):
      H_cnn:  [B, L, D]     CNN 提取的局部特征
      H_out:  [B, L, D]     Encoder 输出的隐状态
      B_U:    [B, H, L, L]  唯一性偏置矩阵 (在每个 Transformer 层中计算)

    子窗口分治模式 (Phase 1):
      将 L 长序列切分为 M 个子窗口，每个子窗口独立处理:
      - 降低注意力复杂度: O(L²) → O(M × S²), 其中 S = L/M
      - 代价: 子窗口间无直接 token 级交互

    Parameters:
        d_model: 隐藏层维度 (D)
        n_layers: Transformer 层数
        n_heads: 注意力头数 (H)
        d_ff: FFN 维度
        cnn_kernel_size: CNN 卷积核大小
        cnn_dilations: CNN 膨胀率序列
        window_size: 滑动窗口大小
        dropout: Dropout 概率
        use_subwindow: 是否使用子窗口分治
        n_sub_windows: 子窗口数量 (M)
    """

    def __init__(self, d_model: int = 768, n_layers: int = 12,
                 n_heads: int = 12, d_ff: int = 3072,
                 cnn_kernel_size: int = 3, cnn_dilations: tuple = (1, 2, 4),
                 window_size: int = 256, dropout: float = 0.1,
                 beta_init: float = 0.5, w_b_init: float = 1.0,
                 use_subwindow: bool = True, n_sub_windows: int = 4):
        super().__init__()

        self.use_subwindow = use_subwindow
        self.n_sub_windows = n_sub_windows

        # CNN 前端 (§2.1.1)
        self.cnn_stem = DilatedCNNStem(
            d_model=d_model,
            kernel_size=cnn_kernel_size,
            dilations=cnn_dilations,
            dropout=dropout,
        )

        # Transformer 层 (§2.1.2-§2.1.5)
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                window_size=window_size,
                dropout=dropout,
                beta_init=beta_init,
                w_b_init=w_b_init,
            )
            for _ in range(n_layers)
        ])

        # 子窗口分治模式下的跨窗口聚合层
        if use_subwindow:
            # 用 CLS token 聚合各子窗口信息
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.cross_window_attn = TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                window_size=n_sub_windows + 1,  # 全局注意力 (M+1 个 token)
                dropout=dropout,
            )

        # 最终归一化
        self.final_norm = nn.LayerNorm(d_model)

    def _process_subwindows(self, x: torch.Tensor,
                            uniqueness_scores: Optional[torch.Tensor]) -> torch.Tensor:
        """
        子窗口分治处理

        参考 KmerFormer 的两级 Transformer 设计:
        Level-1: 在子窗口内做注意力 (共享参数)
        Level-2: 在 CLS 向量间做跨窗口注意力

        Args:
            x: [B, L, D] 嵌入序列
            uniqueness_scores: [B, L] 或 None

        Returns:
            H_out: [B, L, D]
        """
        B, L, D = x.shape
        M = self.n_sub_windows
        S = L // M  # 每个子窗口的 token 数

        # === Level-1: 子窗口内处理 ===
        # 切分: [B, L, D] → [B, M, S, D] → [B*M, S, D]
        x_sub = x[:, :M * S, :].view(B, M, S, D).reshape(B * M, S, D)

        # 唯一性评分也切分
        u_sub = None
        if uniqueness_scores is not None:
            u_sub = uniqueness_scores[:, :M * S].view(B, M, S).reshape(B * M, S)

        # CNN + Transformer (所有子窗口共享参数)
        h = self.cnn_stem(x_sub)  # [B*M, S, D]
        for layer in self.transformer_layers:
            h = layer(h, u_sub)   # [B*M, S, D]

        # 恢复子窗口维度: [B*M, S, D] → [B, M, S, D]
        h_sub = h.view(B, M, S, D)

        # 提取每个子窗口的 CLS 向量 (取第一个 token 作为摘要)
        # 或使用均值池化
        cls_vecs = h_sub.mean(dim=2)  # [B, M, D]

        # === Level-2: 跨子窗口聚合 ===
        # 添加全局 CLS token
        global_cls = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        cross_input = torch.cat([global_cls, cls_vecs], dim=1)  # [B, M+1, D]

        cross_output = self.cross_window_attn(cross_input)  # [B, M+1, D]

        # 将 Level-2 的信息注入回子窗口特征
        # 每个子窗口的 token 加上对应的跨窗口聚合信息
        enriched = h_sub.clone()
        for m in range(M):
            # cross_output[:, m+1, :] 是第 m 个子窗口的跨窗口表示
            enriched[:, m, :, :] = enriched[:, m, :, :] + cross_output[:, m + 1, :].unsqueeze(1)

        # 合并回原始序列形状: [B, M, S, D] → [B, M*S, D]
        H_out = enriched.view(B, M * S, D)

        # 处理尾部残余 token (如果 L 不能整除 M)
        if L > M * S:
            tail = x[:, M * S:, :]  # [B, tail_len, D]
            tail = self.cnn_stem(tail)
            for layer in self.transformer_layers:
                tail = layer(tail, None)
            H_out = torch.cat([H_out, tail], dim=1)

        return H_out

    def _process_full(self, x: torch.Tensor,
                      uniqueness_scores: Optional[torch.Tensor]) -> torch.Tensor:
        """
        全序列处理 (不分子窗口)

        Args:
            x: [B, L, D]
            uniqueness_scores: [B, L] 或 None

        Returns:
            H_out: [B, L, D]
        """
        h = self.cnn_stem(x)  # [B, L, D]
        for layer in self.transformer_layers:
            h = layer(h, uniqueness_scores)  # [B, L, D]
        return h

    def forward(self, x: torch.Tensor,
                uniqueness_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] 嵌入序列 (来自第一步)
            uniqueness_scores: [B, L] 唯一性评分。None = 预训练模式

        Returns:
            H_out: [B, L, D] 编码器输出
        """
        if self.use_subwindow:
            H_out = self._process_subwindows(x, uniqueness_scores)
        else:
            H_out = self._process_full(x, uniqueness_scores)

        return self.final_norm(H_out)
