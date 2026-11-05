"""
第一步：唯一性感知的三重门控嵌入（Uniqueness-Aware Triple-Gated Embedding）

对应专利文档权利要求 3 / 具体实施方式三:
  三重门控融合公式:
    X_phylo = P_vec · W_p + b_p               进化向量投影
    X_U     = U · W_u + b_u                   唯一性特征投影
    g_base  = σ(W_1[X_base ; X_U] + b_1)     碱基门控 (以唯一性为条件)
    g_phylo = σ(W_2[X_phylo; X_U] + b_2)     进化门控 (以唯一性为条件)
    X_final = g_base ⊙ X_base
            + g_phylo ⊙ X_phylo
            + (1 - g_base - g_phylo) ⊙ X_U   唯一性残差融合

核心公式 (设计文档 §1.2):
  碱基嵌入:    X_base = BaseEmbedding(T)
  唯一性投影:  X_U = U·W_u + b_u
  进化投影:    X_phylo = P_vec·W_p + b_p
  双门控:      g_base = σ(W_1[X_base; X_U] + b_1)
               g_phylo = σ(W_2[X_phylo; X_U] + b_2)
  三重融合:    X_final = g_base ⊙ X_base + g_phylo ⊙ X_phylo + (1-g_base-g_phylo) ⊙ X_U
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Dict


class BaseEmbedding(nn.Module):
    """
    KmerFormer 式无碰撞 21-mer 编码

    核心思想 (参考 KmerFormer 技术文档 §6.3.1):
    - base_embed: 5 个碱基 (A,T,C,G,N) → D 维嵌入
    - kmer_pos_embed: 21 个位置 → D 维位置编码
    - kmer_vec[i] = Σ_{j=0}^{20} (base_embed(seq[i,j]) + kmer_pos_embed(j))

    参数量: 仅 5×D + 21×D = 26×D (D=768 时 ≈ 20K 参数)
    覆盖: 4^21 ≈ 4.4 万亿种 21-mer，零碰撞

    Input:  bases_21 [B, L, 21], 碱基编码 (0=A, 1=T, 2=C, 3=G, 4=N)
    Output: X_base [B, L, D]
    """

    def __init__(self, d_model: int = 768, kmer_size: int = 21,
                 base_vocab_size: int = 5):
        super().__init__()
        self.d_model = d_model
        self.kmer_size = kmer_size

        # 碱基嵌入: 5 种碱基 → D 维
        self.base_embed = nn.Embedding(base_vocab_size, d_model)

        # K-mer 内位置编码: 21 个位置 → D 维
        self.kmer_pos_embed = nn.Embedding(kmer_size, d_model)

        # 缩放因子 (类似 Transformer 的 √D 缩放)
        self.scale = math.sqrt(d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.base_embed.weight, std=0.02)
        nn.init.normal_(self.kmer_pos_embed.weight, std=0.02)

    def forward(self, bases_21: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bases_21: [B, L, 21] 碱基编码序列, 每个位置是一条 21-mer

        Returns:
            X_base: [B, L, D] 碱基嵌入
        """
        B, L, K = bases_21.shape

        # 碱基嵌入: [B, L, 21] → [B, L, 21, D]
        base_vecs = self.base_embed(bases_21)

        # K-mer 内位置编码: [21] → [21, D] → broadcast → [B, L, 21, D]
        pos_ids = torch.arange(K, device=bases_21.device)
        kp_vecs = self.kmer_pos_embed(pos_ids)  # [21, D]

        # 逐位置相加后沿 21-mer 轴求和: [B, L, 21, D] → [B, L, D]
        kmer_vec = (base_vecs + kp_vecs).sum(dim=2)  # [B, L, D]

        # 缩放 (防止求和后数值过大)
        kmer_vec = kmer_vec / self.scale

        return kmer_vec


class TripleGatedEmbedding(nn.Module):
    """
    三重门控嵌入 —— 对应专利权利要求 3 / 具体实施方式三

    完整实现三重门控融合，融合三路特征:
      路径 1: X_base  —— 碱基序列嵌入 (k-mer 无碰撞编码)
      路径 2: X_phylo —— GTDB 系统发育树 64 维庞加莱双曲进化向量的线性投影
      路径 3: X_U     —— 泛基因组唯一性评分的线性投影

    门控公式 (专利 §3):
      g_base  = σ( W_1[X_base ; X_U] + b_1 )    碱基门控 (W_1 ∈ R^{2D×D})
      g_phylo = σ( W_2[X_phylo; X_U] + b_2 )    进化门控 (W_2 ∈ R^{2D×D})
      X_final = g_base ⊙ X_base
              + g_phylo ⊙ X_phylo
              + (1 - g_base - g_phylo) ⊙ X_U

    特性:
      - 当 U_i → 1 (高唯一性位点): g_base → 0, g_phylo → 0, 由 X_U 主导
      - 当 U_i → 0 (保守位点):    g_base → 1, g_phylo ↑,  由序列与进化信息联合主导
      - 零数据库退化模式: 将 U 全置 0, phylo_vec 全置 None → 退化为纯碱基模式

    Input:
      bases_21:    [B, L, 21]      碱基编码
      uniqueness:  [B, L]          唯一性评分 (0~1)
      phylo_vecs:  [B, L, d_phylo] 逐 Token 的进化坐标向量 (可选, None 时视作全 0)
    Output:
      X_final: [B, L, D]           三重门控融合后的嵌入
    """

    def __init__(self, d_model: int = 768, kmer_size: int = 21,
                 base_vocab_size: int = 5, d_phylo: int = 64):
        """
        Args:
            d_model:        Transformer 嵌入维度 D (默认 768)
            kmer_size:      k-mer 大小 (默认 21)
            base_vocab_size: 碱基词表大小 (默认 5: A/T/C/G/N)
            d_phylo:        进化向量维度 D_p (默认 64, 来自庞加莱双曲嵌入)
        """
        super().__init__()
        self.d_model = d_model
        self.d_phylo = d_phylo

        # ── 路径 1: 碱基嵌入 (共享 BaseEmbedding 结构) ──────────────────────
        self.base_embedding = BaseEmbedding(d_model, kmer_size, base_vocab_size)

        # ── 路径 2: 进化向量投影 ─────────────────────────────────────────────
        # X_phylo = P_vec · W_p + b_p    (W_p ∈ R^{d_phylo × D})
        # 对应专利公式: X_phylo = P_vec · W_p + b_p
        self.phylo_proj = nn.Linear(d_phylo, d_model)

        # ── 路径 3: 唯一性特征投影 ───────────────────────────────────────────
        # X_U = U · W_u + b_u    (W_u ∈ R^{1 × D})
        self.uniqueness_proj = nn.Linear(1, d_model)

        # ── 门控网络 1: 碱基门控 ─────────────────────────────────────────────
        # g_base = σ(W_1[X_base; X_U] + b_1)    W_1 ∈ R^{2D × D}
        self.gate_base = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid()
        )

        # ── 门控网络 2: 进化门控 ─────────────────────────────────────────────
        # g_phylo = σ(W_2[X_phylo; X_U] + b_2)   W_2 ∈ R^{2D × D}
        self.gate_phylo = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        """
        门控偏置初始化为 +0.5，使初始 g_base ≈ g_phylo ≈ 0.62
        (Sigmoid(0.5) ≈ 0.622)
        保证预训练碱基表征向微调模型的平滑迁移 (专利权利要求 2)
        """
        # 碱基门控
        nn.init.xavier_uniform_(self.gate_base[0].weight)
        nn.init.constant_(self.gate_base[0].bias, 0.5)

        # 进化门控 (偏置稍小，使进化路径初始贡献较碱基路径小)
        nn.init.xavier_uniform_(self.gate_phylo[0].weight)
        nn.init.constant_(self.gate_phylo[0].bias, 0.5)

        # 进化投影
        nn.init.xavier_uniform_(self.phylo_proj.weight)
        nn.init.zeros_(self.phylo_proj.bias)

        # 唯一性投影
        nn.init.xavier_uniform_(self.uniqueness_proj.weight)
        nn.init.zeros_(self.uniqueness_proj.bias)

    def forward(self,
                bases_21: torch.Tensor,
                uniqueness_scores: Optional[torch.Tensor] = None,
                phylo_vecs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        三重门控前向传播

        Args:
            bases_21:         [B, L, 21]      碱基编码 (int64)
            uniqueness_scores:[B, L]           唯一性评分 0~1 (None→全 0)
            phylo_vecs:       [B, L, d_phylo]  进化坐标向量 (None→全 0，退化为双路门控)

        Returns:
            X_final: [B, L, D]  三重门控融合特征
        """
        B, L = bases_21.shape[:2]
        device = bases_21.device

        # ── 兜底: 缺省参数时全置 0 ──────────────────────────────────────────
        if uniqueness_scores is None:
            uniqueness_scores = torch.zeros(B, L, device=device, dtype=torch.float32)
        if phylo_vecs is None:
            phylo_vecs = torch.zeros(B, L, self.d_phylo, device=device, dtype=torch.float32)
        elif phylo_vecs.dim() == 2:
            # [B, d_phylo] -> [B, L, d_phylo] 沿序列长度广播
            phylo_vecs = phylo_vecs.unsqueeze(1).expand(-1, L, -1)

        # ── Step 1: 碱基嵌入 ─────────────────────────────────────────────────
        # X_base = BaseEmbedding(T)    [B, L, D]
        X_base = self.base_embedding(bases_21)

        # ── Step 2: 进化向量投影 ──────────────────────────────────────────────
        # X_phylo = P_vec · W_p + b_p    [B, L, d_phylo] → [B, L, D]
        X_phylo = self.phylo_proj(phylo_vecs)

        # ── Step 3: 唯一性特征投影 ────────────────────────────────────────────
        # X_U = U · W_u + b_u    [B, L, 1] → [B, L, D]
        X_U = self.uniqueness_proj(uniqueness_scores.unsqueeze(-1))

        # ── Step 4: 双门控计算 ────────────────────────────────────────────────
        # g_base  = σ(W_1[X_base;  X_U] + b_1)    [B, L, D]
        # g_phylo = σ(W_2[X_phylo; X_U] + b_2)    [B, L, D]
        g_base  = self.gate_base( torch.cat([X_base,  X_U], dim=-1))  # [B, L, D]
        g_phylo = self.gate_phylo(torch.cat([X_phylo, X_U], dim=-1))  # [B, L, D]

        # ── Step 5: 三重门控融合 ──────────────────────────────────────────────
        # X_final = g_base ⊙ X_base + g_phylo ⊙ X_phylo + (1 - g_base - g_phylo) ⊙ X_U
        # 专利公式: 当 U_i → 1 时, g_base → 0, g_phylo → 0, X_U 主导输出
        X_final = (g_base  * X_base
                 + g_phylo * X_phylo
                 + (1.0 - g_base - g_phylo) * X_U)  # [B, L, D]

        return X_final


class DualGatedEmbedding(nn.Module):
    """
    双路门控嵌入 —— 向后兼容别名 (Phase 1，无系统发育向量)

    完整三重门控实现请使用 TripleGatedEmbedding。
    本类保留用于:
    1. 旧版微调代码的向后兼容
    2. 无进化向量数据时的简化模式

    对应设计文档公式:
      X_base = BaseEmbedding(T)
      X_U = U · W_u + b_u
      g_base = σ(W_1[X_base; X_U] + b_1)
      X_final = g_base ⊙ X_base + (1 - g_base) ⊙ X_U

    当 U_i → 1 (高唯一性) 时, g_base → 0, 输出由 X_U 主导
    当 U_i → 0 (低唯一性) 时, g_base → 1, 输出由 X_base 主导

    Input:
      bases_21: [B, L, 21]   碱基编码
      uniqueness: [B, L]     唯一性评分 (0~1)
    Output:
      X_final: [B, L, D]    融合后的嵌入
    """

    def __init__(self, d_model: int = 768, kmer_size: int = 21,
                 base_vocab_size: int = 5):
        super().__init__()
        self.d_model = d_model

        # 碱基嵌入 (共享 BaseEmbedding)
        self.base_embedding = BaseEmbedding(d_model, kmer_size, base_vocab_size)

        # 唯一性特征投影: 标量 U → D 维
        # 对应设计文档 §1.2.3: X_U = U·W_u + b_u
        # W_u ∈ R^{1×D}, b_u ∈ R^D
        self.uniqueness_proj = nn.Linear(1, d_model)

        # 门控网络: [X_base; X_U] → g_base
        # 对应设计文档 §1.2.4: g_base = σ(W_1[X_base; X_U] + b_1)
        # W_1 ∈ R^{2D×D}
        self.gate_net = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        # 门控偏置初始化为正值，使初始时 g_base ≈ 0.7
        # 这样预训练的碱基嵌入权重可以较好地迁移
        nn.init.xavier_uniform_(self.gate_net[0].weight)
        nn.init.constant_(self.gate_net[0].bias, 0.5)
        nn.init.xavier_uniform_(self.uniqueness_proj.weight)
        nn.init.zeros_(self.uniqueness_proj.bias)

    def forward(self, bases_21: torch.Tensor,
                uniqueness_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            bases_21: [B, L, 21] 碱基编码
            uniqueness_scores: [B, L] 唯一性评分, 0~1 (None 表示不使用评分，充 0 进行零数据库推理)

        Returns:
            X_final: [B, L, D] 门控融合后的嵌入
        """
        if uniqueness_scores is None:
            B, L = bases_21.shape[:2]
            uniqueness_scores = torch.zeros((B, L), device=bases_21.device, dtype=torch.float32)

        # Step 1: 碱基嵌入
        X_base = self.base_embedding(bases_21)  # [B, L, D]

        # Step 2: 唯一性特征投影
        # U: [B, L] → [B, L, 1] → Linear → [B, L, D]
        X_U = self.uniqueness_proj(uniqueness_scores.unsqueeze(-1))  # [B, L, D]

        # Step 3: 门控计算
        # [X_base; X_U]: [B, L, 2D]
        concat = torch.cat([X_base, X_U], dim=-1)
        g_base = self.gate_net(concat)  # [B, L, D], 值域 [0, 1]

        # Step 4: 门控融合
        # X_final = g_base ⊙ X_base + (1 - g_base) ⊙ X_U
        X_final = g_base * X_base + (1.0 - g_base) * X_U  # [B, L, D]

        return X_final


class SequencePositionEncoding(nn.Module):
    """
    序列级位置编码 (可学习)

    为每个 token 在序列中的绝对位置添加位置信息。
    这与 BaseEmbedding 中的 kmer_pos_embed (k-mer 内部位置) 不同，
    这是 token 在整个序列中的位置。
    """

    def __init__(self, d_model: int = 768, max_seq_len: int = 4076):
        super().__init__()
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] 嵌入向量

        Returns:
            x + position_encoding: [B, L, D]
        """
        B, L, D = x.shape
        positions = torch.arange(L, device=x.device)  # [L]
        pos_enc = self.pos_embed(positions)  # [L, D]
        return x + pos_enc
