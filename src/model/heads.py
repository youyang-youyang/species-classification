"""
第四步：分类头与 MLM 头

对应设计文档 §4:
  层次化分类头: Z → Logit_S, Logit_T
  MLM 预训练头: H[mask_pos] → 碱基预测

层次化分类 (§4.1):
  Logit_S = Z·W_S + b_S   (物种 logits)
  Logit_T = Z·W_T + b_T   (毒株 logits, 预留)

层次化约束 (§4.1.4):
  使用 KmerFormer 验证过的软传递门控机制:
  每一级以上一级的概率分布作为条件信息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from collections import OrderedDict


# =========================================================================
#  MLM 预训练头 (第五步 §5.1 阶段一)
# =========================================================================

class MLMHead(nn.Module):
    """
    掩码语言模型头 (预训练阶段)

    对被 mask 的位置，预测原始 21-mer 中每个碱基的类别。

    Input:  H[mask_pos] [N_masked, D]
    Output: logits [N_masked, 21, 5]  (21个位置 × 5种碱基)
    """

    def __init__(self, d_model: int = 768, kmer_size: int = 21,
                 base_vocab_size: int = 5):
        super().__init__()
        self.kmer_size = kmer_size

        self.transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        # 预测 21 个位置的碱基: D → 21 × 5
        self.decoder = nn.Linear(d_model, kmer_size * base_vocab_size)

    def forward(self, hidden_states: torch.Tensor,
                mask_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, L, D] 编码器输出
            mask_positions: [B, L] bool, True 表示被 mask 的位置

        Returns:
            logits: [N_masked, 21, 5]  被 mask 位置的碱基预测
        """
        # 提取被 mask 位置的特征
        masked_h = hidden_states[mask_positions]  # [N_masked, D]

        # 变换 + 预测
        h = self.transform(masked_h)       # [N_masked, D]
        logits = self.decoder(h)           # [N_masked, 21*5]

        # 重塑: [N_masked, 21, 5]
        logits = logits.view(-1, self.kmer_size, 5)

        return logits


# =========================================================================
#  层次化软传递门控分类头 (第四步 §4.1)
# =========================================================================

class HierarchicalLevel(nn.Module):
    """
    单级层次化分类器 (软传递门控)

    参考 KmerFormer §6.7.2 验证过的设计:
      g = σ(Linear(prev_probs → D))              门控向量
      feat_gated = feat * g                       门控加权
      combined = cat[feat, feat_gated, prev_probs] 三路拼接
      logits = Linear(GELU(Linear(combined → D))) 分类

    门控机制的直觉:
    - 上级高置信度 → 门控强化对应特征维度
    - 上级不确定 → 门控减弱，让 feat 自主决策 (软传递)

    Input:
      feat: [B, D']       基因组指纹
      prev_probs: [B, C_prev]  上一级的概率分布
    Output:
      logits: [B, C_this]  本级的 logits
    """

    def __init__(self, d_fingerprint: int, prev_num_classes: int,
                 num_classes: int, dropout: float = 0.1):
        super().__init__()

        # 门控: 上级概率 → D 维门控向量
        self.gate = nn.Sequential(
            nn.Linear(prev_num_classes, d_fingerprint),
            nn.Sigmoid()
        )

        # 三路拼接后投影: [feat, feat_gated, prev_probs] → D → C
        combined_dim = d_fingerprint + d_fingerprint + prev_num_classes
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, d_fingerprint),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fingerprint, num_classes),
        )

    def forward(self, feat: torch.Tensor,
                prev_probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: [B, D'] 基因组指纹
            prev_probs: [B, C_prev] 上一级的 softmax 概率

        Returns:
            logits: [B, C_this] 本级的未归一化 logits
        """
        # 门控加权
        g = self.gate(prev_probs)              # [B, D']
        feat_gated = feat * g                  # [B, D']

        # 三路拼接 + 分类
        combined = torch.cat([feat, feat_gated, prev_probs], dim=-1)
        logits = self.classifier(combined)     # [B, C_this]

        return logits


class HierarchicalClassifier(nn.Module):
    """
    层次化分类头 - 设计文档 §4.1 完整实现

    支持七级分类 (Domain → Phylum → Class → Order → Family → Genus → Species)
    使用软传递门控机制保持层次一致性。

    对应设计文档公式:
      Logit_S = Z·W_S + b_S    (种级)
      Logit_T = Z·W_T + b_T    (株级, 预留)

    层级传递关系 (参考 KmerFormer §6.7.3):
      feat → domain_cls → domain_probs
                                 ↓
      feat → phylum_cls(feat, domain_probs) → phylum_probs
                                                    ↓
      feat → class_cls(feat, phylum_probs) → class_probs
                                                   ↓
      ...
      feat → species_cls(feat, genus_probs) → species_logits

    Input:
      Z: [B, D']       基因组指纹
    Output:
      logits: dict      {level_name: [B, C_level]} 各级 logits
    """

    def __init__(self, d_fingerprint: int,
                 num_classes_per_level: Dict[str, int],
                 dropout: float = 0.1):
        """
        Args:
            d_fingerprint: 基因组指纹维度 D'
            num_classes_per_level: OrderedDict, 按层级顺序
                如: {"domain": 5, "phylum": 100, ..., "species": 16000}
            dropout: Dropout 概率
        """
        super().__init__()

        self.levels = list(num_classes_per_level.keys())
        self.classifiers = nn.ModuleDict()

        prev_num_classes = 0
        for level, num_cls in num_classes_per_level.items():
            if prev_num_classes == 0:
                # 第一级 (Domain): 直接分类，无门控
                self.classifiers[level] = nn.Sequential(
                    nn.Linear(d_fingerprint, d_fingerprint // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_fingerprint // 2, num_cls),
                )
            else:
                # 后续级: 软传递门控
                self.classifiers[level] = HierarchicalLevel(
                    d_fingerprint=d_fingerprint,
                    prev_num_classes=prev_num_classes,
                    num_classes=num_cls,
                    dropout=dropout,
                )
            prev_num_classes = num_cls

    def forward(self, Z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            Z: [B, D'] 基因组指纹

        Returns:
            logits: dict {level_name: [B, C_level]}
        """
        logits = {}
        prev_probs = None

        for level in self.levels:
            if prev_probs is None:
                # 第一级: 直接分类
                logits[level] = self.classifiers[level](Z)
            else:
                # 后续级: 软传递门控
                logits[level] = self.classifiers[level](Z, prev_probs)

            # 计算本级概率用于下一级的门控输入
            prev_probs = F.softmax(logits[level], dim=-1).detach()
            # detach(): 防止梯度通过 softmax 反传到上级分类器
            # 各级分类器独立优化

        return logits
