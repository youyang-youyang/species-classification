"""
第四步（续）：损失函数集合

对应设计文档 §4.1.2 - §4.1.5:

总损失函数:
  L_total = L_CE + λ₁·L_hier + λ₂·L_contrast + λ₃·L_focus

各项损失:
  1. L_CE:        标准交叉熵
  2. L_hier:      层次化约束 (种正确时才计算株损失)
  3. L_contrast:  ANI 引导的对比损失 (InfoNCE)
  4. L_focus:     唯一性区域聚焦损失

公式 (设计文档 §4.1):
  L_focus    = -1/L Σ (1 + δ·U_i) · CE(H_out_i, y_i)
  L_contrast = -Σ log[exp(Z_i·Z_j/τ) / Σ exp(Z_i·Z_a/τ)]
  L_hier     = CE(Logit_S, y_S) + α·CE(Logit_T, y_T)·I(y_S == ŷ_S)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class FocusedLoss(nn.Module):
    """
    唯一性区域聚焦损失 - 设计文档 §4.1.2

    公式:
      L_focus = -1/L Σ_{i=1}^{L} (1 + δ · U_i) · CE(H_out_i, y_i)

    含义: δ 是聚焦系数。U_i 越高，该位置的损失权重越大。
    唯一性高的位置对总损失的贡献是普通位置的 (1+δ) 倍。

    当 δ=5.0 时:
    - U=1 的位置: 权重 = 1+5=6  (唯一性区域)
    - U=0 的位置: 权重 = 1+0=1  (普通区域)
    → 唯一性高的位置对总损失的贡献是普通位置的 6 倍
    """

    def __init__(self, delta: float = 5.0):
        super().__init__()
        self.delta = delta

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                uniqueness_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            logits: [B, C] 分类 logits
            labels: [B] 真实标签
            uniqueness_scores: [B] 样本级唯一性评分 (序列平均)。None 则退化为标准 CE

        Returns:
            loss: scalar
        """
        # 标准交叉熵 (per-sample)
        ce_loss = F.cross_entropy(logits, labels, reduction='none')  # [B]

        if uniqueness_scores is not None:
            # 加权: (1 + δ · U_i) · CE_i
            weights = 1.0 + self.delta * uniqueness_scores  # [B]
            loss = (weights * ce_loss).mean()
        else:
            loss = ce_loss.mean()

        return loss


class ContrastiveLoss(nn.Module):
    """
    ANI 引导的对比损失 - 设计文档 §4.1.3

    公式:
      L_contrast = -Σ_{i∈I} log [ Σ_{j∈P(i)} exp(Z_i·Z_j/τ) /
                                   Σ_{a∈A(i)} exp(Z_i·Z_a/τ) ]

    含义:
    - P(i): 同毒株正样本集 (同物种)
    - A(i): 包含难负例的样本集 (ANI 95%-99% 的近缘物种)
    - τ: 温度系数

    实现: 使用 Supervised Contrastive Learning (SupCon)
    在 batch 内，同物种的样本为正对，不同物种为负对。
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, Z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z: [B, D'] 基因组指纹 (L2 归一化前)
            labels: [B] 物种标签

        Returns:
            loss: scalar
        """
        B = Z.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=Z.device)

        # L2 归一化
        Z_norm = F.normalize(Z, p=2, dim=-1)  # [B, D']

        # 相似度矩阵: [B, B]
        sim_matrix = torch.matmul(Z_norm, Z_norm.T) / self.temperature  # [B, B]

        # 构建正样本掩码: 同物种 = 正对
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
        # 排除自身
        mask_self = ~torch.eye(B, dtype=torch.bool, device=Z.device)
        positive_mask = labels_eq & mask_self  # [B, B]

        # 如果没有正对, 返回 0
        if positive_mask.sum() == 0:
            return torch.tensor(0.0, device=Z.device)

        # 数值稳定: 减去最大值
        sim_max, _ = sim_matrix.max(dim=-1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()

        # log-sum-exp (分母: 所有非自身样本)
        exp_sim = torch.exp(sim_matrix) * mask_self.float()
        log_sum_exp = torch.log(exp_sim.sum(dim=-1, keepdim=True) + 1e-8)  # [B, 1]

        # 正样本的 log 概率
        log_prob = sim_matrix - log_sum_exp  # [B, B]

        # 对每个 anchor, 取其正样本的平均 log 概率
        # 正样本数量: [B]
        n_pos = positive_mask.float().sum(dim=-1)  # [B]
        n_pos = torch.clamp(n_pos, min=1.0)  # 避免除零

        # 正样本 log 概率之和 / 正样本数
        loss = -(positive_mask.float() * log_prob).sum(dim=-1) / n_pos  # [B]
        loss = loss.mean()

        return loss


class HierarchicalConstraintLoss(nn.Module):
    """
    层次化约束损失 - 设计文档 §4.1.4

    公式:
      L_hier = CE(Logit_S, y_S) + α · CE(Logit_T, y_T) · I(y_S == ŷ_S)

    含义: I(·) 是指示函数，仅当种预测正确时才计算更细级别的损失。
    防止种分类错误时，株分类被错误的梯度误导。

    实现: 对每个层级计算 CE，按层级权重加权。
    下级损失仅在上级预测正确时生效。
    """

    def __init__(self, level_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        # 各层级权重 (参考 KmerFormer: species 最高)
        self.level_weights = level_weights or {
            'domain': 0.2, 'phylum': 0.6, 'class': 0.8,
            'order': 1.0, 'family': 1.2, 'genus': 1.5, 'species': 2.0,
        }

    def forward(self, logits: Dict[str, torch.Tensor],
                labels: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            logits: dict {level: [B, C_level]}
            labels: dict {level: [B]}

        Returns:
            loss: scalar
        """
        total_loss = torch.tensor(0.0, device=next(iter(logits.values())).device)
        prev_correct = None  # 上级是否预测正确的掩码

        for level in logits.keys():
            if level not in labels:
                continue

            level_logits = logits[level]
            level_labels = labels[level]

            # 跳过无效标签 (值为 -1)
            valid_mask = level_labels >= 0
            if valid_mask.sum() == 0:
                continue

            # 计算 CE loss (仅有效样本)
            ce_loss = F.cross_entropy(
                level_logits[valid_mask],
                level_labels[valid_mask],
                reduction='none'
            )  # [N_valid]

            # 层次化约束: 仅上级正确时计算本级损失
            if prev_correct is not None:
                constraint_mask = prev_correct[valid_mask]
                ce_loss = ce_loss * constraint_mask.float()

            # 加权
            weight = self.level_weights.get(level, 1.0)
            total_loss = total_loss + weight * ce_loss.mean()

            # 更新本级的预测正确掩码 (用于下一级)
            with torch.no_grad():
                preds = level_logits.argmax(dim=-1)
                prev_correct = (preds == level_labels)  # [B]

        return total_loss


class SpeciesClassificationLoss(nn.Module):
    """
    总损失函数 - 设计文档 §4.1.5

    L_total = L_CE + λ₁·L_hier + λ₂·L_contrast + λ₃·L_focus

    变量速查表 (§4.2):
      δ:     Scalar  聚焦系数
      τ:     Scalar  温度系数
      λ₁₂₃:  Scalar  各项损失权重
    """

    def __init__(self, lambda_hier: float = 0.1,
                 lambda_contrast: float = 0.5,
                 lambda_focus: float = 0.3,
                 delta_focus: float = 5.0,
                 tau_contrast: float = 0.07,
                 level_weights: Optional[Dict[str, float]] = None,
                 classification_only: bool = False):
        super().__init__()

        self.lambda_hier = lambda_hier
        self.classification_only = classification_only
        self.lambda_contrast = lambda_contrast
        self.lambda_focus = lambda_focus

        self.focused_loss = FocusedLoss(delta=delta_focus)
        self.contrastive_loss = ContrastiveLoss(temperature=tau_contrast)
        self.hierarchical_loss = HierarchicalConstraintLoss(level_weights)

    def forward(self, logits: Dict[str, torch.Tensor],
                labels: Dict[str, torch.Tensor],
                Z: torch.Tensor,
                uniqueness_scores: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            logits: dict {level: [B, C]} 各级分类 logits
            labels: dict {level: [B]} 各级真实标签
            Z: [B, D'] 基因组指纹 (用于对比损失)
            uniqueness_scores: [B] 样本级唯一性评分 (序列平均)

        Returns:
            L_total: scalar 总损失
            metrics: dict 各项损失的数值 (用于日志)
        """
        # 1. 标准交叉熵 (物种级, 最重要)
        species_logits = logits.get('species')
        species_labels = labels.get('species')
        if species_logits is not None and species_labels is not None:
            valid = species_labels >= 0
            if valid.sum() > 0:
                L_CE = F.cross_entropy(species_logits[valid], species_labels[valid])
            else:
                L_CE = torch.tensor(0.0, device=Z.device)
        else:
            L_CE = torch.tensor(0.0, device=Z.device)

        if self.classification_only:
            # Upper heads are detached in the classifier: retain direct supervision.
            # Species CE is counted exactly once; upper-level weights reuse config.
            L_upper = Z.sum() * 0.0
            for level, level_logits in logits.items():
                if level == 'species' or level not in labels:
                    continue
                valid = labels[level] >= 0
                if valid.sum() > 0:
                    L_upper = L_upper + self.hierarchical_loss.level_weights.get(level, 1.0) * F.cross_entropy(
                        level_logits[valid], labels[level][valid])
            L_total = L_CE + self.lambda_hier * L_upper
            return L_total, {'L_CE': L_CE.item(), 'L_upper_CE': L_upper.item(),
                             'L_hier': 0.0, 'L_contrast': 0.0, 'L_focus': 0.0,
                             'L_total': L_total.item()}

        # 2. 层次化约束损失
        L_hier = self.hierarchical_loss(logits, labels)

        # 3. 对比损失 (使用物种标签)
        if species_labels is not None:
            L_contrast = self.contrastive_loss(Z, species_labels)
        else:
            L_contrast = torch.tensor(0.0, device=Z.device)

        # 4. 唯一性聚焦损失
        if species_logits is not None and species_labels is not None:
            L_focus = self.focused_loss(species_logits, species_labels, uniqueness_scores)
        else:
            L_focus = torch.tensor(0.0, device=Z.device)

        # 总损失
        L_total = (L_CE
                   + self.lambda_hier * L_hier
                   + self.lambda_contrast * L_contrast
                   + self.lambda_focus * L_focus)

        metrics = {
            'L_CE': L_CE.item(),
            'L_hier': L_hier.item(),
            'L_contrast': L_contrast.item(),
            'L_focus': L_focus.item(),
            'L_total': L_total.item(),
        }

        return L_total, metrics


class MLMLoss(nn.Module):
    """
    MLM 预训练损失

    对被 mask 的位置，计算 21 个碱基位置的交叉熵。
    """

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            logits: [N_masked, 21, 5] 预测的碱基 logits
            targets: [N_masked, 21] 真实碱基

        Returns:
            loss: scalar
            metrics: dict
        """
        # 展平: [N_masked × 21, 5] vs [N_masked × 21]
        logits_flat = logits.reshape(-1, 5)
        targets_flat = targets.reshape(-1).long()

        # 忽略 N (编码=4) 的位置
        valid = targets_flat < 4
        if valid.sum() == 0:
            loss = torch.tensor(0.0, device=logits.device)
            acc = 0.0
        else:
            loss = F.cross_entropy(logits_flat[valid], targets_flat[valid])
            # 计算准确率
            preds = logits_flat[valid].argmax(dim=-1)
            acc = (preds == targets_flat[valid]).float().mean().item()

        return loss, {'mlm_loss': loss.item(), 'mlm_acc': acc}
