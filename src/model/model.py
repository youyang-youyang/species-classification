"""
完整模型组装 - 融合唯一性感知的混合尺度物种精细分类模型

对应设计文档完整架构:

  预训练模式 (第五步 §5.1 阶段一):
    BaseEmbedding → PosEncoding → CNN Stem → Transformer×12 → MLM Head

  微调模式 (第五步 §5.1 阶段二):
    TripleGatedEmbedding → PosEncoding → CNN Stem → Transformer(+B_U)×12
    → MS-MaxPool → HierarchicalClassifier

  推理模式 (第五步 §5.2):
    同微调模式, 但输出概率分布 + 置信度校准

数据流维度追踪表 (设计文档总结表):
  Input:  [B, L, 21]  bases_21 + [B, L] uniqueness
  Step 1: [B, L, D]   门控融合嵌入
  Step 2: [B, L, D]   CNN + Transformer 编码
  Step 3: [B, D']     多尺度池化 → 基因组指纹 Z
  Step 4: [B, C]      层次化分类 logits
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from collections import OrderedDict

from .embedding import BaseEmbedding, DualGatedEmbedding, TripleGatedEmbedding, SequencePositionEncoding
from .encoder import HybridEncoder
from .pooling import UniquenessMSMaxPool
from .heads import MLMHead, HierarchicalClassifier
from .losses import MLMLoss, SpeciesClassificationLoss


class SpeciesClassifier(nn.Module):
    """
    融合唯一性感知的混合尺度物种精细分类模型

    支持两种模式:
    1. pretrain: 纯序列 MLM，学习 DNA 通用表征
    2. finetune: 引入唯一性评分，层次化物种分类

    核心创新 (设计文档总结):
    - 输入层: 动态门控融合碱基与唯一性特征
    - 编码层: 唯一性偏置拉近高唯一性位点距离
    - 池化层: 唯一性加权奖励关键特征
    - 分类层: 聚焦损失对关键位点错误零容忍
    """

    def __init__(self, config, mode: str = 'pretrain',
                 num_classes_per_level: Optional[Dict[str, int]] = None):
        """
        Args:
            config: ModelConfig 实例
            mode: 'pretrain' 或 'finetune'
            num_classes_per_level: 各层级类别数, 微调时必须提供
                如: {"domain": 5, "phylum": 100, ..., "species": 16000}
        """
        super().__init__()
        self.mode = mode
        self.config = config

        # ============================================================
        # 第一步: 嵌入层
        # ============================================================
        if mode == 'pretrain':
            # 预训练: 纯碱基嵌入 (无唯一性信息)
            self.embedding = BaseEmbedding(
                d_model=config.d_model,
                kmer_size=config.kmer_size,
                base_vocab_size=config.base_vocab_size,
            )
        else:
            # 微调: 三重门控嵌入 (碱基 + 进化向量 + 唯一性)
            # 对应专利权利要求 3: 三重门控融合
            self.embedding = TripleGatedEmbedding(
                d_model=config.d_model,
                kmer_size=config.kmer_size,
                base_vocab_size=config.base_vocab_size,
                d_phylo=getattr(config, 'd_phylo', 64),
            )

        # 序列级位置编码
        self.pos_encoding = SequencePositionEncoding(
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
        )

        # 嵌入 Dropout
        self.embed_dropout = nn.Dropout(config.dropout)

        # ============================================================
        # 第二步: 混合尺度稀疏编码器
        # ============================================================
        self.encoder = HybridEncoder(
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            cnn_kernel_size=config.cnn_kernel_size,
            cnn_dilations=config.cnn_dilations,
            window_size=config.attention_window,
            dropout=config.dropout,
            beta_init=config.beta_init,
            w_b_init=config.w_b_init,
            use_subwindow=config.use_subwindow,
            n_sub_windows=config.n_sub_windows,
        )

        # ============================================================
        # 输出头 (根据模式不同)
        # ============================================================
        if mode == 'pretrain':
            # MLM 头
            self.mlm_head = MLMHead(
                d_model=config.d_model,
                kmer_size=config.kmer_size,
                base_vocab_size=config.base_vocab_size,
            )
            self.loss_fn = MLMLoss()
        else:
            assert num_classes_per_level is not None, \
                "num_classes_per_level must be provided for finetune mode"

            # 第三步: 多尺度池化
            self.pooling = UniquenessMSMaxPool(
                d_model=config.d_model,
                d_fingerprint=config.d_fingerprint,
                gamma=config.gamma,
                pool_kernels=config.pool_kernels,
                pool_strides=config.pool_strides,
            )

            # 第四步: 层次化分类头
            self.classifier = HierarchicalClassifier(
                d_fingerprint=config.d_fingerprint,
                num_classes_per_level=num_classes_per_level,
                dropout=config.cls_dropout,
            )

        # 打印参数量
        self._print_params()

    def _print_params(self):
        """打印模型参数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] Mode: {self.mode}")
        print(f"[Model] Total parameters: {total:,}")
        print(f"[Model] Trainable parameters: {trainable:,}")

    def forward_pretrain(self, bases_21: torch.Tensor,
                         mask_positions: torch.Tensor,
                         mlm_targets: torch.Tensor
                         ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        预训练前向传播 (MLM)

        Args:
            bases_21: [B, L, 21] 被 mask 后的碱基编码
            mask_positions: [B, L] bool, True = 被 mask 的位置
            mlm_targets: [B, L, 21] 原始碱基 (用于计算 loss)

        Returns:
            loss: scalar
            metrics: dict
        """
        # Step 1: 嵌入
        x = self.embedding(bases_21)           # [B, L, D]
        x = self.pos_encoding(x)               # [B, L, D]
        x = self.embed_dropout(x)

        # Step 2: 编码 (预训练无唯一性偏置)
        H = self.encoder(x, uniqueness_scores=None)  # [B, L, D]

        # MLM 预测
        mlm_logits = self.mlm_head(H, mask_positions)  # [N_masked, 21, 5]

        # 提取 mask 位置的目标碱基
        targets = mlm_targets[mask_positions]  # [N_masked, 21]

        # 计算损失
        loss, metrics = self.loss_fn(mlm_logits, targets)

        return loss, metrics

    def forward_finetune(self, bases_21: torch.Tensor,
                         uniqueness_scores: Optional[torch.Tensor] = None,
                         phylo_vecs: Optional[torch.Tensor] = None,
                         labels: Optional[Dict[str, torch.Tensor]] = None,
                         loss_fn: Optional[SpeciesClassificationLoss] = None
                         ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor,
                                    Optional[torch.Tensor], Optional[Dict[str, float]]]:
        """
        微调/推理前向传播 (分类)

        对应专利步骤一 → 四全流程:
          bases_21 + U + P_vec → TripleGatedEmbedding → Encoder(+B_U)
          → MS-MaxPool → Classifier

        Args:
            bases_21:          [B, L, 21]      碱基编码
            uniqueness_scores: [B, L]          唯一性评分 (None 表示零数据库极速推理)
            phylo_vecs:        [B, L, d_phylo]  GTDB 进化坐标向量 (None 时自动全 0)
            labels:            dict {level: [B]}, None = 推理模式
            loss_fn:           SpeciesClassificationLoss 实例, None = 推理模式

        Returns:
            logits:  dict {level: [B, C]}
            Z:       [B, D'] 基因组指纹
            loss:    scalar 或 None (推理时)
            metrics: dict 或 None (推理时)
        """
        # Step 1: 三重门控嵌入 (X_base + X_phylo + X_U)
        x = self.embedding(bases_21, uniqueness_scores, phylo_vecs)  # [B, L, D]
        x = self.pos_encoding(x)
        x = self.embed_dropout(x)

        # Step 2: 编码 (带唯一性偏置 B_U)
        H = self.encoder(x, uniqueness_scores)  # [B, L, D]

        # Step 3: 多尺度池化 → 基因组指纹
        Z = self.pooling(H, uniqueness_scores)  # [B, D']

        # Step 4: 层次化分类
        logits = self.classifier(Z)  # dict {level: [B, C]}

        # 计算损失 (训练时)
        loss = None
        metrics = None
        if labels is not None and loss_fn is not None:
            # 计算样本级唯一性评分 (序列平均, 用于 focused loss)
            sample_uniqueness = uniqueness_scores.mean(dim=-1) if uniqueness_scores is not None else None  # [B]
            loss, metrics = loss_fn(logits, labels, Z, sample_uniqueness)

        return logits, Z, loss, metrics

    def forward(self, **kwargs):
        """统一入口"""
        if self.mode == 'pretrain':
            return self.forward_pretrain(**kwargs)
        else:
            return self.forward_finetune(**kwargs)

    def load_pretrained(self, pretrain_ckpt_path: str):
        """
        加载预训练权重到微调模型 (第五步 §5.1 阶段二)

        权重映射:
        - BaseEmbedding → DualGatedEmbedding.base_embedding (碱基嵌入迁移)
        - pos_encoding → pos_encoding (位置编码完全对应)
        - encoder → encoder (CNN + Transformer 完全对应)
        - mlm_head 丢弃 (微调不需要)
        """
        assert self.mode == 'finetune', "Only finetune mode can load pretrained weights"

        ckpt = torch.load(pretrain_ckpt_path, map_location='cpu',
                          weights_only=False)
        if 'model_state' in ckpt:
            pretrain_state = ckpt['model_state']
        elif 'state_dict' in ckpt:
            pretrain_state = ckpt['state_dict']
        else:
            pretrain_state = ckpt

        # 处理 DDP 的 module. 前缀
        pretrain_state = {
            k.replace('module.', ''): v for k, v in pretrain_state.items()
        }

        # 构建映射: 预训练 → 微调
        mapped_state = {}
        for key, value in pretrain_state.items():
            if key.startswith('embedding.'):
                # BaseEmbedding → DualGatedEmbedding.base_embedding
                new_key = key.replace('embedding.', 'embedding.base_embedding.')
                mapped_state[new_key] = value
            elif key.startswith('mlm_head.'):
                # 丢弃 MLM 头
                continue
            elif key.startswith('loss_fn.'):
                # 丢弃预训练损失
                continue
            else:
                # pos_encoding, encoder 完全对应
                mapped_state[key] = value

        # 加载 (strict=False: 微调模型的新模块无预训练权重)
        missing, unexpected = self.load_state_dict(mapped_state, strict=False)

        print(f"[Pretrained] Loaded {len(mapped_state)} parameters")
        print(f"[Pretrained] Missing keys (new modules): {len(missing)}")
        if missing:
            # 只打印新增模块的前几个 key
            new_modules = set(k.split('.')[0] for k in missing)
            print(f"[Pretrained] New modules: {new_modules}")
        if unexpected:
            print(f"[Pretrained] Unexpected keys: {unexpected}")

        return missing, unexpected


def build_model(config, mode: str = 'pretrain',
                num_classes_per_level: Optional[Dict[str, int]] = None,
                pretrain_ckpt: Optional[str] = None) -> SpeciesClassifier:
    """
    构建模型的工厂函数

    Args:
        config: ModelConfig
        mode: 'pretrain' 或 'finetune'
        num_classes_per_level: 微调时各级类别数
        pretrain_ckpt: 预训练权重路径 (仅微调时使用)

    Returns:
        model: SpeciesClassifier
    """
    model = SpeciesClassifier(
        config=config,
        mode=mode,
        num_classes_per_level=num_classes_per_level,
    )

    if pretrain_ckpt and mode == 'finetune':
        model.load_pretrained(pretrain_ckpt)

    return model