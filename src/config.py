"""
全局配置 - 物种分类模型
严格对应设计文档中的预设全局参数：
- B: Batch Size
- L: Sequence Length (4096)
- D: Hidden Dimension (768)
- H: Number of Attention Heads (12)
- d_k: Head Dimension (D/H = 64)
- C_s, C_t: 物种和毒株的类别数量
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import yaml
import os
@dataclass
class ModelConfig:
    """模型架构配置 - 对应设计文档预设全局参数"""
    # === 嵌入层 ===
    base_vocab_size: int = 5            # A=0, T=1, C=2, G=3, N=4
    kmer_size: int = 21                 # 21-mer (与唯一性评分对齐)
    d_model: int = 768                  # D: Hidden Dimension
    max_seq_len: int = 4076            # L: 4096bp → 4096-21+1=4076 个 21-mer tokens
    # === CNN Stem (第二步：空洞卷积前端) ===
    cnn_num_layers: int = 3
    cnn_kernel_size: int = 3
    cnn_dilations: Tuple[int, ...] = (1, 2, 4)   # 膨胀率指数增加
    # === Sparse Transformer (第二步：稀疏注意力) ===
    n_layers: int = 12                  # 12层 Transformer
    n_heads: int = 12                   # H: 12 个注意力头
    d_ff: int = 3072                    # FFN 维度 (4 × D)
    dropout: float = 0.1
    attention_window: int = 256         # 滑动窗口大小
    # === 唯一性偏置 (第二步：Uniqueness Bias Term) ===
    beta_init: float = 0.5             # β: 偏置强度初始值
    w_b_init: float = 1.0              # W_b: 缩放系数初始值
    # === 多尺度池化 (第三步：MS-MaxPool) ===
    pool_kernels: Tuple[int, ...] = (3, 10, 50)    # 细/中/粗粒度
    pool_strides: Tuple[int, ...] = (1, 2, 10)
    gamma: float = 2.0                 # γ: 唯一性奖励系数
    d_fingerprint: int = 768           # D': 基因组指纹维度
    # === 分类头 (第四步：层次化对比分类) ===
    cls_dropout: float = 0.1
    cls_hidden_dim: int = 512          # 分类头中间层维度
    # === 子窗口分治 (Phase 1 临时方案) ===
    use_subwindow: bool = True          # 是否使用子窗口分治
    sub_window_size_bp: int = 1024      # 每个子窗口的碱基数
    n_sub_windows: int = 4              # 4个子窗口 × 1024bp = 4096bp
    @property
    def d_k(self) -> int:
        """d_k = D/H，每个注意力头的维度"""
        return self.d_model // self.n_heads
    @property
    def sub_window_kmer_len(self) -> int:
        """每个子窗口的 21-mer token 数"""
        return self.sub_window_size_bp - self.kmer_size + 1  # 1024-21+1=1004
@dataclass
class DataConfig:
    """数据配置 - 复用 KmerFormer 基础设施的路径"""
    # === 数据路径 ===
    fna_file: str = "/home/user/zhangguoli/species_identification/utils/kmc_refseq/raw/bacteria.genomic.fna"
    scores_file: str = "/home/user/zhangguoli/species_identification/utils/kmc_refseq/scores/bacteria_scores.txt"
    taxdump_dir: str = "/home/user/zhangguoli/species_identification/utils/taxonomy_data/new_taxdump"
    phylo_file: str = "/home/user/zhangguoli/species_identification/utils/gtdb/bacteria_fna_phylo_embeddings.pt"
    # === 索引/缓存路径 ===
    index_dir: str = "./index_bacteria"
    scores_bin_file: str = ""  # 预转换的二进制唯一性评分文件 (可选)
    dna_only: bool = False  # 无 P/U 的纯 DNA 数据，禁止读取评分或使用引导输入
    # === 窗口参数 ===
    window_size_bp: int = 4096          # 主窗口大小
    window_stride_bp: int = 2048        # 步长 (50% overlap)
    # === DataLoader ===
    num_workers: int = 4
    pin_memory: bool = True
    # === 分类层级 ===
    taxonomy_levels: Tuple[str, ...] = (
        "domain", "phylum", "class", "order", "family", "genus", "species"
    )
@dataclass
class TrainingConfig:
    """训练配置"""
    # === 预训练 (第五步 5.1: 阶段一) ===
    pretrain_epochs: int = 5
    pretrain_lr: float = 5e-4
    pretrain_weight_decay: float = 0.01
    mlm_mask_ratio: float = 0.15        # 15% 掩码
    # === 微调 (第五步 5.1: 阶段二) ===
    finetune_epochs: int = 50
    finetune_lr: float = 2e-5
    finetune_weight_decay: float = 0.01
    guidance_dropout: float = 0.0  # 同时关闭 P/U 的训练 batch 概率；微调验证始终仅 DNA
    disable_phylo: bool = False  # 始终关闭 P；此时 guidance_dropout 只控制 U
    # === 通用训练参数 ===
    batch_size_per_gpu: int = 8
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 2000
    max_grad_norm: float = 1.0
    precision: str = "bf16"             # A100 原生支持 BF16
    scheduler: str = "cosine"
    # === 采样策略 ===
    species_balanced_fraction: float = 0.0  # 0 保持原采样；>0 单卡混合有放回采样
    track_species_exposure: bool = False  # 完成反向传播的物种窗口输入次数（含重复）
    steps_per_epoch: Optional[int] = 15000  # 每 epoch 采样步数
    val_steps: Optional[int] = 2000
    # === 损失权重 (第四步：总损失函数 L_total) ===
    # L_total = L_CE + λ₁·L_hier + λ₂·L_contrast + λ₃·L_focus
    lambda_hier: float = 0.1            # λ₁: 层次化约束权重
    classification_loss_only: bool = False  # 普通多层 CE，无正确性门控/对比/聚焦
    lambda_contrast: float = 0.5        # λ₂: 对比损失权重
    lambda_focus: float = 0.3           # λ₃: 聚焦损失权重
    delta_focus: float = 5.0            # δ: 聚焦系数
    tau_contrast: float = 0.07          # τ: 对比温度系数
    # === Checkpoint ===
    save_interval: int = 5              # 每 N epoch 保存
    early_stop_patience: int = 10
    early_stop_metric: str = "species"
    # === 分布式 ===
    gpu_ids: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
@dataclass
class Config:
    """总配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    # === 实验管理 ===
    experiment_name: str = "species_cls_v1"
    output_dir: str = "./runs"
    seed: int = 42
    mode: str = "pretrain"  # "pretrain" or "finetune"
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """从 YAML 文件加载配置"""
        with open(path, 'r') as f:
            d = yaml.safe_load(f)
        model_cfg = ModelConfig(**d.get('model', {}))
        data_cfg = DataConfig(**d.get('data', {}))
        train_cfg = TrainingConfig(**d.get('training', {}))
        return cls(
            model=model_cfg,
            data=data_cfg,
            training=train_cfg,
            experiment_name=d.get('experiment_name', 'species_cls_v1'),
            output_dir=d.get('output_dir', './runs'),
            seed=d.get('seed', 42),
            mode=d.get('mode', 'pretrain'),
        )
    def to_yaml(self, path: str):
        """保存配置到 YAML"""
        import dataclasses
        d = dataclasses.asdict(self)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(d, f, default_flow_style=False, allow_unicode=True)
