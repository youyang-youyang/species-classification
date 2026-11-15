# 961 物种实验：文件交付与运行

更新：2026-11-15。接收方使用 **Docker**，从本页按顺序执行。代码和进化向量从 GitHub 获取；原数据、新得分、预训练权重和 Docker 镜像另行交付。

## 1. 当前可以运行什么

| 实验 | 状态 |
|---|---|
| 纯 DNA 基线 | 原配置保留 |
| DNA＋U＋P，U/P 各自独立 50% dropout | 已接入，按第 4 节检查后使用第 5 节 Docker 命令 |

本实验配置同时设置 `guidance_dropout=0.5`、`guidance_dropout_mode=independent`。每个训练 micro-batch 分别抽取 U/P 是否关闭，四种输入组合理论上各 25%，DNA 始终保留；验证和选最佳模型始终只输入 DNA。旧配置不设置 mode 时仍为联合丢弃。

## 2. 需要手动传输的文件与位置

右列路径均相对于接收方的 `species-classification/` 仓库根目录。目录应原样传输，不能重写 FASTA 换行或 pickle 内容。

| 源服务器完整路径 | 接收方放置位置 | 用途 |
|---|---|---|
| `/home/user/zhangguoli/species_identification/index_bacteria_species961_train1000/` | `index_bacteria_species961_train1000/` | 原始完整数据目录，约 47GiB；以前已完整传过则无需重传 |
| `/home/user/zhangguoli/species_identification/auxiliary_species961_train_counts_cu64_20261111/window_scores_v1/` | `auxiliary_species961_train_counts_cu64_20261111/window_scores_v1/` | 新得分目录，得分二进制约 15.44GB；本实验必需 |
| `/home/user/zhangguoli/species_identification/runs/bacteria_species_cls_v1_pretrain_v2/best_model.pt` | `runs/bacteria_species_cls_v1_pretrain_v2/best_model.pt` | 从相同预训练起点开新实验，必须传；不是 961 微调 epoch003 断点 |
| 已单独交付的 `evo2_docker_parts/`（准确源路径需由镜像提供者确认） | 任意目录，按第 3 节导入 | 包含 `evo2_hf_docker.tar.gz.part_*` 和 `md5.txt`；已有对应镜像则不必重传 |

**进化向量已在 GitHub 中**：`phylo_species961_train_20261113/`，拉取代码即可获得，不需要手动重复传。权重和映射文件已与生成报告中的 SHA256 核对一致。真实树向量覆盖 960/961 物种；`Candidatus Carsonella ruddii`（taxid 114186）的 9 条训练序列使用零向量，缺失信息见报告。

**得分可传压缩包代替整个目录**：源服务器 `/home/user/zhangguoli/species_identification/window_scores_v1_20261113.tar.gz`（约 857MiB）及同路径 `.sha256` 文件。解压后恢复上表中的目录，不需要同时传未压缩版本。

新得分目录中运行必需：`train_window_scores.f32`、`train_row_offsets.npy`、`scores_manifest.json`、`SCORES_COMPLETE`；仅做生成过程追溯时，另附其父目录的 `reference_report.json`、`train_manifest.tsv`、`kmc_summary.json`。

放好后的主要目录：

```text
species-classification/
├── configs/
├── scripts/
├── src/
├── tests/data/
├── index_bacteria_species961_train1000/
│   ├── bacteria_complete.genomic.fna
│   ├── window_index_finetune.pkl
│   ├── taxonomy_db.pkl
│   └── ...（其余原文件也保留）
├── auxiliary_species961_train_counts_cu64_20261111/
│   └── window_scores_v1/
│       ├── train_window_scores.f32
│       ├── train_row_offsets.npy
│       ├── scores_manifest.json
│       ├── SCORES_COMPLETE
│       └── ...
├── phylo_species961_train_20261113/
│   ├── phylo_species961_train.pt
│   ├── mapping.tsv
│   ├── report.json
│   ├── PHYLO_COMPLETE
│   └── ...
└── runs/bacteria_species_cls_v1_pretrain_v2/best_model.pt
```

## 3. 拉取代码与准备环境

首次获取：

```bash
git clone https://github.com/youyang-youyang/species-classification.git
cd species-classification
```

已有仓库先执行 `git pull --ff-only origin main`；若有本地冲突，保留修改后处理，不要强制覆盖。

以下 Docker 示例使用已有交付镜像 `evo2_hf:cuda12.4`，目标 Linux x86_64，主机需安装 Docker、NVIDIA 驱动及 NVIDIA Container Toolkit。镜像是否在目标机可用须通过第 4 节验证；本次未在接收方机器测试。

镜像尚未导入时，在镜像分片目录执行：

```bash
cat evo2_hf_docker.tar.gz.part_* > evo2_hf_docker.tar.gz
md5sum -c md5.txt
# 校验通过后导入
docker load -i evo2_hf_docker.tar.gz
```

如果收到的是旧数据分片，在其所在目录合并；将下方 `/path/to/species-classification` 替换为接收方仓库绝对路径：

```bash
cat index_bacteria_part_* > index_bacteria.tar.gz
tar -xzf index_bacteria.tar.gz -C /path/to/species-classification
```

直接收到完整数据目录时不执行合并。得分压缩包及 `.sha256` 放到仓库根目录后执行：

```bash
sha256sum -c window_scores_v1_20261113.tar.gz.sha256 && \
  tar -xzf window_scores_v1_20261113.tar.gz
```

后续命令均在仓库根目录执行。新独立 P/U 配置已采用相对路径，无需改成原服务器用户名路径。不要重新构建、重写索引或重新划分窗口。

接收方按本页使用 Docker，不需要安装 udocker，也不需要导入名为 evo2_test 的容器。udocker 只是源服务器无 Docker 时使用的运行方式。镜像内需具备原训练依赖；不需要 KMC、GTDB 原始树、gensim 或 BioPython，除非重新生成辅助数据。

## 4. 先验证环境与数据

检查 GPU（示例使用物理 0 号卡）：

```bash
docker run --gpus '"device=0"' --rm \
  -v "$PWD:/workspace" -w /workspace evo2_hf:cuda12.4 \
  python -c "import torch; print(torch.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

检查代码与新得分（不需要 GPU）：

```bash
docker run --rm -v "$PWD:/workspace" -w /workspace evo2_hf:cuda12.4 \
  python tests/data/test_compact_scores.py

docker run --rm -v "$PWD:/workspace" -w /workspace evo2_hf:cuda12.4 \
  python tests/data/test_independent_dropout.py

docker run --rm -v "$PWD:/workspace" -w /workspace evo2_hf:cuda12.4 \
  python -u tests/data/check_compact_scores.py \
  --config configs/finetune_species961_train1000_pu_independent50_seed42.yaml
```

应出现得分六项测试 `OK`、独立 dropout 四项测试 `OK` 和 `COMPACT_DATA_CHECK_PASSED`。该步骤会完整核验得分 SHA256，并抽查 train/val/test 的读取，需扫描约 15GB。

拉取仓库内的进化向量后再检查 P/U 联合读取：

```bash
docker run --rm -v "$PWD:/workspace" -w /workspace evo2_hf:cuda12.4 \
  python -u tests/data/check_species961_phylo.py \
  --index-dir index_bacteria_species961_train1000 \
  --phylo-dir phylo_species961_train_20261113 \
  --scores-dir auxiliary_species961_train_counts_cu64_20261111/window_scores_v1
```

应出现 `PHYLO_DATA_CHECK_PASSED`。源服务器已通过这些 CPU 检查：训练 P 为 `[4,64]`，U 为 `[4,4076]`；验证、测试无 P/U。它不是 GPU 正式训练成功的证明。

## 5. Docker 启动 U/P 独立各 50% dropout 实验

**本节训练 DNA＋U＋P，U/P 两路各自独立 50% 整路 dropout。** 每物种 1000 个训练窗口（包含原有重复采样），10 epochs、seed42、batch4、累积8，原损失权重；从固定预训练权重初始化，验证和选最佳模型均只使用 DNA。

```bash
if [ -e runs/bacteria_species961_train1000_pu_independent50_seed42_v1_finetune ]; then
  echo '实验目录已存在，未启动；请设置新的 experiment-name，避免覆盖'
elif [ ! -f runs/bacteria_species_cls_v1_pretrain_v2/best_model.pt ]; then
  echo '缺少原固定预训练权重，请先完成第2节交付'
else
  nohup docker run --gpus '"device=0"' --rm --shm-size=16g \
    -v "$PWD:/workspace" -w /workspace evo2_hf:cuda12.4 \
    python -u scripts/finetune.py \
      --config configs/finetune_species961_train1000_pu_independent50_seed42.yaml \
      --pretrained runs/bacteria_species_cls_v1_pretrain_v2/best_model.pt \
      --gpus 0 \
    > tests/data/finetune_species961_pu50.log 2>&1 < /dev/null &
fi
```

```bash
tail -f tests/data/finetune_species961_pu50.log
```

预计从 `Epoch 0/9` 开始。新模型保存在 `runs/bacteria_species961_train1000_pu_independent50_seed42_v1_finetune/`；最佳模型为其中的 `best_model.pt`。

启动日志应显示 `dropout_mode=independent; Independent P/U dropout=0.5`，并加载 961000 个训练窗口、24025 个验证窗口和 14841 个向量键（含 UNKNOWN）。初期 u_drop_rate/p_drop_rate 约 0.5、guidance_drop_rate 约 0.25，短期允许随机波动。

源服务器已通过四项 dropout 测试，且用户提供的 udocker 日志已运行到 Step 200，损失有限、两路丢弃比例正常。接收方 Docker/GPU 尚未实机验收，第 4 节检查以及首个训练日志仍必须在接收方完成。

## 6. 对训练好的 P/U 模型进行纯 DNA 验证

```bash
docker run --gpus '"device=0"' --rm --shm-size=16g \
  -v "$PWD:/workspace" -w /workspace evo2_hf:cuda12.4 \
  python -u scripts/evaluate.py \
    --config configs/finetune_species961_train1000_pu_independent50_seed42.yaml \
    --checkpoint runs/bacteria_species961_train1000_pu_independent50_seed42_v1_finetune/best_model.pt \
    --split val --input-mode sequence_only --batch-size 4 --gpus 0 \
    --output runs/bacteria_species961_train1000_pu_independent50_seed42_v1_finetune/val_dna_only.json
```

默认评估完整验证集（24025 窗口）。方案固定后再将 `--split val` 改为 `--split test`，输出文件改为 `test_dna_only.json`，进行最终测试；不要用测试集选择方案。

## 7. 详细说明

- [961得分交付与运行](tests/data/961得分交付与运行.md)：compact 偏移格式、具体代码清单、环境备注。
- [961进化向量交付](tests/data/961进化向量交付.md)：真实映射、代理/缺失处理与拓扑训练方法。
- 原纯 DNA 配置和旧断点续训脚本保留在仓库中；本页命令为新独立 P/U dropout 实验，不恢复旧 epoch003 断点。
