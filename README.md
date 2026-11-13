# 961 物种训练：接收方按本页执行（Docker 版）

> **961 物种辅助输入实验（2026-11-13 更新）**
>
> - 新得分 U 的交付、验证与独立实验配置：[961得分交付与运行](tests/data/961得分交付与运行.md)。
> - 训练用进化向量 P 的交付与覆盖率：[961进化向量交付](tests/data/961进化向量交付.md)。真实树向量覆盖 960/961 物种，缺失物种使用零向量。
> - 代码已支持 compact 训练得分；验证、测试保持纯 DNA。P/U **独立各 50% dropout 尚未实现**，旧 guidance_dropout 是联合丢弃。
> - 大型得分、数据和权重单独交付，不包含在本 Git 仓库。以下保留原 Docker 训练交付流程。


## 1. 需额外下载并准备的交付文件

目标机器为 **Linux x86_64 + GPU（NVIDIA 驱动支持 CUDA 12.1/12.4 及以上）**，需预先安装 Docker 与 `nvidia-container-toolkit`，并预留足够的磁盘空间（数据解压约 47G + 镜像解压与后续 checkpoint 空间）。

| 交付内容 | 说明 | 最终放置位置 |
|---|---|---|
| 数据分片文件 `index_bacteria_part_*` | `index_bacteria.tar.gz` 的分片文件（`part_aa` ~ `part_ad` 等，解压后约 47G；FASTA 保持原样不改换行） | 解压至仓库根目录 `species-classification/index_bacteria_species961_train1000/` |
| Docker 镜像分片 `evo2_docker_parts/` | 含 `evo2_hf_docker.tar.gz.part_*` 与 `md5.txt` | 任意目录（导入后可删除） |
| 预训练权重 `best_model.pt`（若配置需要） | 若本次不是从零初始化、需基于预训练底座微调，请一并下载 | 例如 `species-classification/runs_pretrained/best_model.pt` |

---

## 2. 拉取代码、合并数据与导入 Docker 镜像

镜像内已完整预装 Python 3.10.18、PyTorch 2.1.0+cu121 及全部运行依赖，**无需再手动创建 conda 环境或 pip 安装依赖**。

### 2.1 拉取代码仓库
```bash
git clone https://github.com/youyang-youyang/species-classification.git
cd species-classification
```

### 2.2 合并并解压 961 物种数据目录
进入网盘下载的**数据分片所在目录**：
```bash
cd /path/to/data_parts

# 1. 按顺序合并所有分片还原为 index_bacteria.tar.gz
cat index_bacteria_part_* > index_bacteria.tar.gz

# 2. 解压到代码仓库根目录下
tar -xzf index_bacteria.tar.gz -C /path/to/species-classification/

# 3. 确认解压后的数据目录已在仓库根目录下
ls -lh /path/to/species-classification/
```

### 2.3 合并并导入 Docker 镜像
进入网盘下载的 `evo2_docker_parts/` 目录：
```bash
cd /path/to/evo2_docker_parts

# 1. 合并分片还原完整镜像包
cat evo2_hf_docker.tar.gz.part_* > evo2_hf_docker.tar.gz

# 2. 校验文件完整性（输出 OK 表示完全无损）
md5sum -c md5.txt

# 3. 导入 Docker 镜像（导入后镜像名为 evo2_hf:cuda12.4）
docker load -i evo2_hf_docker.tar.gz
```

---

## 3. 检查路径配置

打开 `species-classification/configs/finetune_species961_train1000_seed42.yaml`，确认其中的数据与索引路径使用的是容器内路径（推荐统一为 `/workspace/index_bacteria_species961_train1000/...` 或相对路径），不要保留原服务器的绝对路径（如 `/home/user/zhangguoli/...`），否则容器内会报 `FileNotFoundError`。

> **提示**：已直接交付完整的 `index_bacteria_species961_train1000/`，**无需也不要调用**首次数据构建脚本 `run_species961_train1000.sh`，也不用执行任何断点续训（`handoff_961` / `resume.pt`）步骤。

---

## 4. GPU 环境自检与正式启动训练

以下默认使用物理 0 号 GPU（`device=0`），先进入代码仓库根目录：
```bash
cd /path/to/species-classification
```

### 4.1 快速验证容器内 GPU 是否正常
```bash
docker run --gpus '"device=0"' --rm \
    -v $(pwd):/workspace \
    -w /workspace \
    evo2_hf:cuda12.4 \
    python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA可用:', torch.cuda.is_available(), 'GPU:', torch.cuda.get_device_name(0))"
```
确认输出 `CUDA可用: True` 且能正常打印显卡型号后，再启动正式训练。

### 4.2 后台启动正式训练（从 Epoch 0 开始）
```bash
nohup docker run --gpus '"device=0"' --rm \
    --shm-size=16g \
    -v $(pwd):/workspace \
    -v $(pwd):/home/user/zhangguoli/species_identification \
    -w /workspace \
    evo2_hf:cuda12.4 \
    python -u scripts/finetune.py \
        --config configs/finetune_species961_train1000_seed42.yaml \
        --experiment-name bacteria_species961_train1000_h100 \
        --gpus 0 \
    > train_961_h100.log 2>&1 < /dev/null &
```
*(注：如果本次微调需要指定预训练底座权重，请在命令中加上 `--pretrained /workspace/你的权重路径/best_model.pt`)*

### 4.3 查看训练日志
```bash
tail -f train_961_h100.log
```
正常启动后将看到索引加载完成，并从 `Epoch 0/9` 开始训练（总轮数 10、batch=4、累积=8，单卡运行）。
