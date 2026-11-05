# 961 物种续训：接收方按本页执行

## 1. 负责人另行交付的文件

| 源服务器绝对路径 | 接收方位置（相对仓库根目录） |
|---|---|
| `/home/user/zhangguoli/species_identification/index_bacteria_species961_train1000/`（完整目录，约47G） | `index_bacteria_species961_train1000/` |
| `/home/user/zhangguoli/species_identification/runs/bacteria_species961_train1000_seed42_v1_finetune/checkpoint_epoch003.pt`（约1.3G） | `handoff_961/resume.pt`（下载后改名） |

该checkpoint同时是截至该轮的最佳模型，已核对哈希，所以只需传一个`.pt`。
这是固定交接点：Epoch 3，120,128次更新，续训从`Epoch 4/9`开始。
`handoff_manifest.json`只对应这份数据和断点，换其他checkpoint需要重新核对清单。
无需其他100/32物种数据、全部历史checkpoint、预训练权重、原始taxdump、P/U评分或原始基因组全集。
配置已包含在GitHub，无需另传。

## 2. 拉取代码、安装环境

目标为Linux x86_64、单卡H100，NVIDIA驱动需支持CUDA 12.1。留足CPU内存以加载索引、模型和优化器，以及后续checkpoint所需磁盘空间。

```bash
git clone https://github.com/youyang-youyang/species-classification.git
cd species-classification
conda create -n species961 python=3.10.18 pip -y
conda activate species961
python -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-runtime.txt
python -m pip check
mkdir -p handoff_961
```

然后将第1节的目录和checkpoint放到指定位置。FASTA必须原样传输，不改变换行或内容。
这是与源服务器核心版本一致的精简环境。原`environment.yml`、`requirements-lock.txt`保留供核对，不必安装其中全部无关工具。
当前模型不依赖evo2、flash-attn、deepspeed。PyTorch安装命令依据[官方v2.1.0 CUDA 12.1说明](https://docs.pytorch.org/get-started/previous-versions/)。

## 3. 校验并自动适配路径

确认已有`handoff_961/resume.pt`，在仓库根目录执行：

```bash
cp handoff_manifest.json handoff_961/manifest.json
cp configs/finetune_species961_train1000_seed42.yaml handoff_961/source_config.yaml
cp handoff_961/resume.pt handoff_961/best_model.pt
python -u scripts/handoff_961.py verify
```

校验会逐文件检查SHA256，抽查train/val/test窗口，核对checkpoint，生成本机绝对路径配置`configs/finetune_species961_h100.yaml`，并将历史最佳模型放到`runs_h100/`。
扫描47G FASTA需要时间；看到`VERIFIED: resume at Epoch 4/9`才表示成功。
首次迁移使用新的输出目录，工具拒绝覆盖已存在的实验目录。

## 4. GPU检查与正式续训

选空闲GPU，以下使用物理0号卡：

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/handoff_961.py smoke
```

在独立临时目录恢复模型、优化器和调度器，运行8个microbatch（batch=4、累积=8）、一次更新、验证、保存和重载；这些测试参数会丢弃。
看到`GPU_SMOKE_PASSED`后执行：

```bash
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/resume_961_h100.sh > resume_h100.log 2>&1 < /dev/null &
tail -n 40 resume_h100.log
```

预期出现`Loaded checkpoint: ..., epoch=3`和`Epoch 4/9`。
保持总轮数10、batch=4、累积8和原学习率计划，单卡运行。只恢复到轮末，未保存的本轮进度不能恢复。
不要调用首次数据构建入口`run_species961_train1000.sh`续训。生成新的续训checkpoint后再次恢复，应选择新断点；启动脚本拒绝自动从旧交接点覆盖已有结果。

## 检查范围

`source_validation.json`记录了源服务器实际检查：完整模型严格加载、优化器恢复通过；train/val/test真实批次均为`[4,4076,21]`。
