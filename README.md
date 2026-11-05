# 961 物种训练交接

本仓库来自原服务器提交 `0df78f8`，提供训练代码、环境记录和迁移检查工具。
GitHub 不包含约 47G 的 FASTA、索引和模型。接收者必须另行取得 `species961_payload.tar`。
它包含完整 `index_bacteria_species961_train1000/`，及 `handoff_961/` 下的固定续训断点、该轮之前的最佳模型、原配置、SHA256清单和原服务器抽查窗口哈希。

## H100 接收方执行

Linux x86_64、NVIDIA 驱动支持 CUDA 12.1、单卡。准备足够 CPU 内存加载索引和约1.3G checkpoint。
源环境 `evo2_hf` 包含许多与本训练无关的工具，原始清单保留用于核对；建议为本项目单独安装精简环境。

```bash
conda create -n species961 python=3.10.18 pip -y
conda activate species961
python -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-runtime.txt
python -m pip check
```

安装 PyTorch 的命令来自 https://docs.pytorch.org/get-started/previous-versions/ 的 v2.1.0 CUDA 12.1 条目。
不需要安装 evo2、flash-attn 或 deepspeed：当前模型代码使用 PyTorch 实现。
这是根据源环境核心依赖整理的安装方案，并非已在对方 H100 上完成安装的声明。

在仓库根目录执行（替换压缩包路径；仅解压可信的本次交付包）：

```bash
tar -xf /收到文件的路径/species961_payload.tar
python -u scripts/handoff_961.py verify
CUDA_VISIBLE_DEVICES=0 python -u scripts/handoff_961.py smoke
```

`verify` 会逐文件检查哈希，抽查 train/val/test 窗口与原服务器读取一致，检查 checkpoint 结构，生成本机绝对路径配置，并将历史最佳模型放到新输出目录。
它扫描47G FASTA，会花一定时间；不要修改 FASTA 的换行、内容或标签库。
`smoke` 在独立临时目录中恢复优化器和调度器，执行8个microbatch（batch=4、累积=8），进行1次参数更新、验证、保存和重新加载，检查指标有限；测试产生的参数会丢弃，不用于正式续训。
看到 `VERIFIED` 和 `GPU_SMOKE_PASSED` 后才执行：

```bash
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/resume_961_h100.sh > resume_h100.log 2>&1 < /dev/null &
tail -n 40 resume_h100.log
```

保持10轮总预算、batch=4、累积=8和原学习率计划。`manifest.json` 记录实际起点。
若打包的是 Epoch 3，正式恢复应显示 Epoch 4/9；若打包的是 Epoch 4，则显示 Epoch 5/9。
原程序仅在轮末保存，无法恢复正在执行的一轮中途。旧checkpoint没有随机数状态，迁移后采样/dropout随机序列不逐步复现。
无需预训练权重、P/U评分或原始taxdump。不要用初次生成数据的 `run_species961_train1000.sh` 续训。
真正的验收还包括正式续训完成一轮并保存新 checkpoint；仅通过短检查不能保证长任务不会遇到资源或硬件问题。

## 源服务器打包

先将 `scripts/handoff_961.py` 同步到源服务器项目同名位置。在已激活的 `evo2_hf` 中：

```bash
python -u scripts/handoff_961.py package --output /新的交付目录
```

源代码/训练配置必须与 Git HEAD 一致。脚本会固定当前最新已保存的 epoch checkpoint，拒绝保存中发生变化的文件；不停止原训练。
输出 `PACKAGING_COMPLETE` 才说明成功。打包需要约50G额外空间；临时固定模型约2.6G，工具不自动删除。
Git代码/环境清单上传GitHub，payload通过可信文件传输或共享存储单独交付。
