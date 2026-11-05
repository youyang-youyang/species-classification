"""
唯一性评分读取器

读取 KmerFormer 预计算的 bacteria_scores.txt 文件。

文件格式:
  >NZ_BEWA01000001.1 Streptomyces rishiriensis strain NBRC 13407...
  0.9805,1,0.9427,0.9614,...(80个值)
  0.8053,1,1,1,...(80个值)
  ...

每行 80 个逗号分隔的 float 值, 对应 FNA 文件每行 80 个碱基。
每个值是一个 21-mer 位置的唯一性评分 (0~1)。

注意: scores.txt 中的值是已经通过 step5_count_to_score.py 转换后的:
  w(c) = exp(-a × (c-1) / 254)
  count=1 → score=1.0 (最高唯一性)
  count=255 → score≈0.007 (最低唯一性)
  count=-1 → score=-1 (含 N, 无效)
  count=0 → score=0 (Novel)
"""

import os
import time
import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class ScoreIndex:
    """单条序列的评分索引"""
    accession: str
    file_offset: int      # 在 scores.txt 中评分数据开始的字节偏移
    num_scores: int        # 评分值总数 (= num_bases - 20)
    line_len_values: int   # 每行的值数 (通常 80)


class ScoreReader:
    """
    唯一性评分读取器

    支持两种模式:
    1. 文本模式: 直接读取 bacteria_scores.txt (首次使用)
    2. 二进制模式: 读取预转换的 .scores.bin (高性能训练)

    建议首次使用时运行 convert_to_binary() 将文本转为 memmap 二进制格式。
    """

    def __init__(self, scores_file: str, scores_bin_file: str = ""):
        self.scores_file = scores_file

        if scores_bin_file and os.path.exists(scores_bin_file):
            # 二进制模式: memmap 随机访问
            self._mode = 'binary'
            self._mmap = np.memmap(scores_bin_file, dtype=np.float32, mode='r')
        else:
            self._mode = 'text'
            self._mmap = None
            self._fh = None

    def read_scores_text(self, index: ScoreIndex,
                         start: int, length: int) -> np.ndarray:
        """
        从文本文件读取评分 (较慢, 用于小规模验证)

        Args:
            index: 序列的评分索引
            start: 起始位置 (值索引, 0-based)
            length: 读取的值数

        Returns:
            scores: [length] float32 数组
        """
        if self._fh is None or self._fh.closed:
            self._fh = open(self.scores_file, 'r')

        scores = []
        line_len = index.line_len_values  # 通常 80

        # 计算起始行和列
        start_line = start // line_len
        start_col = start % line_len

        # Seek 到正确行
        self._fh.seek(index.file_offset)
        # 跳过前面的行
        for _ in range(start_line):
            self._fh.readline()

        # 读取数据
        remaining = length
        col = start_col
        while remaining > 0:
            line = self._fh.readline().strip()
            if not line or line.startswith('>'):
                break
            values = line.rstrip(',').split(',')
            # 从 col 开始取
            to_take = min(remaining, len(values) - col)
            for v in values[col:col + to_take]:
                try:
                    scores.append(float(v))
                except ValueError:
                    scores.append(0.0)
            remaining -= to_take
            col = 0  # 后续行从头开始

        result = np.array(scores, dtype=np.float32)
        # 处理无效值: -1 (含N) 和 0 (Novel) → clip 到 [0, 1]
        result = np.clip(result, 0.0, 1.0)

        # 如果长度不足, padding 为 0
        if len(result) < length:
            padded = np.zeros(length, dtype=np.float32)
            padded[:len(result)] = result
            result = padded

        return result

    def read_scores_binary(self, offset: int, length: int) -> np.ndarray:
        """
        从二进制 memmap 读取评分 (高性能)

        Args:
            offset: 在 .scores.bin 中的起始索引
            length: 读取的值数

        Returns:
            scores: [length] float32 数组
        """
        scores = self._mmap[offset:offset + length].copy()
        scores = np.clip(scores, 0.0, 1.0)
        return scores

    def read_scores(self, index_or_offset, start_or_length: int,
                    length: int = 0) -> np.ndarray:
        """
        统一读取接口

        文本模式: read_scores(score_index, start, length)
        二进制模式: read_scores(offset, length, 0)
        """
        if self._mode == 'binary':
            return self.read_scores_binary(index_or_offset, start_or_length)
        else:
            return self.read_scores_text(index_or_offset, start_or_length, length)

    @staticmethod
    def convert_to_binary(scores_txt_path: str, output_bin_path: str, batch_lines: int = 10000):
        """
        流式低内存高性能转换: 将 scores.txt 转换为二进制 memmap 文件 (.scores.bin)

        优化特性:
        1. 流式写盘 (Zero RAM Leak): 解析每段数据直接写入磁盘，内存占用恒定 < 150MB，彻底杜绝 Linux OOM Killer。
        2. C 级别向量化解析 (np.fromstring): 相比原生 Python 字符串循环提速 20~50 倍。
        3. 实时刷新进度 (flush=True): 即使在 nohup / 管道重定向下也能实时看到进度条和处理速度。
        4. 原子化写入 (.tmp 临时文件): 转换完成前不覆盖最终文件，避免损坏。

        Args:
            scores_txt_path: 输入文本文件路径
            output_bin_path: 输出二进制文件路径
            batch_lines: 批处理解析行数 (默认 10,000 行)

        Returns:
            indices: list of (accession, bin_offset, num_scores)
        """
        print(f"[ScoreReader] 开始流式转换: {scores_txt_path} → {output_bin_path}", flush=True)

        if not os.path.exists(scores_txt_path):
            raise FileNotFoundError(f"Scores file not found: {scores_txt_path}")

        file_size_gb = os.path.getsize(scores_txt_path) / (1024 ** 3)
        print(f"[ScoreReader] 输入文本文件大小: {file_size_gb:.2f} GB", flush=True)

        temp_bin_path = output_bin_path + ".tmp"
        indices = []
        current_offset = 0
        total_scores = 0
        line_count = 0
        start_time = time.time()
        last_log_time = start_time

        with open(scores_txt_path, 'r', buffering=8 * 1024 * 1024) as fin, \
             open(temp_bin_path, 'wb', buffering=8 * 1024 * 1024) as fout:

            current_acc = None
            acc_scores_count = 0
            chunk_lines = []

            def flush_chunk():
                nonlocal current_offset, acc_scores_count, total_scores
                if not chunk_lines:
                    return

                # 清理并快速拼接多行
                clean_parts = [l.strip().rstrip(',') for l in chunk_lines if l.strip()]
                chunk_lines.clear()
                if not clean_parts:
                    return

                content = ",".join(clean_parts)
                try:
                    arr = np.fromstring(content, dtype=np.float32, sep=',')
                except Exception:
                    tokens = [t.strip() for t in content.split(',') if t.strip()]
                    floats = []
                    for t in tokens:
                        try:
                            floats.append(float(t))
                        except ValueError:
                            floats.append(0.0)
                    arr = np.array(floats, dtype=np.float32)

                # 数值裁剪到 [0.0, 1.0] (处理 -1 等无效值)
                np.clip(arr, 0.0, 1.0, out=arr)

                # 直接写入二进制文件，不滞留内存
                arr.tofile(fout)
                num = len(arr)
                acc_scores_count += num
                total_scores += num

            for line in fin:
                line_count += 1
                if line.startswith('>'):
                    # 刷新上一条序列剩余的评分
                    flush_chunk()

                    # 记录上一条序列索引
                    if current_acc is not None:
                        indices.append((current_acc, current_offset, acc_scores_count))
                        current_offset += acc_scores_count
                        acc_scores_count = 0

                    current_acc = line[1:].strip().split()[0]
                else:
                    chunk_lines.append(line)
                    if len(chunk_lines) >= batch_lines:
                        flush_chunk()

                # 定期打印进度 (每 50 万行或每 10 秒)
                now = time.time()
                if (line_count % 500000 == 0) or (now - last_log_time >= 10.0):
                    elapsed = now - start_time
                    speed_lines = line_count / max(elapsed, 1e-5)
                    speed_mb = (fout.tell() / (1024 ** 2)) / max(elapsed, 1e-5)
                    print(f"[{time.strftime('%H:%M:%S')}] 已处理 {line_count:,} 行 | "
                          f"已保存序列: {len(indices):,} 条 | "
                          f"总评分数: {total_scores:,} | "
                          f"速度: {speed_lines:,.0f} lines/s ({speed_mb:.1f} MB/s)", flush=True)
                    last_log_time = now

            # 处理文件末尾最后一条序列
            flush_chunk()
            if current_acc is not None:
                indices.append((current_acc, current_offset, acc_scores_count))
                current_offset += acc_scores_count

        # 原子性重命名
        if os.path.exists(output_bin_path):
            os.remove(output_bin_path)
        os.rename(temp_bin_path, output_bin_path)

        total_time = time.time() - start_time
        bin_size_gb = os.path.getsize(output_bin_path) / (1024 ** 3)
        print(f"\n[ScoreReader] ✓ 转换完成！耗时: {total_time/60:.1f} 分钟", flush=True)
        print(f"  总序列数: {len(indices):,}", flush=True)
        print(f"  总评分点数: {total_scores:,}", flush=True)
        print(f"  输出二进制文件: {output_bin_path} ({bin_size_gb:.2f} GB)\n", flush=True)

        return indices

    def close(self):
        if hasattr(self, '_fh') and self._fh is not None and not self._fh.closed:
            self._fh.close()
        if self._mmap is not None:
            del self._mmap
            self._mmap = None
