"""
FNA 文件 O(1) 随机读取器

参考 KmerFormer 的 fna_reader.py 设计:
- 记录每条序列的字节偏移 fna_seq_start
- 利用定长行 (80bp/行) 直接计算目标位置的字节偏移
- 一次 seek + 一次 read, 无需遍历整个文件

FNA 文件格式:
  >NZ_BEWA01000001.1 Streptomyces rishiriensis strain NBRC 13407...
  GTGTCTCGTCGTCGAGATCGGGCCGGGCGAAGAACCGTGCGGCGGTGTCCGGGTGTTCCCGCTCAGTGAGAGCTGCGGCG
  ACAGCGCGCTGCTGCGTCAGAGACAGGGTGGGGGTTTCCGCTAGCGCGTCCCAGAAGGAGTGCGAGTGCATGAGGCCAGA
  ...
"""

import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class SequenceInfo:
    """单条序列的元数据"""
    accession: str              # 序列 accession (如 NZ_BEWA01000001.1)
    description: str            # 完整 header 行
    fna_file: str               # FNA 文件路径
    fna_seq_start: int          # 序列碱基在文件中的起始字节偏移
    line_len_bytes: int         # 每行字节数 (含换行符, 通常 81)
    line_len_bases: int         # 每行碱基数 (通常 80)
    num_bases: int              # 序列总碱基数
    is_variable_length: bool    # 是否为变长行 (最后一行通常短于 80)


class FNAReader:
    """
    FNA 文件 O(1) 随机读取器

    核心原理 (参考 KmerFormer fna_reader.py):
    给定 window_start (碱基下标), 计算文件字节偏移:
      first_line = window_start // line_len_bases
      col_offset = window_start % line_len_bases
      byte_offset = fna_seq_start + first_line × line_len_bytes + col_offset
    然后 seek → read → 去除换行符, 得到碱基字符串。
    """

    def __init__(self, fna_file: str):
        self.fna_file = fna_file
        self._fh = None

    def _ensure_open(self):
        """确保文件句柄打开 (惰性打开, 支持多进程)"""
        if self._fh is None or self._fh.closed:
            self._fh = open(self.fna_file, 'r')

    def read_bases(self, seq_info: SequenceInfo,
                   window_start: int, window_size: int) -> str:
        """
        O(1) 读取指定窗口的碱基序列

        Args:
            seq_info: 序列元数据
            window_start: 窗口起始位置 (碱基下标, 0-based)
            window_size: 窗口大小 (碱基数)

        Returns:
            sequence: 碱基字符串, 长度 = min(window_size, 剩余碱基数)
        """
        self._ensure_open()

        # 计算可读碱基数
        available = seq_info.num_bases - window_start
        read_len = min(window_size, available)
        if read_len <= 0:
            return ""

        line_len_bases = seq_info.line_len_bases
        line_len_bytes = seq_info.line_len_bytes

        # 计算文件字节偏移
        first_line = window_start // line_len_bases
        col_offset = window_start % line_len_bases
        byte_offset = seq_info.fna_seq_start + first_line * line_len_bytes + col_offset

        # 计算需要读取的字节数 (考虑换行符)
        # 从 col_offset 开始, 读 read_len 个碱基, 可能跨多行
        remaining_in_first_line = line_len_bases - col_offset
        if read_len <= remaining_in_first_line:
            # 不跨行
            bytes_to_read = read_len
        else:
            # 跨行: 计算总行数和换行符数
            after_first = read_len - remaining_in_first_line
            full_lines = after_first // line_len_bases
            last_part = after_first % line_len_bases
            # 总字节: 第一行剩余 + 换行 + full_lines × line_len_bytes + last_part
            bytes_to_read = remaining_in_first_line + 1  # +1 for \n
            bytes_to_read += full_lines * line_len_bytes
            if last_part > 0:
                bytes_to_read += last_part
            else:
                bytes_to_read -= 1  # 最后一行刚好用完, 不需要额外换行

        # 多读一些以确保足够 (换行符可能是 \r\n)
        bytes_to_read = min(bytes_to_read + 10, os.path.getsize(self.fna_file) - byte_offset)

        # Seek + Read
        self._fh.seek(byte_offset)
        raw = self._fh.read(bytes_to_read)

        # 去除换行符, 提取碱基
        bases = raw.replace('\n', '').replace('\r', '')[:read_len]

        return bases.upper()

    def close(self):
        if self._fh is not None and not self._fh.closed:
            self._fh.close()

    def __del__(self):
        self.close()
