#!/usr/bin/env python3
"""
物种分类体系构建与 Accession 对齐脚本 (独立预处理模块)

功能:
  1. 解析 NCBI rankedlineage.dmp 与 merged.dmp 构建细菌层级树。
  2. 扫描 bacteria.genomic.fna 序列 Header，鲁棒提取物种名并反查映射到真实 TaxID。
  3. 过滤出实际参与训练的数据集的 used_taxids 集合。
  4. 仅为真实存在的物种构建 Label Encoders (避免全量 54.9 万类别导致的显存爆炸)。
  5. 输出:
     - taxonomy_db.pkl   (精简高效的分类编码器，供模型分类头使用)
     - acc_to_taxid.pkl  (Accession -> TaxID 字典，供后续切窗口使用)

用法示例:
  python scripts/build_taxonomy.py --config configs/finetune_bacteria.yaml
  
  或者直接指定路径:
  python scripts/build_taxonomy.py \
      --taxdump-dir /home/user/xiezheng/hierarchical_model/taxonomy_data/new_taxdump \
      --fna-file /home/user/xiezheng/refseq/ori/bacteria.genomic.fna \
      --output-dir ./index_bacteria
"""

import os
import sys
import re
import pickle
import argparse
from typing import Tuple, Dict, Set
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data.taxonomy import TaxonomyDB


def clean_name(name: str) -> str:
    """清理分类单元名称：去除方括号、引号及多余空白，转为小写"""
    if not name:
        return ""
    return re.sub(r'[\[\]\'"]', '', name).strip().lower()


def extract_species_and_genus(desc: str) -> Tuple[str, str]:
    """
    从 FASTA Header 的 description 字段鲁棒提取物种名 (Species) 与属名 (Genus)
    
    示例:
      'Streptomyces rishiriensis strain NBRC 13407, whole genome shotgun sequence'
      -> ('streptomyces rishiriensis', 'streptomyces')
    """
    if not desc:
        return "", ""

    # 常见截断模式：strain, str., subsp., isolate, clone, 逗号, 括号
    cutoff_patterns = [
        r'\bstrain\b', r'\bstr\.\b', r'\bsubsp\.\b', r'\bsubstr\.\b',
        r'\bisolate\b', r'\bclone\b', r'\btype strain\b',
        r',.*', r'\(.*?\)'
    ]
    name_part = desc.strip()
    for pat in cutoff_patterns:
        m = re.search(pat, name_part, flags=re.IGNORECASE)
        if m:
            name_part = name_part[:m.start()].strip()

    words = [clean_name(w) for w in name_part.split() if clean_name(w)]
    if not words:
        return "", ""

    # 处理 Candidatus 暂定种 (如 Candidatus Pelagibacter ubique)
    if words[0] == 'candidatus' and len(words) >= 3:
        genus = f"candidatus {words[1]}"
        species = f"candidatus {words[1]} {words[2]}"
        return species, genus
    elif len(words) >= 2:
        genus = words[0]
        species = f"{words[0]} {words[1]}"
        return species, genus
    elif len(words) == 1:
        return words[0], words[0]
    return "", ""


def parse_assembly_summary(summary_path: str) -> Dict[str, int]:
    """解析 NCBI assembly_summary.txt 获取 assembly_accession -> taxid 映射"""
    acc_to_taxid = {}
    if not os.path.exists(summary_path):
        print(f"[Warning] assembly_summary not found: {summary_path}")
        return acc_to_taxid

    print(f"[1/3] 解析 assembly_summary.txt: {summary_path}")
    with open(summary_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 7:
                continue
            assembly_acc = parts[0]
            taxid = int(parts[5]) if parts[5].isdigit() else int(parts[6])
            acc_to_taxid[assembly_acc] = taxid

    print(f"      成功加载 {len(acc_to_taxid):,} 条 assembly 映射")
    return acc_to_taxid


def build_acc_to_taxid_from_fna(fna_file: str, taxonomy_db: TaxonomyDB) -> Tuple[Dict[str, int], Set[int]]:
    """
    扫描 FNA 序列 Header，基于物种名反查 TaxID
    
    Returns:
        acc_to_taxid: {accession: taxid}
        used_taxids: set of actual taxids
    """
    print("[2/3] 构建物种名反查索引表 (NCBI species -> TaxID)...")
    name_to_taxid: Dict[str, int] = {}
    for taxid, lineage in taxonomy_db._lineage.items():
        sp = lineage.get('species', '')
        if sp:
            name_to_taxid[clean_name(sp)] = taxid

    print(f"      反查表建立完毕，涵盖 {len(name_to_taxid):,} 个已知物种名。")
    print(f"[2/3] 开始扫描 FNA Header 匹配物种: {fna_file}")

    acc_to_taxid: Dict[str, int] = {}
    used_taxids: Set[int] = set()
    total_seqs = 0
    matched_seqs = 0

    with open(fna_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, desc="  Scanning FNA Headers", unit=" lines"):
            if not line.startswith('>'):
                continue
            total_seqs += 1
            header = line[1:].strip()
            parts = header.split(maxsplit=1)
            acc = parts[0]
            desc = parts[1] if len(parts) > 1 else ""

            sp_name, _ = extract_species_and_genus(desc)
            if sp_name in name_to_taxid:
                tid = name_to_taxid[sp_name]
                acc_to_taxid[acc] = tid
                used_taxids.add(tid)
                matched_seqs += 1

    match_rate = matched_seqs / max(total_seqs, 1) * 100
    print(f"      扫描结果: 总序列数 {total_seqs:,} | 成功匹配 TaxID 数 {matched_seqs:,} ({match_rate:.2f}%)")
    print(f"      训练集实际涵盖的不同物种 (TaxID) 数量: {len(used_taxids):,} 个")

    return acc_to_taxid, used_taxids


def main():
    parser = argparse.ArgumentParser(description='Build compact taxonomy database and accession mapping')
    parser.add_argument('--config', type=str, default='',
                        help='Path to config yaml file (optional)')
    parser.add_argument('--taxdump-dir', type=str, default='',
                        help='Directory containing rankedlineage.dmp and merged.dmp')
    parser.add_argument('--fna-file', type=str, default='',
                        help='Path to bacteria.genomic.fna')
    parser.add_argument('--output-dir', type=str, default='',
                        help='Directory to save taxonomy_db.pkl and acc_to_taxid.pkl')
    parser.add_argument('--assembly-summary', type=str, default='',
                        help='Path to assembly_summary.txt (optional)')
    parser.add_argument('--target-domain', type=str, default='Bacteria',
                        help='Target domain filter (default: Bacteria)')
    args = parser.parse_args()

    # 参数解析优先级: 命令行参数 > 配置文件
    taxdump_dir = args.taxdump_dir
    fna_file = args.fna_file
    output_dir = args.output_dir

    if args.config:
        cfg = Config.from_yaml(args.config)
        if not taxdump_dir:
            taxdump_dir = cfg.data.taxdump_dir
        if not fna_file:
            fna_file = cfg.data.fna_file
        if not output_dir:
            output_dir = cfg.data.index_dir

    if not taxdump_dir or not fna_file:
        parser.error("Must provide either --config or both --taxdump-dir and --fna-file!")

    if not output_dir:
        output_dir = "./index_bacteria"

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print("  物种分类体系构建与真实类别数精简流程")
    print(f"  Taxdump 目录:  {taxdump_dir}")
    print(f"  FNA 文件:      {fna_file}")
    print(f"  输出目录:      {output_dir}")
    print(f"  目标生物域:    {args.target_domain}")
    print("=" * 65)

    # 1. 载入原始 DMP 结构
    print("[1/3] 从 NCBI taxdump 加载分类树...")
    taxonomy_db = TaxonomyDB()
    taxonomy_db.load_taxdump(taxdump_dir, target_domain=args.target_domain)

    # 2. 扫描 FNA 提取真实 accession -> taxid 映射及 used_taxids
    if args.assembly_summary:
        acc_to_taxid = parse_assembly_summary(args.assembly_summary)
        used_taxids = set(acc_to_taxid.values())
    else:
        acc_to_taxid, used_taxids = build_acc_to_taxid_from_fna(fna_file, taxonomy_db)

    # 保存 acc_to_taxid
    acc_path = os.path.join(output_dir, 'acc_to_taxid.pkl')
    with open(acc_path, 'wb') as f:
        pickle.dump(acc_to_taxid, f)
    print(f"      ✓ 成功保存 Accession 映射至: {acc_path}")

    # 3. 关键步骤：仅为 used_taxids 构建 Label Encoders
    print(f"[3/3] 基于训练集实际涵盖的 {len(used_taxids):,} 个有效物种构建标签编码器...")
    taxonomy_db.build_label_encoders(used_taxids=used_taxids)

    # 验证分类维度
    print("\n--- 最终模型分类头类别数统计 ---")
    for level, n_cls in taxonomy_db.num_classes.items():
        print(f"  {level:10s}: {n_cls:>6,} classes")
    print("--------------------------------")

    # 保存 taxonomy_db.pkl
    db_path = os.path.join(output_dir, 'taxonomy_db.pkl')
    taxonomy_db.save(db_path)
    print(f"✓ 成功保存分类库至: {db_path}\n")
    print("下一步提示:")
    print(f"  现在可以运行 preprocess.py 生成训练窗口索引，类别数已压缩为正常规模，不会发生 OOM！")


if __name__ == '__main__':
    main()
