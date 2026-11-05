#!/usr/bin/env python3
"""
生成细菌数据集的进化向量查找表 (bacteria_fna_phylo_embeddings.pt)

对应专利文档步骤一: 获取 GTDB 系统发育树进化坐标向量 P_vec ∈ R^{B×L×D_p}

策略: 按需精确提取 (Targeted Extraction)
  - 只针对 bacteria.genomic.fna 中实际存在的 Accession 生成向量
  - 输出格式: {accession_id: Tensor([d_phylo])}

输入文件:
  1. FNA 文件:      bacteria.genomic.fna (提取 NCBI Accession ID 与物种描述)
  2. 元数据文件:    bac120_metadata.tsv (建立 Accession/物种名/属名 → GTDB 节点的完整映射)
  3. 树文件:        bac120.tree (Newick 格式, 用于训练庞加莱双曲节点嵌入)

为什么需要物种名与属名桥梁映射:
  1. FNA 里的 Accession 是序列/Contig 级别 (如 NZ_BEWA01000001.1),
     而 GTDB 树里的节点是基因组 Assembly 级别 (如 GB_GCA_001955685.1).
  2. 序列 Accession 无法通过简单的字符串前缀替换转换为 Assembly ID.
  3. 通过 FASTA Header 中的物种名 (如 Streptomyces rishiriensis) 对齐
     GTDB metadata 中的 gtdb_taxonomy (s__... / g__...), 可以实现高精度的真实 GTDB 覆盖率.
  4. 遇到未完全收录亚种时，通过属名 (g__) 自动退步匹配同属代表菌，最大程度保留系统发育演化拓扑.

推荐运行方式:
  python scripts/generate_phylo_embeddings.py \
      --fna_path /path/to/bacteria.genomic.fna \
      --tree_path /path/to/bac120.tree \
      --metadata_path /path/to/bac120_metadata.tsv \
      --output_path ./bacteria_fna_phylo_embeddings.pt
"""

import os
import sys
import re
import argparse
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# 文本清洗与物种/属名提取工具
# ─────────────────────────────────────────────────────────────────────────────

def clean_name(name: str) -> str:
    """清理分类单元名称：去除方括号、引号及多余空白，转为小写。"""
    if not name:
        return ""
    name = re.sub(r'[\[\]\'"]', '', name).strip()
    return name.lower()


def extract_species_and_genus(desc: str) -> Tuple[str, str]:
    """
    从 FASTA Header 的 description 字段提取物种名 (Species) 与属名 (Genus)

    示例:
      'Streptomyces rishiriensis strain NBRC 13407...' -> ('Streptomyces rishiriensis', 'Streptomyces')
      'Escherichia coli str. K-12 substr. MG1655, complete genome' -> ('Escherichia coli', 'Escherichia')
      'Candidatus Pelagibacter ubique HTCC1062' -> ('Candidatus Pelagibacter ubique', 'Candidatus Pelagibacter')
      '[Clostridium] difficile 630' -> ('Clostridium difficile', 'Clostridium')

    Returns:
        (species_name, genus_name) 均为小写清洗后的字符串，若提取失败则为空字符串
    """
    if not desc:
        return "", ""

    # 移除常见截断模式：strain, str., subsp., isolate, 逗号等
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

    # 处理 Candidatus 暂定种
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


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: 解析 FASTA Header, 提取 Accession 及对应分类名
# ─────────────────────────────────────────────────────────────────────────────

def parse_fasta_headers(fna_path: str) -> Dict[str, Dict[str, str]]:
    """
    解析 FASTA 文件头部, 提取 Accession ID、物种名和属名

    Returns:
        Dict[accession_id -> {"species": str, "genus": str, "raw_desc": str}]
    """
    print(f"[1/4] 解析 FASTA 序列 Header: {fna_path}")
    accession_info: Dict[str, Dict[str, str]] = {}

    if not os.path.exists(fna_path):
        print(f"[Warning] FNA 文件不存在: {fna_path}，将以空字典继续（仅输出 UNKNOWN 兜底）")
        return accession_info

    with open(fna_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line.startswith('>'):
                continue
            header = line[1:].strip()
            parts = header.split(maxsplit=1)
            accession = parts[0]
            desc = parts[1] if len(parts) > 1 else ""

            sp_name, genus_name = extract_species_and_genus(desc)
            accession_info[accession] = {
                "species": sp_name,
                "genus": genus_name,
                "raw_desc": desc
            }

    print(f"    成功提取 {len(accession_info):,} 条 Accession ID。")
    return accession_info


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: 获取 / 训练 GTDB 庞加莱双曲嵌入 (Poincaré Embeddings)
# ─────────────────────────────────────────────────────────────────────────────

def build_deterministic_hash_vec(name: str, dim: int) -> np.ndarray:
    """
    基于名称的确定性哈希嵌入 (兜底算法)
    同一名称永远输出相同向量，向量位于庞加莱圆盘内 (模长 ≈ 0.85 < 1)
    """
    seed = int.from_bytes(
        name.encode('utf-8')[:4], byteorder='little', signed=False
    ) % (2 ** 31 - 1)
    sub_rng = np.random.RandomState(seed)
    raw_v = sub_rng.randn(dim).astype(np.float32)
    vec = raw_v / (np.linalg.norm(raw_v) + 1e-8) * 0.85
    return vec


def load_poincare_embeddings_from_tree(
    tree_path: str,
    dim: int = 64,
    epochs: int = 20
) -> Dict[str, np.ndarray]:
    """
    从 GTDB Newick 树文件训练庞加莱双曲节点嵌入

    核心修复:
      1. Newick 树中内部节点大多 clade.name 为 None 或数字支持率 (如 '1.0')。
         为每个内部节点动态分配唯一 ID (如 '_int_0')，避免树关系断裂及支持率数字合并冲突。
      2. 提取整棵树的所有父子边，保留完整全量拓扑。
    """
    print(f"[2/4] 从 GTDB bac120.tree 训练庞加莱双曲节点嵌入: {tree_path}")
    node_vectors: Dict[str, np.ndarray] = {}

    try:
        from gensim.models.poincare import PoincareModel
        from Bio import Phylo

        if not os.path.exists(tree_path):
            print(f"    [Warning] 树文件不存在: {tree_path}，跳过庞加莱训练。")
            return node_vectors

        print(f"    读取 Newick 树拓扑结构...")
        tree = Phylo.read(tree_path, "newick")

        # 1. 为内部节点赋予唯一编号，叶节点保留原始名称
        internal_id = 0
        terminal_count = 0
        for clade in tree.find_clades():
            if not clade.is_terminal():
                clade.name = f"_int_node_{internal_id}"
                internal_id += 1
            else:
                terminal_count += 1
                if clade.name:
                    clade.name = clade.name.strip("'\" \t\r\n")

        print(f"    树结构解析完毕: 叶节点 {terminal_count:,} 个, 内部节点 {internal_id:,} 个。")

        # 2. 提取全量父子分支关系
        relations: List[Tuple[str, str]] = []
        for clade in tree.find_clades():
            for child in clade.clades:
                if clade.name and child.name:
                    relations.append((clade.name, child.name))

        if len(relations) == 0:
            print("    [Warning] 未能提取有效分支关系，跳过庞加莱训练。")
            return node_vectors

        print(f"    提取到 {len(relations):,} 条进化分支关系，开始训练庞加莱双曲嵌入 (epochs={epochs})...")
        model = PoincareModel(relations, size=dim, burn_in=2)
        model.train(epochs=epochs, print_every=max(1, epochs // 5))

        for key in model.kv.key_to_index:
            # 内部节点无需暴露至外层查找表，仅需保存叶节点或所有节点
            node_vectors[key] = model.kv[key].astype(np.float32)

        print(f"    庞加莱嵌入训练完成，共 {len(node_vectors):,} 个节点向量。")

    except ImportError as e:
        print(f"    [Notice] 未检测到必要依赖 ({e})，将对所有 Accession 使用确定性哈希嵌入。")

    return node_vectors


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: 解析 bac120_metadata.tsv 建立多级映射桥梁
# ─────────────────────────────────────────────────────────────────────────────

def load_metadata_mapping(
    metadata_path: str,
    valid_tree_nodes: Optional[Set[str]] = None
) -> Dict[str, Dict[str, str]]:
    """
    解析 bac120_metadata.tsv，构建多级映射字典:
      - assembly:  Assembly ID / Accession (如 GCA_001955685.1, GCF_...) -> GTDB 节点名
      - species:   物种名 (如 streptomyces rishiriensis) -> GTDB 代表节点名
      - genus:     属名 (如 streptomyces) -> GTDB 代表节点名
      - organism:  NCBI 生物名 -> GTDB 节点名

    优先绑定存在于 valid_tree_nodes 中且为 GTDB 代表基因组 (gtdb_representative=t) 的节点。
    """
    print(f"[2.5/4] 解析 GTDB 元数据文件，建立 Accession/物种名/属名 多级映射: {metadata_path}")
    mapping: Dict[str, Dict[str, str]] = {
        "assembly": {},
        "species": {},
        "genus": {},
        "organism": {}
    }

    if not os.path.exists(metadata_path):
        print(f"    [Warning] 元数据文件不存在: {metadata_path}，跳过精准映射。")
        return mapping

    try:
        with open(metadata_path, 'r', encoding='utf-8', errors='ignore') as f:
            header_line = f.readline().strip()
            header = header_line.split('\t')

            def get_col_idx(col_name: str) -> int:
                return header.index(col_name) if col_name in header else -1

            idx_gtdb     = get_col_idx('accession')
            idx_ncbi_acc = get_col_idx('ncbi_genbank_assembly_accession')
            idx_tax      = get_col_idx('gtdb_taxonomy')
            idx_org      = get_col_idx('ncbi_organism_name')
            idx_rep      = get_col_idx('gtdb_representative')
            idx_rep_gen  = get_col_idx('gtdb_genome_representative')

            if idx_gtdb == -1:
                print(f"    [Warning] 未在元数据中找到 'accession' 列，跳过精准映射。")
                return mapping

            for line in f:
                parts = line.strip().split('\t')
                if len(parts) <= idx_gtdb:
                    continue

                gtdb_node = parts[idx_gtdb]   # e.g. "GB_GCA_001955685.1"
                rep_node  = parts[idx_rep_gen] if (idx_rep_gen != -1 and len(parts) > idx_rep_gen) else gtdb_node
                is_rep    = (parts[idx_rep].strip().lower() in ('t', 'true')) if (idx_rep != -1 and len(parts) > idx_rep) else False

                # 优先选择存在于有效树节点中的 ID
                target_node = gtdb_node
                if valid_tree_nodes:
                    if target_node not in valid_tree_nodes and rep_node in valid_tree_nodes:
                        target_node = rep_node

                is_in_tree = (valid_tree_nodes is None) or (target_node in valid_tree_nodes)

                # 1. Assembly Accession 映射
                ncbi_acc = parts[idx_ncbi_acc] if (idx_ncbi_acc != -1 and len(parts) > idx_ncbi_acc) else ""
                for acc_str in [gtdb_node, target_node, ncbi_acc]:
                    if acc_str and acc_str not in ('none', 'N/A'):
                        clean_acc = acc_str.replace('RS_', '').replace('GB_', '')
                        mapping["assembly"][acc_str] = target_node
                        mapping["assembly"][clean_acc] = target_node
                        mapping["assembly"][clean_acc.split('.')[0]] = target_node

                # 2. GTDB 分类学字段提取 (s__ 物种名, g__ 属名)
                if idx_tax != -1 and len(parts) > idx_tax:
                    tax_str = parts[idx_tax]
                    m_sp = re.search(r's__([^;]+)', tax_str)
                    m_gn = re.search(r'g__([^;]+)', tax_str)

                    if m_sp:
                        sp_clean = clean_name(m_sp.group(1))
                        # 优先绑定在树中且为代表基因组的节点
                        if sp_clean:
                            curr_sp = mapping["species"].get(sp_clean)
                            if curr_sp is None or (is_in_tree and is_rep):
                                mapping["species"][sp_clean] = target_node

                    if m_gn:
                        gn_clean = clean_name(m_gn.group(1))
                        if gn_clean:
                            curr_gn = mapping["genus"].get(gn_clean)
                            if curr_gn is None or (is_in_tree and is_rep):
                                mapping["genus"][gn_clean] = target_node

                # 3. NCBI Organism Name 映射
                if idx_org != -1 and len(parts) > idx_org:
                    org_name = parts[idx_org]
                    if org_name and org_name not in ('none', 'N/A'):
                        org_sp, org_gn = extract_species_and_genus(org_name)
                        if org_sp:
                            curr_osp = mapping["species"].get(org_sp)
                            if curr_osp is None or (is_in_tree and is_rep):
                                mapping["species"][org_sp] = target_node
                        if org_gn:
                            curr_ogn = mapping["genus"].get(org_gn)
                            if curr_ogn is None or (is_in_tree and is_rep):
                                mapping["genus"][org_gn] = target_node

        print(f"    成功建立多级映射条目:")
        print(f"      - Assembly 键数:  {len(mapping['assembly']):,}")
        print(f"      - 物种 (Species): {len(mapping['species']):,}")
        print(f"      - 属 (Genus):     {len(mapping['genus']):,}")
    except Exception as e:
        print(f"    [Error] 解析元数据文件失败: {e}，跳过精准映射。")

    return mapping


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: 映射 Accession 到进化向量查找表
# ─────────────────────────────────────────────────────────────────────────────

def build_embeddings_dict(
    accession_info: Dict[str, Dict[str, str]],
    gtdb_node_vectors: Dict[str, np.ndarray],
    d_phylo: int = 64,
    metadata_mapping: Optional[Dict[str, Dict[str, str]]] = None
) -> Dict[str, torch.Tensor]:
    """
    构建最终的 {Accession -> Tensor[d_phylo]} 字典

    匹配优先级:
      1. Assembly / Accession 精准命中
      2. 物种名 (Species Name) 精准命中 GTDB 基因组
      3. 属名 (Genus Name) 退步匹配同属代表基因组 (保留宏观系统发育拓扑)
      4. 直接名称命中
      5. 确定性哈希向量兜底
    """
    print(f"[3/4] 匹配 Accession → {d_phylo} 维进化向量...")
    phylo_dict: Dict[str, torch.Tensor] = {}

    hit_assembly = 0
    hit_species  = 0
    hit_genus    = 0
    hit_direct   = 0
    hit_hash     = 0

    asm_map = metadata_mapping.get("assembly", {}) if metadata_mapping else {}
    sp_map  = metadata_mapping.get("species", {}) if metadata_mapping else {}
    gn_map  = metadata_mapping.get("genus", {}) if metadata_mapping else {}

    # 向量张量缓存 (相同物种/节点共享同一 Tensor，极大提升构建速度并减小内存)
    node_tensor_cache: Dict[str, torch.Tensor] = {}

    def get_or_create_tensor(key_name: str, raw_array: Any) -> torch.Tensor:
        if key_name not in node_tensor_cache:
            node_tensor_cache[key_name] = torch.tensor(raw_array, dtype=torch.float32)
        return node_tensor_cache[key_name]

    for acc, info in tqdm(accession_info.items(), desc="Mapping Accessions", ncols=85):
        sp_name  = info["species"]
        gn_name  = info["genus"]
        matched_tensor = None

        # ── 优先级 1: Assembly / Accession 直接映射 ──
        for acc_candidate in [acc, acc.split('.')[0]]:
            gtdb_node = asm_map.get(acc_candidate)
            if gtdb_node and gtdb_node in gtdb_node_vectors:
                matched_tensor = get_or_create_tensor(gtdb_node, gtdb_node_vectors[gtdb_node])
                hit_assembly += 1
                break

        # ── 优先级 2: 物种名精准对齐 GTDB 基因组 ──
        if matched_tensor is None and sp_name:
            gtdb_node = sp_map.get(sp_name)
            if gtdb_node and gtdb_node in gtdb_node_vectors:
                matched_tensor = get_or_create_tensor(gtdb_node, gtdb_node_vectors[gtdb_node])
                hit_species += 1

        # ── 优先级 3: 属名退步匹配同属代表基因组 ──
        if matched_tensor is None and gn_name:
            gtdb_node = gn_map.get(gn_name)
            if gtdb_node and gtdb_node in gtdb_node_vectors:
                matched_tensor = get_or_create_tensor(gtdb_node, gtdb_node_vectors[gtdb_node])
                hit_genus += 1

        # ── 优先级 4: 原始节点名直接命中 ──
        if matched_tensor is None and acc in gtdb_node_vectors:
            matched_tensor = get_or_create_tensor(acc, gtdb_node_vectors[acc])
            hit_direct += 1

        # ── 优先级 5: 确定性哈希兜底 ──
        if matched_tensor is None:
            fallback_key = sp_name if sp_name else (gn_name if gn_name else acc)
            if fallback_key not in node_tensor_cache:
                hash_v = build_deterministic_hash_vec(fallback_key, d_phylo)
                node_tensor_cache[fallback_key] = torch.tensor(hash_v, dtype=torch.float32)
            matched_tensor = node_tensor_cache[fallback_key]
            hit_hash += 1

        phylo_dict[acc] = matched_tensor

    # 固定 UNKNOWN 兜底全 0 向量
    phylo_dict["UNKNOWN"] = torch.zeros(d_phylo, dtype=torch.float32)

    total = len(accession_info)
    if total > 0:
        real_hit = hit_assembly + hit_species + hit_genus + hit_direct
        print("\n" + "=" * 60)
        print("  进化向量匹配统计:")
        print(f"    1. Assembly 精准命中:    {hit_assembly:>9,} / {total:,} ({hit_assembly/total*100:6.2f}%)")
        print(f"    2. GTDB 物种名命中:      {hit_species:>9,} / {total:,} ({hit_species/total*100:6.2f}%)")
        print(f"    3. GTDB 属名退步命中:    {hit_genus:>9,} / {total:,} ({hit_genus/total*100:6.2f}%)")
        print(f"    4. 直接名称命中:         {hit_direct:>9,} / {total:,} ({hit_direct/total*100:6.2f}%)")
        print(f"    5. 确定性哈希兜底:       {hit_hash:>9,} / {total:,} ({hit_hash/total*100:6.2f}%)")
        print("    " + "-" * 56)
        print(f"    ★ 真实 GTDB 进化向量覆盖率: {real_hit:>9,} / {total:,} ({real_hit/total*100:6.2f}%)")
        print("=" * 60 + "\n")

    return phylo_dict


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: 验证与保存
# ─────────────────────────────────────────────────────────────────────────────

def validate_phylo_dict(phylo_dict: Dict[str, torch.Tensor], d_phylo: int) -> bool:
    """验证进化向量字典的基本格式正确性。"""
    print(f"[4/4] 验证进化向量字典格式...")
    errors = []

    if "UNKNOWN" not in phylo_dict:
        errors.append("缺少 'UNKNOWN' 兜底 Key")

    sample_keys = list(phylo_dict.keys())[:5]
    for key in sample_keys:
        v = phylo_dict[key]
        if not isinstance(v, torch.Tensor):
            errors.append(f"Key '{key}' 对应值不是 Tensor: {type(v)}")
        elif v.shape != (d_phylo,):
            errors.append(f"Key '{key}' 向量 Shape 不正确: {v.shape}，期望 ({d_phylo},)")
        elif v.dtype != torch.float32:
            errors.append(f"Key '{key}' 数据类型不是 float32: {v.dtype}")

    if errors:
        for e in errors:
            print(f"    [ERROR] {e}")
        return False

    print(f"    字典 Key 总数: {len(phylo_dict):,}")
    print(f"    向量维度: {d_phylo}")
    print(f"    示例 Keys (前 3 条):")
    for i, (k, v) in enumerate(list(phylo_dict.items())[:3]):
        norm = v.norm().item()
        print(f"      [{i}] {k}: norm={norm:.4f}, 前 5 值={v[:5].tolist()}")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="生成细菌数据集进化向量查找表 (bacteria_fna_phylo_embeddings.pt)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--fna_path", type=str,
        default="/home/user/zhangguoli/species_identification/utils/kmc_refseq/raw/bacteria.genomic.fna",
        help="细菌基因组 FASTA 文件路径"
    )
    parser.add_argument(
        "--tree_path", type=str, default="",
        help="GTDB bac120.tree 文件路径 (Newick 格式, 推荐提供)"
    )
    parser.add_argument(
        "--metadata_path", type=str, default="",
        help="GTDB bac120_metadata.tsv 文件路径 (推荐提供: 用于多级 Accession/物种名/属名 映射)"
    )
    parser.add_argument(
        "--output_path", type=str,
        default="./bacteria_fna_phylo_embeddings.pt",
        help="输出文件路径 (.pt)"
    )
    parser.add_argument(
        "--d_phylo", type=int, default=64,
        help="进化向量维度 D_p"
    )
    parser.add_argument(
        "--poincare_epochs", type=int, default=20,
        help="庞加莱双曲嵌入训练轮次 (仅在提供 tree_path 时有效)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  细菌数据集进化向量生成脚本 (多级映射增强版)")
    print(f"  FNA 文件:    {args.fna_path}")
    print(f"  树文件:      {args.tree_path or '(未提供, 使用哈希兜底)'}")
    print(f"  元数据文件:  {args.metadata_path or '(未提供, 跳过精准映射)'}")
    print(f"  输出路径:    {args.output_path}")
    print(f"  向量维度:    {args.d_phylo}")
    print("=" * 60)

    # Step 1: 解析 FASTA Header
    accession_info = parse_fasta_headers(args.fna_path)

    # Step 2: 获取/训练 GTDB 庞加莱节点嵌入
    tree_path = args.tree_path if args.tree_path else None
    gtdb_vectors: Dict[str, np.ndarray] = {}
    if tree_path:
        gtdb_vectors = load_poincare_embeddings_from_tree(
            tree_path=tree_path,
            dim=args.d_phylo,
            epochs=args.poincare_epochs
        )
    else:
        print("[2/4] 未提供树文件，全部使用确定性哈希嵌入。")

    # Step 2.5: 解析 bac120_metadata.tsv，建立多级映射
    metadata_mapping = {}
    if args.metadata_path:
        valid_tree_nodes = set(gtdb_vectors.keys()) if gtdb_vectors else None
        metadata_mapping = load_metadata_mapping(args.metadata_path, valid_tree_nodes=valid_tree_nodes)
    else:
        print("[2.5/4] 未提供元数据文件，跳过多级分类映射。")

    # Step 3: 构建 Accession → Tensor[d_phylo] 字典
    phylo_dict = build_embeddings_dict(
        accession_info=accession_info,
        gtdb_node_vectors=gtdb_vectors,
        d_phylo=args.d_phylo,
        metadata_mapping=metadata_mapping
    )

    # Step 4: 验证并保存
    ok = validate_phylo_dict(phylo_dict, d_phylo=args.d_phylo)
    if not ok:
        print("[ERROR] 验证失败，请检查以上错误信息。")
        sys.exit(1)

    out_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(out_dir, exist_ok=True)
    torch.save(phylo_dict, args.output_path)
    print(f"\n✓ 成功保存进化向量字典至: {args.output_path}")
    print(f"  字典 Key 总数: {len(phylo_dict):,}  (含 1 个 'UNKNOWN' 兜底)")
    print(f"  单个向量 Shape: ({args.d_phylo},) float32")
    print("\n使用方法:")
    print("  import torch")
    print(f"  phylo_db = torch.load('{args.output_path}')")
    print("  vec = phylo_db.get('NZ_BEWA01000001.1', phylo_db['UNKNOWN'])")
    print("  # vec: Tensor([64])")


if __name__ == "__main__":
    main()
