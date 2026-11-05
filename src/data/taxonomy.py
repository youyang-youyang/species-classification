"""
NCBI 分类数据库解析 (参考 KmerFormer taxonomy.py)

解析 rankedlineage.dmp, 建立 TaxID → 七级分类路径映射。

核心数据结构:
  _lineage:   Dict[taxid, {domain, phylum, class, order, family, genus, species}]
  _merged:    Dict[old_taxid, new_taxid]
  label2idx:  Dict[level, {name: idx}]
  idx2label:  Dict[level, [name]]
"""

import os
import pickle
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict


# 病毒域的 superkingdom 映射 (ICTV 2020 后)
SUPERKINGDOM_TO_DOMAIN = {
    'Bacteria': 'Bacteria',
    'Archaea': 'Archaea',
    'Eukaryota': 'Eukaryota',
    'Viruses': 'Viral',
    # ICTV 2020 新增 Realms
    'Riboviria': 'Viral',
    'Duplodnaviria': 'Viral',
    'Monodnaviria': 'Viral',
    'Varidnaviria': 'Viral',
    'Adnaviria': 'Viral',
    'Ribozyviria': 'Viral',
    'Anelloviridae': 'Viral',
}

TAXONOMY_LEVELS = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']
RANKED_LINEAGE_LEVELS = ['species', 'genus', 'family', 'order', 'class', 'phylum',
                         'kingdom', 'superkingdom']


class TaxonomyDB:
    """
    NCBI 分类数据库

    参考 KmerFormer taxonomy.py 的设计:
    1. 解析 rankedlineage.dmp → TaxID → 七级路径
    2. 处理 merged.dmp (旧 TaxID 合并)
    3. 构建 label2idx / idx2label 编码器
    4. num_classes 各层级类别数
    """

    def __init__(self):
        self._lineage: Dict[int, Dict[str, str]] = {}
        self._merged: Dict[int, int] = {}
        self.label2idx: Dict[str, Dict[str, int]] = {}
        self.idx2label: Dict[str, List[str]] = {}
        self.num_classes: Dict[str, int] = {}

    def load_taxdump(self, taxdump_dir: str, target_domain: str = 'Bacteria'):
        """
        加载 NCBI new_taxdump 数据

        Args:
            taxdump_dir: 包含 rankedlineage.dmp 和 merged.dmp 的目录
            target_domain: 目标域 (过滤)
        """
        # 1. 加载 merged.dmp (旧 TaxID → 新 TaxID)
        merged_path = os.path.join(taxdump_dir, 'merged.dmp')
        if os.path.exists(merged_path):
            self._load_merged(merged_path)

        # 2. 加载 rankedlineage.dmp
        lineage_path = os.path.join(taxdump_dir, 'rankedlineage.dmp')
        self._load_lineage(lineage_path, target_domain)

        print(f"[TaxonomyDB] Loaded {len(self._lineage)} taxa for domain '{target_domain}'")
        print(f"[TaxonomyDB] Merged TaxIDs: {len(self._merged)}")

    def _load_merged(self, path: str):
        """解析 merged.dmp"""
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 2:
                    old_id = int(parts[0].strip())
                    new_id = int(parts[1].strip())
                    self._merged[old_id] = new_id

    def _load_lineage(self, path: str, target_domain: str):
        """
        解析 rankedlineage.dmp

        格式: tax_id | tax_name | species | genus | family | order | class | phylum | kingdom | superkingdom
        """
        with open(path, 'r') as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split('|')]
                if len(parts) < 10:
                    continue

                taxid = int(parts[0])
                tax_name = parts[1]
                species = parts[2] or tax_name  # 有些行 species 为空, 用 tax_name 代替
                genus = parts[3]
                family = parts[4]
                order = parts[5]
                cls = parts[6]      # 'class' 是 Python 关键字
                phylum = parts[7]
                kingdom = parts[8]
                superkingdom = parts[9]

                # 确定 domain
                domain = SUPERKINGDOM_TO_DOMAIN.get(superkingdom, '')

                # 真菌特殊处理
                if superkingdom == 'Eukaryota' and kingdom.lower() == 'fungi':
                    domain = 'Fungi'

                # 过滤目标域
                if domain != target_domain:
                    continue

                # 跳过不完整的分类路径
                if not species:
                    continue

                self._lineage[taxid] = {
                    'domain': domain,
                    'phylum': phylum or 'Unknown_phylum',
                    'class': cls or 'Unknown_class',
                    'order': order or 'Unknown_order',
                    'family': family or 'Unknown_family',
                    'genus': genus or 'Unknown_genus',
                    'species': species,
                }

    def resolve_taxid(self, taxid: int) -> int:
        """解析可能被合并的 TaxID"""
        return self._merged.get(taxid, taxid)

    def get_lineage(self, taxid: int) -> Optional[Dict[str, str]]:
        """获取 TaxID 的分类路径"""
        taxid = self.resolve_taxid(taxid)
        return self._lineage.get(taxid)

    def build_label_encoders(self, used_taxids: Optional[Set[int]] = None):
        """
        构建标签编码器

        重要: 必须在 build_index() 之前调用!
        (参考 KmerFormer §5.3 的关键顺序约束)

        Args:
            used_taxids: 实际用到的 TaxID 集合。None 则使用全部
        """
        # 收集各层级出现的标签
        level_labels: Dict[str, Set[str]] = {level: set() for level in TAXONOMY_LEVELS}

        for taxid, lineage in self._lineage.items():
            if used_taxids is not None and taxid not in used_taxids:
                continue
            for level in TAXONOMY_LEVELS:
                val = lineage.get(level, '')
                if val:
                    level_labels[level].add(val)

        # 构建编码器 (排序以保证确定性)
        for level in TAXONOMY_LEVELS:
            sorted_labels = sorted(level_labels[level])
            self.label2idx[level] = {name: idx for idx, name in enumerate(sorted_labels)}
            self.idx2label[level] = sorted_labels
            self.num_classes[level] = len(sorted_labels)

        print("[TaxonomyDB] Label encoders built:")
        for level in TAXONOMY_LEVELS:
            print(f"  {level}: {self.num_classes[level]} classes")

    def encode_lineage(self, taxid: int) -> Dict[str, int]:
        """
        将 TaxID 编码为各层级的整数标签

        Returns:
            labels: {level: int_label}, 未找到时为 -1
        """
        lineage = self.get_lineage(taxid)
        if lineage is None:
            return {level: -1 for level in TAXONOMY_LEVELS}

        labels = {}
        for level in TAXONOMY_LEVELS:
            val = lineage.get(level, '')
            if level in self.label2idx and val in self.label2idx[level]:
                labels[level] = self.label2idx[level][val]
            else:
                labels[level] = -1

        return labels

    def save(self, path: str):
        """保存到 pickle"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"[TaxonomyDB] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'TaxonomyDB':
        """从 pickle 加载"""
        with open(path, 'rb') as f:
            db = pickle.load(f)
        print(f"[TaxonomyDB] Loaded from {path}")
        print(f"  Total taxa: {len(db._lineage)}")
        if db.num_classes:
            for level in TAXONOMY_LEVELS:
                print(f"  {level}: {db.num_classes.get(level, 0)} classes")
        return db
