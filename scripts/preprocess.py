"""
数据预处理入口脚本

执行顺序 (参考 KmerFormer §11 关键约束):
  Step 1: 解析 NCBI rankedlineage.dmp → TaxonomyDB
  Step 2: build_label_encoders() ← 必须在 build_index 之前!
  Step 3: 扫描 FNA → 提取序列元信息
  Step 4: 扫描 scores.txt → 评分偏移
  Step 5: (可选) 转换 scores.txt → .scores.bin (memmap 二进制)
  Step 6: 构建窗口索引 → window_index_pretrain.pkl / window_index_finetune.pkl
  Step 7: Stratified 划分 (以 accession 为粒度)

用法:
  python scripts/preprocess.py --config configs/pretrain_bacteria.yaml --mode all
  python scripts/preprocess.py --config configs/pretrain_bacteria.yaml --mode scores_bin

仅重新划分已有索引 (不扫描原始数据、不覆盖旧索引):
  python scripts/preprocess.py --config configs/finetune_bacteria.yaml \
      --mode resplit --output-dir index_bacteria_stratified \
      --val-ratio 0.1 --test-ratio 0.1 --seed 42
  可选 --group-map sources.tsv，表头 accession<TAB>group_id；group_id 为经过核实的
  assembly/菌株来源组，必须覆盖微调和预训练索引中的所有 accession。
  不传映射时仅按 accession 隔离。默认双来源组物种按比例随机分到 train+val
  或 train+test；--two-group-policy val 可复现旧脚本的处理。
  完成标记是 split_report.json 中 status=complete。失败目录不能用于训练。
  报告 old_pretrain_group_exposure 中 train/val→test 表示新测试组曾参与旧预训练
  或模型选择；继续使用旧预训练权重时，不能宣称整个训练流程未接触该测试组。
  输出配置仅改变路径，尚未修复进化输入捷径。索引很大时需要足够主机内存和磁盘。
"""

import sys
import os
import argparse
import pickle
import csv
import json
import shutil
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data.taxonomy import TaxonomyDB
from src.data.index_builder import IndexBuilder
from src.data.score_reader import ScoreReader


def resplit_existing(config, args):
    """复用已有窗口元组；不扫描序列，不重建窗口，不覆盖旧目录。"""
    source = os.path.abspath(config.data.index_dir)
    if not args.output_dir:
        raise ValueError('--mode resplit requires --output-dir (a new directory)')
    destination = os.path.abspath(args.output_dir)
    if os.path.exists(destination):
        raise FileExistsError(f'Output directory must not exist: {destination}')
    taxonomy_path = os.path.join(source, 'taxonomy_db.pkl')
    taxonomy = TaxonomyDB.load(taxonomy_path)
    group_map = None
    if args.group_map:
        group_map = {}
        with open(args.group_map, encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            if not {'accession', 'group_id'}.issubset(reader.fieldnames or []):
                raise ValueError('Group TSV requires accession and group_id columns')
            for row in reader:
                acc, group = row['accession'].strip(), row['group_id'].strip()
                if not acc or not group or (acc in group_map and group_map[acc] != group):
                    raise ValueError(f'Empty/conflicting group mapping: {acc}')
                group_map[acc] = group

    names = ('train', 'val', 'test')
    def load_index(filename, width):
        with open(filename, 'rb') as f:
            old = pickle.load(f)
        if not isinstance(old, dict) or not all(s in old for s in names):
            raise ValueError(f'Missing split keys: {filename}')
        if any(not isinstance(old[s], list) for s in names):
            raise ValueError('Split values must be lists')
        for s in names:
            for w in old[s]:
                if len(w) != width:
                    raise ValueError(f'Expected {width}-field windows in {filename}')
        return old

    print('[Resplit] Loading finetune index; large indexes require substantial host RAM.', flush=True)
    old = load_index(os.path.join(source, 'window_index_finetune.pkl'), 4)
    species, metadata, label_cache = {}, {}, {}
    for split in names:
        for info, start, offset, taxid in old[split]:
            if taxid not in label_cache:
                label_cache[taxid] = taxonomy.encode_lineage(taxid).get('species', -1)
            label = label_cache[taxid]
            acc = info.accession
            signature = (info.fna_file, info.fna_seq_start, info.num_bases,
                         info.line_len_bytes, info.line_len_bases, offset, label)
            if acc in metadata and metadata[acc] != signature:
                raise ValueError(f'Conflicting sequence/score/label metadata: {acc}')
            species[acc], metadata[acc] = label, signature
    if not species:
        raise ValueError('Empty finetune index')
    del metadata, label_cache
    assignment = IndexBuilder.species_group_assignment(
        species, group_map, args.val_ratio, args.test_ratio, args.seed,
        args.two_group_policy)
    group_split = {}
    group_sizes = Counter()
    seen_groups = set()
    for acc, split in assignment.items():
        group = group_map[acc] if group_map is not None else acc
        group_split[group] = split
        if group not in seen_groups:
            group_sizes[species[acc]] += 1
            seen_groups.add(group)
    del seen_groups
    report = {
        'source_dir': source, 'seed': args.seed,
        'val_ratio': args.val_ratio, 'test_ratio': args.test_ratio,
        'two_group_policy': args.two_group_policy,
        'group_map': os.path.abspath(args.group_map) if args.group_map else None,
        'isolation': 'provided_group_id' if group_map is not None else 'accession_only',
        'sequence_similarity_checked': False,
        'species_group_count_histogram': dict(Counter(group_sizes.values())),
        'singleton_species_train_only': sum(n == 1 for n in group_sizes.values()),
        'two_group_species': sum(n == 2 for n in group_sizes.values()),
        'splits': {},
    }
    labels_by_split = {s: {species[a] for a in assignment if assignment[a] == s} for s in names}
    for s in names:
        if not labels_by_split[s].issubset(labels_by_split['train']):
            raise ValueError(f'Unseen species in {s}')
        report['splits'][s] = {
            'species': len(labels_by_split[s]),
            'accessions': sum(v == s for v in assignment.values()),
            'groups': sum(v == s for v in group_split.values()),
        }

    def redistribute(old_index, stage):
        result = {s: [] for s in names}
        transitions = {s: {t: 0 for t in names} for s in names}
        before = sum(len(old_index[s]) for s in names)
        for previous in names:
            for window in old_index[previous]:
                acc = window[0].accession
                if acc not in assignment:
                    if group_map is not None and acc not in group_map:
                        raise ValueError(f'Pretrain accession missing group mapping: {acc}')
                    group = group_map[acc] if group_map is not None else acc
                    # 预训练独有组放 train；共享组沿用微调分区，防止跨阶段泄漏。
                    assignment[acc] = group_split.setdefault(group, 'train')
                target = assignment[acc]
                result[target].append(window)
                transitions[previous][target] += 1
            old_index[previous].clear()
        if sum(len(result[s]) for s in names) != before:
            raise ValueError('Window count changed')
        report[stage] = {
            'total_windows_preserved': before,
            'windows': {s: len(result[s]) for s in names},
            'old_split_to_new_split_windows': transitions,
        }
        return result

    # 输出目录是全新目录；成功标记最后写入，失败时不能使用半成品。
    os.makedirs(destination)
    def save_index(data, filename):
        path = os.path.join(destination, filename)
        with open(path + '.tmp', 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(path + '.tmp', path)
    new = redistribute(old, 'finetune')
    save_index(new, 'window_index_finetune.pkl')
    del new, old
    pretrain_path = os.path.join(source, 'window_index_pretrain.pkl')
    if os.path.exists(pretrain_path):
        print('[Resplit] Aligning pretrain index to the same source-group split.', flush=True)
        old = load_index(pretrain_path, 2)
        # 来源组层面的历史接触：包括旧预训练 val 的模型选择接触。
        exposure = {s: Counter() for s in names}
        for previous in names:
            prior_groups = set()
            for window in old[previous]:
                acc = window[0].accession
                if group_map is not None and acc not in group_map:
                    raise ValueError(f'Pretrain accession missing group mapping: {acc}')
                prior_groups.add(group_map[acc] if group_map is not None else acc)
            for group in prior_groups:
                if group in group_split:
                    exposure[previous][group_split[group]] += 1
        report['old_pretrain_group_exposure'] = {s: dict(v) for s, v in exposure.items()}
        new = redistribute(old, 'pretrain')
        save_index(new, 'window_index_pretrain.pkl')
        del new, old
    else:
        report['pretrain'] = {'status': 'missing; prior exposure cannot be audited'}
    shutil.copy2(taxonomy_path, os.path.join(destination, 'taxonomy_db.pkl'))
    with open(os.path.join(destination, 'split_manifest.tsv'), 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['accession', 'group_id', 'species_label', 'split'])
        for acc in sorted(assignment):
            writer.writerow([acc, group_map[acc] if group_map is not None else acc,
                             species.get(acc, -1), assignment[acc]])
    import yaml
    with open(args.config, encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    config_dict.setdefault('data', {})['index_dir'] = destination
    # 仅移动索引；序列、评分和进化文件继续使用原路径。
    for field in ('fna_file', 'scores_file', 'scores_bin_file', 'phylo_file', 'taxdump_dir'):
        value = getattr(config.data, field, '')
        if value:
            config_dict['data'][field] = os.path.abspath(value)
    with open(os.path.join(destination, 'config_resplit.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump(config_dict, f, allow_unicode=True, sort_keys=False)
    report['status'] = 'complete'
    with open(os.path.join(destination, 'split_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print('Resplit complete. config_resplit.yaml changes paths only; it does NOT remove shortcut inputs.')


def parse_assembly_summary(summary_path: str) -> dict:
    """
    解析 assembly_summary.txt 获取 accession → taxid 映射

    NCBI assembly_summary.txt 格式:
    # assembly_accession	bioproject	...	species_taxid	...
    GCF_000005845.2	PRJNA224116	...	562	...

    也可以从 FNA header 中提取 (如果 header 含有 TaxID 信息)
    """
    acc_to_taxid = {}

    if not os.path.exists(summary_path):
        print(f"[Warning] assembly_summary not found: {summary_path}")
        return acc_to_taxid

    with open(summary_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 7:
                continue
            assembly_acc = parts[0]  # GCF_000005845.2
            taxid = int(parts[5]) if parts[5].isdigit() else int(parts[6])
            acc_to_taxid[assembly_acc] = taxid

    print(f"[Preprocess] Loaded {len(acc_to_taxid):,} accession→taxid mappings")
    return acc_to_taxid


def build_acc_to_taxid_from_headers(fna_file: str, taxonomy_db: TaxonomyDB) -> dict:
    """
    从 FNA header 中推断 accession → taxid 映射

    如果没有 assembly_summary.txt, 可以从 header 的物种名推断。
    这是一个备用方案, 准确性取决于 header 格式。

    FNA header 示例:
    >NZ_BEWA01000001.1 Streptomyces rishiriensis strain NBRC 13407, ...
    """
    acc_to_taxid = {}
    # 构建 species_name → taxid 反向索引
    name_to_taxid = {}
    for taxid, lineage in taxonomy_db._lineage.items():
        sp = lineage.get('species', '')
        if sp:
            name_to_taxid[sp.lower()] = taxid

    print(f"[Preprocess] Building accession→taxid from FNA headers...")
    with open(fna_file, 'r') as f:
        for line in f:
            if not line.startswith('>'):
                continue
            acc = line[1:].split()[0]
            # 尝试从 header 提取物种名
            desc = line[1:].strip()
            # 移除 accession
            desc_no_acc = ' '.join(desc.split()[1:])
            # 去除 ", whole genome shotgun sequence" 等后缀
            desc_clean = desc_no_acc.split(',')[0].split(' strain ')[0].strip()

            if desc_clean.lower() in name_to_taxid:
                acc_to_taxid[acc] = name_to_taxid[desc_clean.lower()]

    print(f"[Preprocess] Matched {len(acc_to_taxid):,} accessions from headers")
    return acc_to_taxid


def main():
    parser = argparse.ArgumentParser(description='Data Preprocessing (Window Index & Splitting)')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'taxonomy', 'scan', 'scores_bin', 'index', 'resplit'],
                        help='Preprocessing step to run')
    parser.add_argument('--assembly-summary', type=str, default='',
                        help='Path to assembly_summary.txt for accession→taxid mapping')
    parser.add_argument('--target-domain', type=str, default='Bacteria')
    parser.add_argument('--force', action='store_true', help='Force rebuild')
    parser.add_argument('--output-dir', default='', help='New directory for resplit outputs')
    parser.add_argument('--group-map', default='', help='TSV: accession and group_id; full coverage required')
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--two-group-policy', choices=['balanced', 'val', 'test'], default='balanced')
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.mode == 'resplit':
        resplit_existing(config, args)
        return
    index_dir = config.data.index_dir
    os.makedirs(index_dir, exist_ok=True)

    taxonomy_path = os.path.join(index_dir, 'taxonomy_db.pkl')
    acc_to_taxid_path = os.path.join(index_dir, 'acc_to_taxid.pkl')

    # ================================================================
    # Step 1 & 2: 加载 TaxonomyDB (推荐由 scripts/build_taxonomy.py 生成)
    # ================================================================
    if os.path.exists(taxonomy_path) and not args.force:
        print(f"[Preprocess] 加载已构建好的精简分类库: {taxonomy_path}")
        taxonomy_db = TaxonomyDB.load(taxonomy_path)
    else:
        print("=" * 60)
        print("[Notice] 未检测到 taxonomy_db.pkl 或指定了 --force。")
        print("建议先运行: python scripts/build_taxonomy.py --config " + args.config)
        print("正在进行基础加载...")
        print("=" * 60)
        taxonomy_db = TaxonomyDB()
        taxonomy_db.load_taxdump(config.data.taxdump_dir, target_domain=args.target_domain)

    # ================================================================
    builder = IndexBuilder(
        fna_file=config.data.fna_file,
        scores_file=config.data.scores_file,
        taxonomy_db=taxonomy_db,
        window_size_bp=config.data.window_size_bp,
        window_stride_bp=config.data.window_stride_bp,
    )

    # ================================================================
    # Step 3 & 4: 扫描 FNA 序列与评分偏移
    # ================================================================
    scan_path = os.path.join(index_dir, 'scan_results.pkl')
    need_scan = (args.mode in ('all', 'scan')) or (args.mode == 'index' and not os.path.exists(scan_path)) or args.force

    if need_scan:
        print("=" * 60)
        print("Step 3-4: Scanning FNA and Scores")
        print("=" * 60)

        sequences = builder.scan_fna()
        score_offsets = builder.scan_scores()

        # 保存扫描结果
        with open(scan_path, 'wb') as f:
            pickle.dump({
                'sequences': sequences,
                'score_offsets': score_offsets,
            }, f)
        print(f"Scan results saved: {scan_path}")
    elif args.mode == 'index':
        print(f"[Preprocess] 复用已有序列扫描缓存: {scan_path}")

    # ================================================================
    # Step 5: (可选) 转换 scores.txt → .scores.bin
    # ================================================================
    if args.mode in ('all', 'scores_bin'):
        scores_bin = os.path.join(index_dir, 'bacteria_scores.bin')
        if not os.path.exists(config.data.scores_file):
            print(f"[Preprocess] Warning: scores_file not found ({config.data.scores_file}), skipping scores_bin conversion.", flush=True)
        elif not os.path.exists(scores_bin) or args.force:
            print("=" * 60, flush=True)
            print("Step 5: Converting scores to binary", flush=True)
            print("=" * 60, flush=True)

            indices = ScoreReader.convert_to_binary(
                config.data.scores_file, scores_bin
            )
            with open(os.path.join(index_dir, 'scores_bin_index.pkl'), 'wb') as f:
                pickle.dump(indices, f)
        else:
            print(f"Binary scores already exist: {scores_bin} (use --force to overwrite)", flush=True)

    # ================================================================
    # Step 6 & 7: 构建窗口索引并 Stratified 划分
    # ================================================================
    if args.mode in ('all', 'index'):
        print("=" * 60)
        print("Step 6-7: Building window index")
        print("=" * 60)

        # 加载扫描结果
        with open(scan_path, 'rb') as f:
            scan = pickle.load(f)
        sequences = scan['sequences']
        score_offsets = scan['score_offsets']

        # 获取 accession → taxid 映射 (优先读取 build_taxonomy.py 生成的高精度缓存)
        acc_to_taxid = {}
        if os.path.exists(acc_to_taxid_path) and not args.force:
            print(f"[Preprocess] 加载 Accession→TaxID 映射表: {acc_to_taxid_path}")
            with open(acc_to_taxid_path, 'rb') as f:
                acc_to_taxid = pickle.load(f)
            print(f"  已成功读取 {len(acc_to_taxid):,} 条映射。")
        elif args.assembly_summary:
            acc_to_taxid = parse_assembly_summary(args.assembly_summary)
        else:
            acc_to_taxid = build_acc_to_taxid_from_headers(config.data.fna_file, taxonomy_db)

        # 构建预训练窗口索引 (不需要 scores 和 taxid)
        pretrain_index_path = os.path.join(index_dir, 'window_index_pretrain.pkl')
        if not os.path.exists(pretrain_index_path) or args.force:
            pretrain_windows = builder.build_windows(
                sequences, score_offsets, acc_to_taxid, mode='pretrain'
            )
            pretrain_splits = IndexBuilder.stratified_split(pretrain_windows)
            with open(pretrain_index_path, 'wb') as f:
                pickle.dump(pretrain_splits, f)
            print(f"Pretrain index saved: {pretrain_index_path}")
        else:
            print(f"[Preprocess] Pretrain index already exists, skipping: {pretrain_index_path}")

        # 构建微调窗口索引 (需要 scores 和 taxid)
        scores_bin_index_path = os.path.join(index_dir, 'scores_bin_index.pkl')
        scores_bin_path = os.path.join(index_dir, 'bacteria_scores.bin')
        has_scores = os.path.exists(config.data.scores_file) or (os.path.exists(scores_bin_path) and os.path.exists(scores_bin_index_path))

        if has_scores:
            # 优先使用二进制 memmap 评分索引 (float32 索引，微调 Dataset 高性能要求)
            if os.path.exists(scores_bin_index_path):
                print(f"[Preprocess] 加载二进制评分 memmap 偏移索引: {scores_bin_index_path}")
                with open(scores_bin_index_path, 'rb') as f:
                    bin_indices = pickle.load(f)
                # bin_indices 结构: [(acc, current_offset, length), ...]
                score_offsets_finetune = {
                    item[0]: (item[1], item[2]) for item in bin_indices
                }
            else:
                score_offsets_finetune = score_offsets

            finetune_windows = builder.build_windows(
                sequences, score_offsets_finetune, acc_to_taxid, mode='finetune'
            )
            finetune_splits = IndexBuilder.stratified_split(finetune_windows)

            finetune_index_path = os.path.join(index_dir, 'window_index_finetune.pkl')
            with open(finetune_index_path, 'wb') as f:
                pickle.dump(finetune_splits, f)
            print(f"Finetune index saved: {finetune_index_path}")
        else:
            print("[Preprocess] Warning: neither scores_file nor scores_bin found, skipping finetune index generation.")

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
