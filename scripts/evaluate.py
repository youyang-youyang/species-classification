"""
评估入口脚本

在测试集上计算各层级 Top-1/Top-5 准确率，
并可选与 Kraken2 / KmerFormer 进行对比评估。

用法:
  python scripts/evaluate.py \
      --config configs/finetune_bacteria.yaml \
      --checkpoint runs/finetune/best_model.pt \
      --split test
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict, Counter
from tqdm import tqdm
import json
import pickle
import csv
import hashlib
from array import array


def select_evaluation_windows(windows, taxonomy_db, sampling, per_species,
                              max_samples, seed, manifest_path, split,
                              allowed_species=None):
    """Fingerprint ordered split and labels; balanced sampling rotates accessions."""
    digest = hashlib.sha256()
    labels = {}
    groups = {}
    reuse = bool(manifest_path and os.path.exists(manifest_path))
    for i, (info, start, offset, taxid) in enumerate(tqdm(windows, desc='Auditing evaluation split')):
        if taxid not in labels:
            labels[taxid] = int(taxonomy_db.encode_lineage(taxid).get('species', -1))
        label = labels[taxid]
        digest.update(json.dumps([info.accession, int(start), int(offset), int(taxid), label],
                                 separators=(',', ':')).encode('utf-8') + b'\n')
        if (sampling == 'species_balanced' and not reuse and label >= 0
                and (allowed_species is None or label in allowed_species)):
            by_accession = groups.setdefault(label, {})
            if info.accession not in by_accession:
                by_accession[info.accession] = array('Q')
            by_accession[info.accession].append(i)
    digest.update(json.dumps(taxonomy_db.idx2label['species'], ensure_ascii=True).encode())
    settings = dict(version=1, split=split, split_size=len(windows),
                    split_sha256=digest.hexdigest(), sampling=sampling,
                    windows_per_species=per_species if sampling == 'species_balanced' else None,
                    max_samples=max_samples, seed=seed)
    if allowed_species is not None:
        settings['allowed_species'] = sorted(allowed_species)
    if reuse:
        with open(manifest_path, encoding='utf-8') as f:
            manifest = json.load(f)
        if manifest['settings'] != settings:
            raise ValueError('Sample manifest does not match split, labels or sampling settings')
        indices = manifest['indices']
        if (not indices or any(type(i) is not int or i < 0 or i >= len(windows) for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError('Invalid or duplicate manifest indices')
    else:
        rng = np.random.default_rng(seed)
        if sampling == 'random':
            indices = (rng.choice(len(windows), size=max_samples, replace=False).tolist()
                       if 0 < max_samples < len(windows) else list(range(len(windows))))
        else:
            indices = []
            for label in sorted(groups):
                accession_groups = groups[label]
                accessions = sorted(accession_groups)
                # Pick at most K distinct accessions first, then fill by rounds.
                order = rng.permutation(len(accessions))
                pools = []
                for j in order:
                    source = accession_groups[accessions[int(j)]]
                    take = min(per_species, len(source))
                    pool = [source[int(k)] for k in rng.choice(len(source), size=take, replace=False)]
                    pools.append(pool)
                    if len(pools) >= per_species:
                        break
                selected = []
                while len(selected) < per_species and pools:
                    remaining = []
                    for pool in pools:
                        selected.append(pool.pop())
                        if pool:
                            remaining.append(pool)
                        if len(selected) == per_species:
                            break
                    pools = remaining
                indices.extend(selected)
            rng.shuffle(indices)
        if not indices:
            raise ValueError('No eligible evaluation windows')
        manifest = dict(settings=settings, indices=indices)
        if manifest_path:
            # Exclusive creation prevents concurrent jobs silently replacing a manifest.
            with open(manifest_path, 'x', encoding='utf-8') as f:
                json.dump(manifest, f)
    selected = [windows[i] for i in indices]
    support = Counter(labels[w[3]] for w in selected if labels[w[3]] >= 0)
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    report = dict(settings, manifest_sha256=manifest_hash,
                  n_evaluated_species=len(support),
                  n_species_not_evaluated=len(taxonomy_db.idx2label['species']) - len(support),
                  per_species_sample_min=min(support.values()) if support else 0,
                  per_species_sample_max=max(support.values()) if support else 0,
                  n_species_below_requested=sum(n < per_species for n in support.values())
                      if sampling == 'species_balanced' else None)
    return selected, report


def select_diagnostic_species(counts, per_bin, seed):
    """Choose shared train/val species, stratified by training support only."""
    if per_bin <= 0:
        raise ValueError('Diagnostic species count must be positive')
    rng = np.random.default_rng(seed)
    selected = []
    bins = []
    for lower, upper in ((1, 100), (100, 1000), (1000, 10000),
                         (10000, 100000), (100000, None)):
        eligible = sorted(label for label, count in counts['train'].items()
                          if label >= 0 and count >= lower
                          and (upper is None or count < upper)
                          and counts['val'].get(label, 0) > 0)
        chosen = sorted(int(x) for x in rng.choice(
            eligible, size=min(per_bin, len(eligible)), replace=False))
        selected.extend(chosen)
        bins.append(dict(train_window_range=f'[{lower}, {upper if upper else "inf"})',
                         eligible_species=len(eligible), selected_species=chosen))
    if not selected:
        raise ValueError('No species shared by train and val')
    return set(selected), bins


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def replay_seen_pair(index_data, taxonomy_db, config, checkpoint, per_species, seed):
    """Replay whole completed epochs; counts verify species exposure, not old file identity."""
    from scripts.finetune import MixedSpeciesSampler
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    epoch, updates = int(ckpt['epoch']), int(ckpt['global_step'])
    recorded = {int(k): int(v) for k, v in ckpt['species_exposure'].items()}
    del ckpt
    cfg = config.training
    steps, batch = cfg.steps_per_epoch, cfg.batch_size_per_gpu
    if (not steps or steps % cfg.gradient_accumulation_steps
            or cfg.species_balanced_fraction <= 0 or epoch < 0
            or updates != (epoch + 1) * steps // cfg.gradient_accumulation_steps):
        raise ValueError('Replay requires unchanged single-GPU mixed sampling and complete epochs from epoch 0')
    train = index_data['train']
    sampler = MixedSpeciesSampler(train, taxonomy_db, steps * batch,
                                  cfg.species_balanced_fraction, config.seed)
    exposures, counts, label_cache = Counter(), Counter(), {}
    def label(window):
        taxid = int(window[3])
        if taxid not in label_cache:
            label_cache[taxid] = int(taxonomy_db.encode_lineage(taxid)['species'])
        return label_cache[taxid]
    for e in range(epoch + 1):
        sampler.set_epoch(e)
        for i in sampler:
            exposures[i] += 1
            counts[label(train[i])] += 1
        print(f'[Replay] epoch={e}, inputs={sum(counts.values()):,}', flush=True)
    if any(counts[k] != recorded.get(k, 0) for k in set(counts) | set(recorded)):
        raise ValueError('Replayed species counts differ from checkpoint; refusing manifest')
    del sampler
    val_species = {label(w) for w in index_data['val']}
    shared = {k for k, n in counts.items() if n > 0 and k >= 0} & val_species
    seen_indices = sorted(exposures)
    seen_windows = [train[i] for i in seen_indices]
    selected_train, train_report = select_evaluation_windows(
        seen_windows, taxonomy_db, 'species_balanced', per_species, 0, seed,
        None, 'train', allowed_species=shared)
    selected_val, val_report = select_evaluation_windows(
        index_data['val'], taxonomy_db, 'species_balanced', per_species, 0, seed,
        None, 'val', allowed_species=shared)
    # Selection returns the original tuple objects, preserving duplicate-index identity.
    train_ids = {id(w) for w in selected_train}
    val_ids = {id(w) for w in selected_val}
    ti = [i for i in seen_indices if id(train[i]) in train_ids]
    vi = [i for i, w in enumerate(index_data['val']) if id(w) in val_ids]
    if len(ti) != len(selected_train) or len(vi) != len(selected_val):
        raise ValueError('Aliased index entries; cannot map selected windows unambiguously')
    train_accessions = {train[i][0].accession for i in ti}
    if train_accessions & {index_data['val'][i][0].accession for i in vi}:
        raise ValueError('Selected train and validation accessions overlap')
    return dict(version=1, checkpoint_epoch=epoch, checkpoint_global_step=updates,
                window_presentations=sum(counts.values()), unique_seen_windows=len(exposures),
                n_shared_species=len(shared), windows_per_species=per_species, seed=seed,
                indices={'train': ti, 'val': vi},
                selected_train_exposures=[exposures[i] for i in ti],
                reports={'train': train_report, 'val': val_report},
                note='Reconstructed using current sampler/config/index/taxonomy. Species counts match '
                     'checkpoint; historical file identity and unchanged single-GPU settings are assumptions. '
                     'Not a contemporaneous per-window training log. Overlapping windows may remain.')


def count_species_windows(index_data, taxonomy_db):
    """Count indexed windows, not actual training exposures; do not scan test."""
    counts = {}
    for split in ('train', 'val'):
        taxid_counts = Counter(int(window[3]) for window in tqdm(
            index_data.get(split, []), desc=f'Counting {split} windows'))
        species_counts = Counter()
        for taxid, count in taxid_counts.items():
            species = taxonomy_db.encode_lineage(taxid).get('species', -1)
            species_counts[int(species)] += count
        counts[split] = species_counts
        print(f'[Counts] {split}: {sum(species_counts.values()):,} windows; '
              f'invalid species labels: {species_counts.get(-1, 0):,}')
    return counts


def summarize_species_support(counts, details, species_names):
    """Keep unevaluated species distinct from evaluated species with zero accuracy."""
    rows = []
    for label, name in enumerate(species_names):
        detail = details.get(str(label), {})
        n = int(detail.get('n_samples', 0))
        correct = int(detail.get('n_correct', 0))
        rows.append(dict(species_label=label, species_name=name,
                         train_windows=counts['train'].get(label, 0),
                         val_windows=counts['val'].get(label, 0),
                         eval_windows=n, eval_correct=correct,
                         eval_accuracy=correct / n if n else None))
    groups = []
    for lower, upper in ((0, 1), (1, 100), (100, 1000), (1000, 10000),
                         (10000, 100000), (100000, None)):
        members = [r for r in rows if r['train_windows'] >= lower
                   and (upper is None or r['train_windows'] < upper)]
        evaluated = [r for r in members if r['eval_windows'] > 0]
        n = sum(r['eval_windows'] for r in evaluated)
        reliable = [r for r in evaluated if r['eval_windows'] >= 10]
        groups.append(dict(
            train_window_range=f'[{lower}, {upper if upper is not None else "inf"})',
            n_species=len(members), n_evaluated_species=len(evaluated),
            train_windows=sum(r['train_windows'] for r in members),
            eval_windows=n,
            micro_accuracy=sum(r['eval_correct'] for r in evaluated) / n if n else None,
            macro_accuracy=sum(r['eval_accuracy'] for r in evaluated) / len(evaluated)
                if evaluated else None,
            n_species_eval_ge10=len(reliable),
            macro_accuracy_eval_ge10=sum(r['eval_accuracy'] for r in reliable) / len(reliable)
                if reliable else None))
    return rows, groups

from src.config import Config
from src.model.model import build_model
from src.data.dataset import FinetuneDataset, collate_finetune
from src.data.fna_reader import FNAReader
from src.data.score_reader import ScoreReader
from src.data.taxonomy import TaxonomyDB


def write_window_reports(output_path, windows, rankings, taxonomy_db, window_size, index_dir, split):
    """Join sequential evaluation predictions to source windows; no dense confusion matrix."""
    if len(windows) != len(rankings):
        raise ValueError('Prediction/window count mismatch')
    names = taxonomy_db.idx2label['species']
    sources = {}
    manifest = os.path.join(index_dir, 'split_manifest.tsv')
    if os.path.exists(manifest):
        with open(manifest, encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f, delimiter='\t'):
                uid = row.get('internal_id')
                if uid:
                    if uid in sources:
                        raise ValueError('Duplicate internal_id in source manifest')
                    sources[uid] = row
    prefix = os.path.splitext(output_path)[0]
    fields = ['evaluation_row', 'split', 'internal_id', 'assembly_accession',
              'sequence_accession', 'source_fna', 'start_0based', 'end_exclusive',
              'taxid', 'true_species', 'predicted_species', 'correct', 'true_rank_in_top5',
              'top5_species']
    confusion = Counter()
    with open(prefix + '_windows.csv', 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, (window, ranking) in enumerate(zip(windows, rankings)):
            info, start, _, taxid = window
            label = taxonomy_db.encode_lineage(taxid)['species']
            if not ranking or any(p < 0 or p >= len(names) for p in ranking):
                raise ValueError('Invalid species prediction')
            source = sources.get(info.accession, {})
            if source and source['split'] != split:
                raise ValueError('Source manifest split mismatch')
            valid = 0 <= label < len(names)
            pred = ranking[0]
            if valid:
                confusion[(label, pred)] += 1
            writer.writerow(dict(evaluation_row=i, split=split, internal_id=info.accession,
                assembly_accession=source.get('assembly_accession', ''),
                sequence_accession=source.get('sequence_accession', info.accession),
                source_fna=source.get('fna_path', ''), start_0based=start,
                end_exclusive=start + window_size, taxid=taxid,
                true_species=names[label] if valid else '', predicted_species=names[pred],
                correct=int(label == pred) if valid else '',
                true_rank_in_top5=ranking.index(label) + 1 if label in ranking else '',
                top5_species=json.dumps([names[p] for p in ranking], ensure_ascii=False)))
    with open(prefix + '_confusion.csv', 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['true_species', 'predicted_species', 'count', 'correct'])
        for (label, pred), count in sorted(confusion.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow([names[label], names[pred], count, int(label == pred)])
    return {'windows_csv': prefix + '_windows.csv', 'confusion_csv': prefix + '_confusion.csv',
            'source_manifest_matched_windows': sum(w[0].accession in sources for w in windows),
            'note': 'Rows follow sequential evaluation order. Coordinates are 0-based, end-exclusive; '
                    'blank true_rank_in_top5 means absent or invalid label. Confusion includes correct predictions.'}


def compute_topk_accuracy(logits: torch.Tensor, labels: torch.Tensor,
                          k: int = 5) -> float:
    """计算 Top-K 准确率"""
    valid = labels >= 0
    if valid.sum() == 0:
        return 0.0
    logits_v = logits[valid]
    labels_v = labels[valid]
    _, topk_preds = logits_v.topk(min(k, logits_v.size(-1)), dim=-1)
    correct = (topk_preds == labels_v.unsqueeze(-1)).any(dim=-1)
    return correct.float().mean().item()


@torch.no_grad()
def evaluate(model, dataloader, device, taxonomy_db, input_mode='full', window_report=False):
    """
    全面评估

    Returns:
        results: dict 包含各层级 Top-1/Top-5 准确率和混淆矩阵统计
    """
    model.eval()

    # 累积各级预测和标签
    all_preds = defaultdict(list)
    all_labels = defaultdict(list)
    all_top5_correct = defaultdict(list)
    species_rankings = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        bases_21 = batch['bases_21'].to(device)
        uniqueness_scores = batch['uniqueness_scores']
        if uniqueness_scores is not None:
            uniqueness_scores = uniqueness_scores.to(device)
        phylo_vecs = batch['phylo_vecs'].to(device) if 'phylo_vecs' in batch else None
        labels = {k: v.to(device) for k, v in batch['labels'].items()}

        if input_mode in ('no_phylo', 'sequence_only'):
            phylo_vecs = None
        if input_mode in ('no_uniqueness', 'sequence_only'):
            uniqueness_scores = None

        logits, Z, _, _ = model.forward_finetune(
            bases_21=bases_21,
            uniqueness_scores=uniqueness_scores,
            phylo_vecs=phylo_vecs,
        )

        for level in logits:
            if level in labels:
                top5 = logits[level].topk(min(5, logits[level].size(-1)), dim=-1).indices
                all_top5_correct[level].append((top5 == labels[level].unsqueeze(-1)).any(dim=-1).cpu())
                all_preds[level].append(logits[level].argmax(dim=-1).cpu())
                all_labels[level].append(labels[level].cpu())
                if window_report and level == 'species':
                    species_rankings.extend(top5.cpu().tolist())

    # 计算指标
    results = {}
    if window_report:
        results['_species_rankings'] = species_rankings
    for level in all_preds:
        preds = torch.cat(all_preds[level])
        labels = torch.cat(all_labels[level])
        valid = labels >= 0

        if valid.sum() == 0:
            continue

        # Top-1 准确率
        top1 = (preds[valid] == labels[valid]).float().mean().item()
        # Top-5 准确率
        top5 = torch.cat(all_top5_correct[level])[valid].float().mean().item()

        n_valid = valid.sum().item()
        results[level] = {
            'top1_acc': top1,
            'top5_acc': top5,
            'n_samples': n_valid,
        }

        print(f"  {level:>10s}: Top-1 {top1*100:.2f}% | Top-5 {top5*100:.2f}% "
              f"| N={n_valid:,}")

    # 物种级别的详细分析
    if 'species' in all_preds:
        preds = torch.cat(all_preds['species'])
        labels = torch.cat(all_labels['species'])
        valid = labels >= 0
        preds_v = preds[valid]
        labels_v = labels[valid]

        # 每个物种的准确率
        species_correct = defaultdict(int)
        species_total = defaultdict(int)
        for p, l in zip(preds_v.numpy(), labels_v.numpy()):
            species_total[l] += 1
            if p == l:
                species_correct[l] += 1

        per_species_acc = {
            sp: species_correct[sp] / species_total[sp]
            for sp in species_total
            if species_total[sp] > 0
        }
        results['per_species_acc'] = per_species_acc
        results['per_species_details'] = {
            str(int(sp)): {'n_samples': int(species_total[sp]),
                           'n_correct': int(species_correct[sp]),
                           'accuracy': float(per_species_acc[sp])}
            for sp in species_total
        }

        # 统计
        accs = list(per_species_acc.values())
        if accs:
            results['species_macro_acc'] = np.mean(accs)
            results['species_median_acc'] = np.median(accs)
            results['n_species_zero'] = sum(1 for a in accs if a == 0)
            results['n_species_perfect'] = sum(1 for a in accs if a == 1.0)

            print(f"\n  Species Macro Avg: {np.mean(accs)*100:.2f}%")
            print(f"  Species Median:    {np.median(accs)*100:.2f}%")
            print(f"  Species w/ 0% acc: {results['n_species_zero']}")
            print(f"  Species w/ 100%:   {results['n_species_perfect']}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Model Evaluation')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--prepare-only', action='store_true',
                        help='CPU only: prepare manifest and coverage report; no model forward')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path for results')
    parser.add_argument('--input-mode', choices=['full', 'no_phylo', 'no_uniqueness', 'sequence_only'],
                        default='full', help='Inputs retained during evaluation')
    parser.add_argument('--max-samples', type=int, default=0,
                        help='Random window subset size; 0 evaluates the entire split')
    parser.add_argument('--seed', type=int, default=42,
                        help='Subset seed; keep identical across input modes')
    parser.add_argument('--species-support-report', action='store_true',
                        help='Count train/val windows per species and save accuracy/support CSV')
    parser.add_argument('--window-report', action='store_true',
                        help='Save per-window species Top-5, source coordinates and sparse confusion CSV')
    parser.add_argument('--sampling', choices=['random', 'species_balanced'], default='random')
    parser.add_argument('--windows-per-species', type=int, default=20)
    parser.add_argument('--sample-manifest', help='Create or reuse a fixed JSON sampling manifest')
    parser.add_argument('--diagnostic-species-per-bin', type=int, default=0,
                        help='Paired train/val diagnostic: select this many shared species per support bin')
    parser.add_argument('--prepare-seen-pair', metavar='JSON',
                        help='CPU replay of mixed sampling; write paired seen-train/val manifest')
    parser.add_argument('--seen-pair-manifest', metavar='JSON',
                        help='Evaluate fixed replay-derived train or val windows')
    args = parser.parse_args()
    if args.prepare_seen_pair or args.seen_pair_manifest:
        if (not args.checkpoint or args.split == 'test' or args.prepare_only
                or args.sample_manifest or args.diagnostic_species_per_bin or args.max_samples
                or args.sampling != 'random'
                or (args.prepare_seen_pair and args.seen_pair_manifest)):
            parser.error('Seen-pair mode needs checkpoint and train/val; omit other sampling/prepare flags')
        if args.seen_pair_manifest and args.input_mode != 'sequence_only':
            parser.error('Seen-pair diagnostic requires --input-mode sequence_only')
    if args.prepare_only and not args.sample_manifest:
        parser.error('--prepare-only requires --sample-manifest')
    if not args.prepare_only and not args.checkpoint:
        parser.error('--checkpoint is required unless --prepare-only')
    if args.max_samples < 0:
        parser.error('--max-samples must be non-negative')
    if args.windows_per_species <= 0:
        parser.error('--windows-per-species must be positive')
    if args.sampling == 'species_balanced' and args.max_samples:
        parser.error('species_balanced uses --windows-per-species; omit --max-samples')
    if args.sampling == 'species_balanced' and not args.sample_manifest:
        parser.error('species_balanced requires --sample-manifest for reproducibility')
    if args.diagnostic_species_per_bin < 0:
        parser.error('--diagnostic-species-per-bin must be non-negative')
    if args.diagnostic_species_per_bin and (args.split == 'test' or args.sampling != 'species_balanced'):
        parser.error('Paired diagnostic requires train/val and species_balanced sampling')
    if args.split == 'train' and not (args.diagnostic_species_per_bin or args.seen_pair_manifest or args.prepare_seen_pair):
        parser.error('Train evaluation requires --diagnostic-species-per-bin; it is diagnostic, not validation')

    config = Config.from_yaml(args.config)
    if config.data.dna_only and not (args.prepare_only or args.prepare_seen_pair) and args.input_mode != 'sequence_only':
        parser.error('DNA-only data requires --input-mode sequence_only')

    # 加载分类数据库
    taxonomy_path = os.path.join(config.data.index_dir, 'taxonomy_db.pkl')
    taxonomy_db = TaxonomyDB.load(taxonomy_path)

    # 加载窗口索引
    index_path = os.path.join(config.data.index_dir, 'window_index_finetune.pkl')
    with open(index_path, 'rb') as f:
        index_data = pickle.load(f)
    seen_pair = None
    if args.prepare_seen_pair or args.seen_pair_manifest:
        print('[Replay] Checking checkpoint, index and taxonomy fingerprints (CPU)...', flush=True)
        fingerprints = {name: file_sha256(path) for name, path in (
            ('checkpoint', args.checkpoint), ('index', index_path), ('taxonomy', taxonomy_path))}
        if args.prepare_seen_pair:
            if os.path.exists(args.prepare_seen_pair):
                raise FileExistsError(args.prepare_seen_pair)
            seen_pair = replay_seen_pair(index_data, taxonomy_db, config,
                                        args.checkpoint, args.windows_per_species, args.seed)
            seen_pair['fingerprints'] = fingerprints
            seen_pair['config_sha256'] = file_sha256(args.config)
            with open(args.prepare_seen_pair, 'x', encoding='utf-8') as f:
                json.dump(seen_pair, f, ensure_ascii=False)
            print('[Replay] Saved ' + args.prepare_seen_pair, flush=True)
            print(json.dumps({k: seen_pair[k] for k in (
                'checkpoint_epoch', 'window_presentations', 'unique_seen_windows', 'n_shared_species')}))
            print(f"[Replay] train={len(seen_pair['indices']['train'])}, val={len(seen_pair['indices']['val'])}")
            return
        with open(args.seen_pair_manifest, encoding='utf-8') as f:
            seen_pair = json.load(f)
        if seen_pair['version'] != 1 or seen_pair['fingerprints'] != fingerprints:
            raise ValueError('Seen-pair checkpoint/index/taxonomy mismatch')
    support_counts = count_species_windows(index_data, taxonomy_db) if (
        args.species_support_report or args.diagnostic_species_per_bin) else None
    diagnostic_species = None
    diagnostic_bins = None
    if args.diagnostic_species_per_bin:
        diagnostic_species, diagnostic_bins = select_diagnostic_species(
            support_counts, args.diagnostic_species_per_bin, args.seed)
        print(f'[Diagnostic] {len(diagnostic_species)} shared train/val species; '
              'train-index windows are not necessarily previously seen by the model.', flush=True)
    windows = index_data.get(args.split, [])
    print(f"[Evaluate] {args.split} set: {len(windows):,} windows")
    if not windows:
        raise ValueError(f"No windows in split: {args.split}")
    sampling_report = None
    if seen_pair is not None:
        indices = seen_pair['indices'][args.split]
        if (not indices or len(set(indices)) != len(indices)
                or any(type(i) is not int or i < 0 or i >= len(windows) for i in indices)):
            raise ValueError('Invalid seen-pair indices')
        windows = [windows[i] for i in indices]
        sampling_report = seen_pair['reports'][args.split]
    elif args.sample_manifest or args.sampling == 'species_balanced':
        windows, sampling_report = select_evaluation_windows(
            windows, taxonomy_db, args.sampling, args.windows_per_species,
            args.max_samples, args.seed, args.sample_manifest, args.split,
            allowed_species=diagnostic_species)
        print('[Sampling] ' + json.dumps(sampling_report))
    elif args.max_samples and args.max_samples < len(windows):
        indices = np.random.default_rng(args.seed).choice(
            len(windows), size=args.max_samples, replace=False)
        windows = [windows[int(i)] for i in indices]
    print(f"[Evaluate] mode={args.input_mode}, samples={len(windows):,}, seed={args.seed}")

    if args.prepare_only:
        # No sequence readers, model, checkpoint or CUDA initialization on this path.
        cache = {}
        def coverage(items):
            counts = Counter()
            for window in items:
                taxid = window[3]
                if taxid not in cache:
                    cache[taxid] = taxonomy_db.encode_lineage(taxid).get('species', -1)
                counts[int(cache[taxid])] += 1
            return {'n_windows': sum(counts.values()),
                    'n_valid_species': sum(label >= 0 for label in counts),
                    'invalid_windows': counts.get(-1, 0),
                    'species_window_counts': dict(sorted(counts.items()))}
        prefix_size = (config.training.val_steps * config.training.batch_size_per_gpu
                       if config.training.val_steps else len(index_data[args.split]))
        report = {'sampling': sampling_report, 'selected': coverage(windows),
                  'current_validation_prefix_size': prefix_size,
                  'current_validation_prefix': coverage(index_data[args.split][:prefix_size]),
                  'note': 'Coverage only, no accuracy or actual training-exposure claims. '
                          'Prefix assumes single GPU and configured training batch size. '
                          'Manifest is not automatically used by finetune.py.'}
        output_path = args.output or args.sample_manifest + '.coverage.json'
        if os.path.abspath(output_path) == os.path.abspath(args.sample_manifest):
            raise ValueError('Coverage output must differ from sample manifest')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f'[Prepare only] CPU preparation complete: {output_path}', flush=True)
        return

    device = torch.device(f'cuda:{args.gpus}')

    # Reader
    fna_reader = FNAReader(config.data.fna_file)
    score_reader = None if config.data.dna_only else ScoreReader(config.data.scores_file, config.data.scores_bin_file)

    # 加载 GTDB 进化向量查找表
    phylo_dict = None
    phylo_path = getattr(config.data, 'phylo_file', '')
    if args.input_mode in ('full', 'no_uniqueness') and phylo_path and os.path.exists(phylo_path):
        phylo_dict = torch.load(phylo_path, map_location='cpu', weights_only=False)
    elif args.input_mode in ('full', 'no_uniqueness'):
        raise FileNotFoundError(f"{args.input_mode} evaluation requires phylo_file: {phylo_path}")

    # Dataset
    dataset = FinetuneDataset(
        window_index=windows,
        fna_reader=fna_reader,
        score_reader=score_reader,
        taxonomy_db=taxonomy_db,
        window_size_bp=config.data.window_size_bp,
        kmer_size=config.model.kmer_size,
        phylo_dict=phylo_dict,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=config.data.num_workers,
        collate_fn=collate_finetune,
    )

    # 模型
    model = build_model(
        config=config.model,
        mode='finetune',
        num_classes_per_level=taxonomy_db.num_classes,
    )
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state', ckpt)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model = model.to(device)

    print(f"\n{'='*60}")
    print(f"Evaluating on {args.split} set")
    print(f"{'='*60}")

    results = evaluate(model, dataloader, device, taxonomy_db,
                       input_mode=args.input_mode, window_report=args.window_report)
    results['evaluation_settings'] = {
        'index_path': os.path.abspath(index_path),
        'max_samples': args.max_samples,
        'input_mode': args.input_mode,
        'split': args.split,
        'n_samples': len(windows),
        'seed': args.seed,
        'checkpoint': args.checkpoint,
        'checkpoint_epoch': ckpt.get('epoch'),
        'checkpoint_global_step': ckpt.get('global_step'),
        'checkpoint_window_presentations': (
            sum(ckpt['species_exposure'].values())
            if 'species_exposure' in ckpt else None),
        'sampling': 'replayed_seen_pair' if seen_pair is not None else args.sampling,
        'seen_pair_manifest_sha256': file_sha256(args.seen_pair_manifest) if seen_pair is not None else None,
        'seen_pair_note': seen_pair['note'] if seen_pair is not None else None,
        'sampling_report': sampling_report,
        'diagnostic_species_bins': diagnostic_bins,
        'diagnostic_note': ('Training-index evaluation does not prove actual prior exposure. '
                            'Do not use this selected subset as the overall validation score.')
                            if diagnostic_species is not None else None,
    }

    # 保存结果
    output_path = args.output or os.path.join(
        os.path.dirname(args.checkpoint),
        f'eval_{args.split}_{args.input_mode}'
        f'{"_diagnostic" if args.diagnostic_species_per_bin else ""}'
        f'{"_species_balanced" if args.sampling == "species_balanced" else ""}_results.json'
    )

    if args.window_report:
        results['window_reports'] = write_window_reports(
            output_path, windows, results.pop('_species_rankings'), taxonomy_db,
            config.data.window_size_bp, os.path.dirname(index_path), args.split)
        print('[Window reports] ' + json.dumps(results['window_reports']))

    if support_counts is not None:
        rows, groups = summarize_species_support(
            support_counts, results.get('per_species_details', {}),
            taxonomy_db.idx2label['species'])
        results['species_support_groups'] = groups
        results['species_support_note'] = (
            'Counts are indexed windows, not actual training exposures. '
            'Accuracy is on the selected evaluation split; null means not evaluated. '
            'Macro averages include evaluated species only; windows may overlap.')
        csv_path = os.path.splitext(output_path)[0] + '_species_support.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'species_label', 'species_name', 'train_windows', 'val_windows',
                'eval_windows', 'eval_correct', 'eval_accuracy'])
            writer.writeheader()
            writer.writerows(rows)
        print(f'Species support report: {csv_path}')
        for group in groups:
            print('[Species support] ' + json.dumps(group))

    # 移除不可序列化的内容
    serializable = {k: v for k, v in results.items()
                    if k != 'per_species_acc'}
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved: {output_path}")


if __name__ == '__main__':
    main()
