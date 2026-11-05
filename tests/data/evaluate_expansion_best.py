"""Evaluate 100/961-class checkpoints on val or test and summarize fixed groups."""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


def species_set(root, folder):
    with (root / folder / 'split_manifest.tsv').open(encoding='utf-8-sig') as f:
        return {int(r['species_taxid']) for r in csv.DictReader(f, delimiter='\t')}


def summarize(rows, groups):
    table = ['| 评估范围 | 正确／总数 | Top-1 | 宏平均 | Top-5 |',
             '|---|---:|---:|---:|---:|']
    for name, ids in groups:
        selected = [r for r in rows if int(r['taxid']) in ids]
        counts = Counter(int(r['taxid']) for r in selected)
        if set(counts) != ids or set(counts.values()) != {25}:
            raise ValueError(f'{name}: expected exactly 25 windows for every species')
        correct = Counter()
        for r in selected:
            correct[int(r['taxid'])] += int(r['correct'])
        n = len(selected)
        hits = sum(correct.values())
        macro = sum(correct[t] / counts[t] for t in ids) / len(ids)
        top5 = sum(bool(r['true_rank_in_top5']) for r in selected)
        table.append(f'| {name} | {hits:,}／{n:,} | {hits/n:.3%} | {macro:.3%} | {top5/n:.3%} |')
    return '\n'.join(table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['100', '961', 'both'], default='both')
    parser.add_argument('--split', choices=['val', 'test'], default='val',
                        help='test reuses the latest completed val evaluation snapshot')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    ids32 = species_set(root, 'index_bacteria_small32_train1000')
    ids100 = species_set(root, 'index_bacteria_small100_train1000')
    if len(ids32) != 32 or len(ids100) != 100 or not ids32 < ids100:
        raise ValueError('Original 32/100 species manifests do not match expected nesting')
    jobs = {
        '100': ('bacteria_small100_train1000_20epochs_seed42_v1_finetune',
                'finetune_small100_train1000_20epochs.yaml', 'index_bacteria_small100_train1000'),
        '961': ('bacteria_species961_train1000_seed42_v1_finetune',
                'finetune_species961_train1000_seed42.yaml', 'index_bacteria_species961_train1000'),
    }
    selected_jobs = jobs if args.model == 'both' else {args.model: jobs[args.model]}
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    # Snapshot both weights before inference so each evaluated version is recorded.
    prepared = []
    for model, (run, config_name, index_name) in selected_jobs.items():
        source = root / 'runs' / run / 'best_model.pt'
        config_path = root / 'configs' / config_name
        reference = None
        if args.split == 'test':
            candidates = sorted((root / 'runs' / run).glob('eval_fixed_val_*'), reverse=True)
            reference = next((p for p in candidates if (p / 'summary.md').is_file()), None)
            if reference is None:
                raise FileNotFoundError(f'{model}: no completed val evaluation; run val first')
            source = reference / 'best_model_snapshot.pt'
            config_path = reference / 'config_eval.yaml'
            print(f'[{model}] Reusing val snapshot: {reference}', flush=True)
        if not source.is_file() or not config_path.is_file():
            raise FileNotFoundError(f'Check checkpoint/config: {source}; {config_path}')
        ids = species_set(root, index_name)
        if len(ids) != int(model) or not ids100 <= ids:
            raise ValueError(f'{model}: species manifest mismatch')
        out = root / 'runs' / run / (f'eval_fixed_{args.split}_' + stamp)
        out.mkdir()
        snapshot = out / 'best_model_snapshot.pt'
        before = source.stat()
        shutil.copy2(source, snapshot)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError('Checkpoint changed during copy; rerun after checkpoint saving finishes')
        digest = hashlib.sha256()
        with snapshot.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(block)
        if reference is not None:
            expected = json.loads((reference / 'snapshot.json').read_text())['sha256']
            if digest.hexdigest() != expected:
                raise ValueError(f'{model}: snapshot hash differs from val; refusing test evaluation')
        (out / 'snapshot.json').write_text(json.dumps(dict(
            source=str(source), snapshot=str(snapshot), sha256=digest.hexdigest(),
            source_mtime_ns=after.st_mtime_ns, captured_at=stamp,
            validation_reference=str(reference) if reference else None), indent=2))
        cfg = yaml.safe_load(config_path.read_text())
        cfg['data']['num_workers'] = 0
        local_config = out / 'config_eval.yaml'
        local_config.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        prepared.append((model, ids, out, snapshot, local_config))
        print(f'[{model}] Snapshot: {snapshot}', flush=True)
    for model, ids, out, snapshot, config in prepared:
        result = out / 'results.json'
        subprocess.run([sys.executable, '-u', str(root / 'scripts/evaluate.py'),
                        '--config', str(config), '--checkpoint', str(snapshot),
                        '--split', args.split, '--gpus', '0', '--batch-size', '4',
                        '--input-mode', 'sequence_only', '--max-samples', '0',
                        '--seed', '42', '--window-report', '--output', str(result)],
                       cwd=root, check=True)
        with (out / 'results_windows.csv').open(encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        groups = [(f'全部{model}物种', ids)]
        if model == '961':
            groups += [('原100物种', ids100), ('新增861物种', ids - ids100)]
        groups += [('原32物种', ids32), ('原100中的新增68物种', ids100 - ids32)]
        settings = json.loads(result.read_text())['evaluation_settings']
        split_label = '验证集' if args.split == 'val' else '测试集'
        report = (f"模型：{model} 类；checkpoint Epoch {settings['checkpoint_epoch']}；"
                  f'完整{split_label}；纯 DNA；预测保留完整分类空间。\n\n' + summarize(rows, groups))
        (out / 'summary.md').write_text(report + '\n', encoding='utf-8')
        print('\n' + report + f'\n\n结果目录：{out}', flush=True)


if __name__ == '__main__':
    main()
