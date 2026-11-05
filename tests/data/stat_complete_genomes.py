#!/usr/bin/env python3
"""Read-only assembly inventory; stdlib only, no GPU or training dependencies."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re


def read_summary(path):
    fields = None
    seen = set()
    with open(path, encoding='utf-8-sig') as f:
        for line in f:
            header = line.lstrip('#').lstrip(' \t').rstrip('\r\n')
            if line.startswith('#') and header.split('\t', 1)[0] == 'assembly_accession':
                fields = header.split('\t')
                required = {'assembly_accession', 'assembly_level', 'species_taxid', 'taxid'}
                if not required.issubset(fields):
                    raise ValueError('Missing required assembly summary columns')
            elif line.startswith('#') or not line.strip():
                continue
            else:
                if fields is None:
                    raise ValueError('Missing # assembly_accession header')
                values = line.rstrip('\r\n').split('\t')
                if len(values) != len(fields):
                    raise ValueError('Malformed summary row: ' + values[0])
                row = dict(zip(fields, values))
                accession = row['assembly_accession']
                if accession in seen:
                    raise ValueError('Duplicate assembly accession: ' + accession)
                seen.add(accession)
                yield row


def inventory(root):
    paths = defaultdict(list)
    def fail(error):
        raise error
    for directory, _, files in os.walk(root, onerror=fail):
        for name in files:
            if not (name.endswith('_genomic.fna.gz') or name.endswith('_genomic.fna')):
                continue
            match = re.match(r'^(GC[AF]_\d+\.\d+)_', name)
            if match:
                paths[match.group(1)].append(str(Path(directory, name).resolve()))
    return paths


def resolve_file(row, paths):
    candidates = paths.get(row['assembly_accession'], [])
    basename = row.get('ftp_path', '').rstrip('/').split('/')[-1]
    exact = [p for p in candidates if Path(p).name in (
        basename + '_genomic.fna.gz', basename + '_genomic.fna')]
    candidates = exact or candidates
    if not candidates:
        return '', 'missing'
    if len(candidates) != 1:
        return '', 'ambiguous'
    return candidates[0], 'found'


def scan_fasta(path, window, stride):
    opener = gzip.open if path.endswith('.gz') else open
    lengths = []
    length = None
    with opener(path, 'rt', encoding='ascii') as f:
        for line in f:
            if line.startswith('>'):
                if length is not None:
                    lengths.append(length)
                length = 0
            elif line.strip():
                if length is None:
                    raise ValueError('Sequence before FASTA header')
                length += len(''.join(line.split()))
    if length is not None:
        lengths.append(length)
    if not lengths or any(n == 0 for n in lengths):
        raise ValueError('Empty FASTA or empty sequence')
    return dict(sequences=len(lengths), bases=sum(lengths),
                estimated_windows=sum((n - window) // stride + 1 for n in lengths if n >= window))


def distribution(counts):
    values = sorted(counts.values())
    total = sum(values)
    if not values:
        return {'n_species': 0, 'total': 0}
    def percentile(p):
        x = (len(values) - 1) * p
        low, high = math.floor(x), math.ceil(x)
        return values[low] + (values[high] - values[low]) * (x - low)
    def share(p):
        return sum(values[-max(1, math.ceil(len(values) * p)):]) / total if total else None
    return dict(n_species=len(values), total=total, min=values[0], median=percentile(.5),
                p90=percentile(.9), p99=percentile(.99), max=values[-1],
                top_1pct_species_share=share(.01), top_10pct_species_share=share(.1),
                species_with_1=sum(n == 1 for n in values),
                species_with_2=sum(n == 2 for n in values),
                species_with_ge3=sum(n >= 3 for n in values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assembly-summary', default='/home/user/guohongzhe/result/data/bacteria/assembly_summary.txt')
    parser.add_argument('--genomes-root', default='/home/user/guohongzhe/ftp.ncbi.nlm.nih.gov/genomes/refseq/bacteria')
    parser.add_argument('--output-dir', default='/home/user/zhangguoli/species_identification/tests/data/complete_stats_full', help='New directory, never overwrite')
    parser.add_argument('--scan-fasta', action='store_true', help='Read ALL matched local FASTAs, possibly slow')
    parser.add_argument('--window-size', type=int, default=4096)
    parser.add_argument('--stride', type=int, default=2048)
    args = parser.parse_args()
    if args.window_size <= 0 or args.stride <= 0:
        parser.error('Window size and stride must be positive')
    if not Path(args.genomes_root).is_dir():
        parser.error('Genomes root does not exist')
    rows = list(read_summary(args.assembly_summary))
    if not rows:
        raise ValueError('Empty summary')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    print(f'Inventory: {len(rows):,} summary assemblies; walking local directories...', flush=True)
    paths = inventory(args.genomes_root)
    groups = {k: Counter() for k in ('summary_all', 'summary_complete', 'local_all', 'local_complete')}
    window_groups = {'local_all': Counter(), 'local_complete': Counter()}
    species_rows = defaultdict(Counter)
    statuses, levels, versions = Counter(), Counter(), Counter()
    errors = []
    fields = ['assembly_accession', 'species_taxid', 'taxid', 'organism_name', 'assembly_level',
              'version_status', 'excluded_from_refseq', 'file_status', 'fna_path',
              'sequences', 'bases', 'estimated_windows', 'scan_status']
    with open(out / 'assemblies.tsv', 'w', encoding='utf-8', newline='') as f, \
            open(out / 'complete_candidates.tsv', 'w', encoding='utf-8', newline='') as c:
        writer = csv.DictWriter(f, fields, delimiter='\t', extrasaction='ignore')
        complete_writer = csv.DictWriter(c, fields, delimiter='\t', extrasaction='ignore')
        writer.writeheader()
        complete_writer.writeheader()
        for i, row in enumerate(rows, 1):
            row = dict(row)
            complete = row['assembly_level'] == 'Complete Genome'
            levels[row['assembly_level']] += 1
            versions[row.get('version_status', 'unknown')] += 1
            species = row['species_taxid']
            valid = species.isdigit() and int(species) > 0
            path, status = resolve_file(row, paths)
            statuses[('complete_' if complete else 'other_') + status] += 1
            row.update(fna_path=path, file_status=status, scan_status='not_scanned')
            keys = ['summary_all'] + (['summary_complete'] if complete else [])
            if status == 'found':
                keys += ['local_all'] + (['local_complete'] if complete else [])
            if valid:
                for key in keys:
                    groups[key][species] += 1
                    species_rows[species][key + '_assemblies'] += 1
            else:
                statuses['invalid_species_taxid'] += 1
            if args.scan_fasta and status == 'found':
                try:
                    row.update(scan_fasta(path, args.window_size, args.stride))
                    row['scan_status'] = 'ok'
                    if valid:
                        for key in ['local_all'] + (['local_complete'] if complete else []):
                            window_groups[key][species] += row['estimated_windows']
                            species_rows[species][key + '_windows'] += row['estimated_windows']
                except (OSError, EOFError, ValueError) as exc:
                    row['scan_status'] = 'error'
                    errors.append({'assembly': row['assembly_accession'], 'error': str(exc)})
            writer.writerow(row)
            if complete and status == 'found' and valid:
                complete_writer.writerow(row)
            if i % 1000 == 0:
                print(f'Processed {i:,}/{len(rows):,}', flush=True)
    columns = ['species_taxid'] + [k + '_assemblies' for k in groups]
    if args.scan_fasta:
        columns += [k + '_windows' for k in window_groups]
    with open(out / 'species_support.tsv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, columns, delimiter='\t')
        writer.writeheader()
        for species in sorted(species_rows, key=int):
            writer.writerow({'species_taxid': species,
                             **{k: species_rows[species][k] for k in columns[1:]}})
    report = dict(status='complete_with_scan_errors' if errors else 'complete',
                  summary_path=str(Path(args.assembly_summary).resolve()),
                  summary_sha256=hashlib.sha256(Path(args.assembly_summary).read_bytes()).hexdigest(),
                  genomes_root=str(Path(args.genomes_root).resolve()), assembly_levels=dict(levels),
                  version_statuses=dict(versions), file_statuses=dict(statuses),
                  assembly_distributions={k: distribution(v) for k, v in groups.items()},
                  window_distributions={k: distribution(v) for k, v in window_groups.items()} if args.scan_fasta else None,
                  local_species_lost=len(set(groups['local_all']) - set(groups['local_complete'])),
                  scan_errors=errors, window_size=args.window_size, stride=args.stride,
                  notes=['All summary rows retained; version/exclusion statuses reported, not silently filtered.',
                         'Species identified by summary species_taxid, not parsed organism names.',
                         'Complete Genome only; Chromosome excluded. Plasmids retained.',
                         'Window estimates per FASTA record: max(0, floor((length-window)/stride)+1).',
                         'Counts precede taxonomy validation, deduplication, split and training sampling.',
                         'Ambiguous local matches excluded; inspect assemblies.tsv. Failed scans are partial totals.',
                         'complete_candidates.tsv is an inventory, not an approved training manifest.'])
    with open(out / 'report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
