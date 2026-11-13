#!/usr/bin/env python3
"""Validate the frozen split and build a train-only, strand-specific KMC database."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import subprocess


def records(path):
    name, chunks = None, []
    with open(path, 'rb') as stream:
        for line in stream:
            if line.startswith(b'>'):
                if name is not None:
                    yield name, b''.join(chunks).upper()
                name, chunks = line[1:].split()[0].decode('ascii'), []
            else:
                chunks.append(line.strip())
    if name is not None:
        yield name, b''.join(chunks).upper()


def run(args):
    print('RUN', ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True)


def smoke(kmc, tools, out):
    folder = out / 'smoke'
    folder.mkdir()
    seqs = ['A' * 300, 'T' * 50, 'ACGT' * 20 + 'NN' + 'GATTACA' * 8]
    fasta = folder / 'input.fa'
    fasta.write_text(''.join(f'>s{i}\n{s}\n' for i, s in enumerate(seqs)))
    expected = collections.Counter(s[i:i+21] for s in seqs for i in range(len(s)-20)
                                   if set(s[i:i+21]) <= set('ACGT'))
    run([kmc, '-k21', '-b', '-ci1', '-cs255', '-fm', '-t2', '-m2',
         fasta, folder / 'db', folder])
    run([tools, 'transform', folder / 'db', 'dump', folder / 'dump.tsv'])
    actual = dict((k, int(v)) for k, v in
                  (line.split() for line in (folder / 'dump.tsv').read_text().splitlines()))
    if actual != {k: min(v, 255) for k, v in expected.items()}:
        raise ValueError('KMC differs from naive strand-specific counts')
    (folder / 'PASSED').write_text('Naive counts, reverse complements, N boundaries, saturation checked.\n')
    print('SMOKE PASSED', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--kmc-bin-dir', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--memory-gb', type=int, default=64)
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    kmc, tools = args.kmc_bin_dir / 'kmc', args.kmc_bin_dir / 'kmc_tools'
    smoke(kmc, tools, out)
    if args.smoke_only:
        return
    manifest = args.index_dir / 'split_manifest.tsv'
    with manifest.open() as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    by_id = {r['internal_id']: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError('Duplicate internal IDs')
    for key in ('group_id', 'assembly_accession'):
        splits = collections.defaultdict(set)
        for row in rows:
            splits[row[key]].add(row['split'])
        if any(len(s) != 1 for s in splits.values()):
            raise ValueError('Split leakage in ' + key)
    train = [r for r in rows if r['split'] == 'train']
    if len({r['species_taxid'] for r in train}) != 961:
        raise ValueError('Expected 961 training species')
    if any(r['split'] not in ('train', 'val', 'test') for r in rows):
        raise ValueError('Unexpected split')
    with (out / 'train_manifest.tsv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(train)
    complement = bytes.maketrans(b'ACGTRYKMSWBDHVN', b'TGCAYRMKSWVHDBN')
    bases, count, seen = 0, 0, set()
    partial = out / 'train_reference.fna.partial'
    with partial.open('wb') as dest:
        for name, seq in records(args.index_dir / 'bacteria_complete.genomic.fna'):
            if name not in by_id or name in seen:
                raise ValueError('Unknown or duplicate FASTA ID: ' + name)
            seen.add(name)
            row = by_id[name]
            if row['split'] != 'train':
                continue
            digest = min(hashlib.sha256(seq).hexdigest(),
                         hashlib.sha256(seq.translate(complement)[::-1]).hexdigest())
            if len(seq) != int(row['length']) or digest != row['sequence_sha256']:
                raise ValueError('Sequence differs from frozen manifest: ' + name)
            dest.write(('>' + name + '\n').encode())
            for start in range(0, len(seq), 80):
                dest.write(seq[start:start+80] + b'\n')
            count += 1
            bases += len(seq)
            if count % 100 == 0:
                print(f'EXTRACT {count}/{len(train)} records; {bases} bp', flush=True)
    if seen != set(by_id) or count != len(train):
        raise ValueError('Missing FASTA records')
    fasta = out / 'train_reference.fna'
    partial.rename(fasta)
    report = dict(records=count, bases=bases, species=961, k=21, canonical=False,
                  min_count=1, counter_max=255, reference_split='train',
                  manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    (out / 'reference_report.json').write_text(json.dumps(report, indent=2) + '\n')
    tmp = out / 'kmc_tmp'
    tmp.mkdir()
    run([kmc, '-k21', '-b', '-ci1', '-cs255', '-fm', '-sm',
         f'-t{args.threads}', f'-m{args.memory_gb}', f'-j{out / "kmc_summary.json"}',
         fasta, out / 'train21', tmp])
    if not all((out / ('train21' + suffix)).stat().st_size > 0
               for suffix in ('.kmc_pre', '.kmc_suf')):
        raise ValueError('Missing KMC database output')
    (out / 'COUNTS_COMPLETE').write_text('Count database complete; window scores not generated yet.\n')
    print('COUNTS COMPLETE (window scoring is a separate stage)', flush=True)


if __name__ == '__main__':
    main()
