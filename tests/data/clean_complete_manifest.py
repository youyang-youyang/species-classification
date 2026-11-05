#!/usr/bin/env python3
"""Create and independently validate assembly-isolated cleaning manifests (no training)."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, '/home/user/zhangguoli/species_identification')
sys.path.insert(0, '/home/user/zhangguoli/species_identification/tests/data')
from src.data.index_builder import IndexBuilder
from audit_complete_species import fasta_fingerprint
from stat_complete_genomes import distribution


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def records(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    accession, sequence = None, bytearray()
    with opener(path, 'rb') as f:
        for line in f:
            if line.startswith(b'>'):
                if accession is not None:
                    yield accession, bytes(sequence).upper()
                accession = line[1:].split()[0].decode('ascii')
                sequence.clear()
            elif line.strip():
                if accession is None:
                    raise ValueError('Sequence before header')
                sequence.extend(b''.join(line.split()))
    if accession is not None:
        yield accession, bytes(sequence).upper()


def sequence_hash(data):
    table = bytes.maketrans(b'ACGTRYKMSWBDHVN', b'TGCAYRMKSWVHDBN')
    return min(hashlib.sha256(data).hexdigest(),
               hashlib.sha256(data.translate(table)[::-1]).hexdigest())


def write_tsv(path, rows, fields):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fields, delimiter='\t', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def read_tsv(path):
    with open(path, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def plan(rows, seed=42, val_ratio=.1, test_ratio=.1):
    """Rows are record-level; exclude cross-species hashes, union same-species hashes."""
    hash_species = defaultdict(set)
    for r in rows:
        if not r['reason']:
            hash_species[r['sequence_sha256']].add(r['species_taxid'])
    parent = {}
    def find(a):
        parent.setdefault(a, a)
        if parent[a] != a:
            parent[a] = find(parent[a])
        return parent[a]
    first, same_assembly_hash = {}, set()
    for r in rows:
        if r['reason']:
            continue
        a, h = r['assembly_accession'], r['sequence_sha256']
        if len(hash_species[h]) > 1:
            r['reason'] = 'cross_species_identical_record'
            continue
        if (a, h) in same_assembly_hash:
            r['reason'] = 'duplicate_record_within_assembly'
            continue
        same_assembly_hash.add((a, h))
        root = find(a)
        if h in first:
            other = find(first[h])
            parent[max(root, other)] = min(root, other)
        first[h] = a
    species_groups = defaultdict(set)
    for r in rows:
        if not r['reason']:
            r['group_id'] = find(r['assembly_accession'])
            species_groups[r['species_taxid']].add(r['group_id'])
    for r in rows:
        if not r['reason'] and len(species_groups[r['species_taxid']]) < 3:
            r['reason'] = 'fewer_than_3_independent_groups'
    kept = [r for r in rows if not r['reason']]
    if not kept:
        raise ValueError('No species with three independent groups remains')
    assembly_species = {r['assembly_accession']: int(r['species_taxid']) for r in kept}
    assembly_groups = {r['assembly_accession']: r['group_id'] for r in kept}
    assignment = IndexBuilder.species_group_assignment(
        assembly_species, assembly_groups, val_ratio, test_ratio, seed)
    for r in kept:
        r['split'] = assignment[r['assembly_accession']]
    return rows


def validate(directory, verify_sources=False):
    directory = Path(directory)
    metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    rows = read_tsv(directory / 'records.tsv')
    if sha256(directory / 'records.tsv') != metadata['records_sha256']:
        raise ValueError('Manifest checksum mismatch')
    seen, labels = set(), {}
    species_splits, assemblies, groups, hashes = defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
    support = {s: Counter() for s in ('train', 'val', 'test')}
    group_species, hash_species = defaultdict(set), defaultdict(set)
    for r in rows:
        key = (r['assembly_accession'], r['sequence_accession'])
        if key in seen:
            raise ValueError('Duplicate record ID')
        seen.add(key)
        expected = max(0, (int(r['length']) - metadata['window_size']) // metadata['stride'] + 1)
        if int(r['windows']) != expected:
            raise ValueError('Window count mismatch')
        if r['reason']:
            if r['split']:
                raise ValueError('Excluded record assigned a split')
            continue
        split, species = r['split'], int(r['species_taxid'])
        if split not in support or expected <= 0 or not r['group_id']:
            raise ValueError('Invalid retained record')
        species_splits[species].add(split)
        assemblies[r['assembly_accession']].add(split)
        labels.setdefault(r['assembly_accession'], species)
        if labels[r['assembly_accession']] != species:
            raise ValueError('Assembly has conflicting species labels')
        groups[r['group_id']].add(split)
        group_species[r['group_id']].add(species)
        hashes[r['sequence_sha256']].add(split)
        hash_species[r['sequence_sha256']].add(species)
        support[split][species] += expected
    if not species_splits or any(v != {'train', 'val', 'test'} for v in species_splits.values()):
        raise ValueError('Species coverage differs between splits')
    for mapping in (assemblies, groups, hashes, group_species, hash_species):
        if any(len(v) != 1 for v in mapping.values()):
            raise ValueError('Cross-split leakage or conflicting species for group/hash')
    if verify_sources:
        for path, digest in metadata['source_sha256'].items():
            if sha256(path) != digest:
                raise ValueError('Source FASTA changed: ' + path)
    return dict(status='passed', species=len(species_splits), assemblies=len(assemblies),
                independent_groups=len(groups), retained_records=sum(not r['reason'] for r in rows),
                exclusions=dict(Counter(r['reason'] for r in rows if r['reason'])),
                split_window_distributions={s: distribution(c) for s, c in support.items()},
                source_checksums_verified=verify_sources,
                limitations=['No ANI, near-duplicate, circular-rotation or partial-window similarity checks.',
                             'No training index or scores generated; do not pass these manifests directly to finetune.py.'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eligible', default='/home/user/zhangguoli/species_identification/tests/data/complete_977_audit/eligible_unique_assemblies.tsv', help='eligible_unique_assemblies.tsv from completed audit')
    p.add_argument('--output-dir', default='/home/user/zhangguoli/species_identification/tests/data/complete_clean_v1')
    p.add_argument('--validate', metavar='DIRECTORY', nargs='?', const='/home/user/zhangguoli/species_identification/tests/data/complete_clean_v1')
    p.add_argument('--verify-sources', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--window-size', type=int, default=4096)
    p.add_argument('--stride', type=int, default=2048)
    args = p.parse_args()
    if args.validate:
        result = validate(args.validate, args.verify_sources)
        print(json.dumps(result, indent=2))
        return
    if not args.eligible or not args.output_dir or min(args.window_size, args.stride) <= 0:
        p.error('Need --eligible and new --output-dir; positive window parameters')
    eligible = read_tsv(args.eligible)
    if not eligible or len({r['assembly_accession'] for r in eligible}) != len(eligible):
        raise ValueError('Empty or duplicate assembly list')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    rows, sources = [], {}
    for i, a in enumerate(sorted(eligible, key=lambda r: r['assembly_accession']), 1):
        if a['issues'] or a['assembly_level'] != 'Complete Genome' or a['version_status'] != 'latest':
            raise ValueError('Ineligible assembly in input')
        path = str(Path(a['fna_path']).resolve())
        sources[path] = sha256(path)
        stats, _ = fasta_fingerprint(path, args.window_size, args.stride)
        if stats['sequence_signature'] != a['sequence_signature']:
            raise ValueError('FASTA changed since audit: ' + path)
        for accession, data in records(path):
            n = max(0, (len(data) - args.window_size) // args.stride + 1)
            rows.append(dict(assembly_accession=a['assembly_accession'], sequence_accession=accession,
                             species_taxid=int(a['resolved_species_taxid']), fna_path=path,
                             length=len(data), sequence_sha256=sequence_hash(data), windows=n,
                             ambiguous_bases=len(data) - sum(data.count(bytes([b])) for b in b'ACGT'),
                             reason='' if n else 'shorter_than_window', group_id='', split=''))
        if i % 100 == 0:
            print(f'[Clean] {i}/{len(eligible)} assemblies', flush=True)
    plan(rows, args.seed)
    write_tsv(out / 'records.tsv', rows, list(rows[0]))
    metadata = dict(version=1, seed=args.seed, window_size=args.window_size, stride=args.stride,
                    split_ratios={'train': .8, 'val': .1, 'test': .1},
                    source_sha256=sources, eligible_sha256=sha256(args.eligible),
                    records_sha256=sha256(out / 'records.tsv'),
                    split_rule='Existing IndexBuilder.species_group_assignment; ratios by groups, not windows',
                    label_rule='Resolved NCBI species TaxID; no organism-name guessing')
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    result = validate(out)
    (out / 'validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
