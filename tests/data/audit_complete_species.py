#!/usr/bin/env python3
"""Audit >=3-assembly species from complete_candidates.tsv; CPU, no source mutations."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, '/home/user/zhangguoli/species_identification/tests/data')

from stat_complete_genomes import distribution


def load_taxonomy(directory):
    nodes, merged = {}, {}
    with open(Path(directory) / 'nodes.dmp', encoding='utf-8') as f:
        for line in f:
            fields = [x.strip() for x in line.split('|')]
            nodes[int(fields[0])] = (int(fields[1]), fields[2])
    path = Path(directory) / 'merged.dmp'
    if path.exists():
        with open(path, encoding='utf-8') as f:
            for line in f:
                fields = line.split('|')
                merged[int(fields[0])] = int(fields[1])
    return nodes, merged


def resolve(taxid, merged):
    seen = set()
    while taxid in merged:
        if taxid in seen:
            raise ValueError('TaxID merge cycle')
        seen.add(taxid)
        taxid = merged[taxid]
    return taxid


def check_taxonomy(taxid, species, nodes, merged):
    taxid, species = resolve(taxid, merged), resolve(species, merged)
    if species not in nodes or nodes[species][1] != 'species':
        return species, 'species_taxid_missing_or_not_species_rank'
    seen, found, bacteria = set(), None, False
    while taxid not in seen:
        seen.add(taxid)
        if taxid not in nodes:
            return species, 'taxid_or_ancestor_missing'
        parent, rank = nodes[taxid]
        if rank == 'species' and found is None:
            found = taxid
        bacteria = bacteria or taxid == 2
        if parent == taxid:
            break
        taxid = parent
    else:
        return species, 'taxonomy_cycle'
    if not bacteria:
        return species, 'not_bacteria'
    return species, 'ok' if found == species else 'species_ancestry_mismatch'


def fasta_fingerprint(path, window, stride):
    """Order/header/linewrap-independent record multiset; exact reverse complements canonicalized.

    Circular rotations and near-duplicates are NOT detected.
    """
    records, accessions, seq = [], [], bytearray()
    bases = windows = ambiguous = 0
    table = bytes.maketrans(b'ACGTRYKMSWBDHVN', b'TGCAYRMKSWVHDBN')
    def finish():
        nonlocal bases, windows, ambiguous
        if not seq:
            raise ValueError('Empty FASTA record')
        data = bytes(seq).upper()
        if set(data) - set(b'ACGTRYKMSWBDHVN'):
            raise ValueError('Invalid DNA characters')
        digest = min(hashlib.sha256(data).hexdigest(),
                     hashlib.sha256(data.translate(table)[::-1]).hexdigest())
        records.append((len(data), digest))
        bases += len(data)
        windows += max(0, (len(data) - window) // stride + 1)
        ambiguous += sum(data.count(bytes([b])) for b in b'RYKMSWBDHVN')
        seq.clear()
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rb') as f:
        for line in f:
            if line.startswith(b'>'):
                if accessions:
                    finish()
                header = line[1:].split()
                if not header:
                    raise ValueError('Empty FASTA header')
                accessions.append(header[0].decode('ascii'))
            elif line.strip():
                if not accessions:
                    raise ValueError('Sequence before header')
                seq.extend(b''.join(line.split()))
    if not accessions:
        raise ValueError('Empty FASTA')
    finish()
    if len(set(accessions)) != len(accessions):
        raise ValueError('Duplicate sequence accession inside assembly')
    signature = hashlib.sha256(json.dumps(sorted(records)).encode()).hexdigest()
    return dict(sequence_signature=signature, sequences=len(records), bases=bases,
                estimated_windows=windows, ambiguous_bases=ambiguous), records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', default='/home/user/zhangguoli/species_identification/tests/data/complete_stats_full/complete_candidates.tsv')
    parser.add_argument('--taxdump-dir', default='/home/user/zhangguoli/species_identification/utils/taxonomy_data/new_taxdump')
    parser.add_argument('--output-dir', default='/home/user/zhangguoli/species_identification/tests/data/complete_977_audit')
    parser.add_argument('--min-assemblies', type=int, default=3)
    parser.add_argument('--window-size', type=int, default=4096)
    parser.add_argument('--stride', type=int, default=2048)
    args = parser.parse_args()
    if min(args.min_assemblies, args.window_size, args.stride) <= 0:
        parser.error('Counts and window parameters must be positive')
    with open(args.candidates, encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    sizes = Counter(r['species_taxid'] for r in rows)
    selected = [r for r in rows if sizes[r['species_taxid']] >= args.min_assemblies]
    if len({r['assembly_accession'] for r in rows}) != len(rows):
        raise ValueError('Duplicate assembly rows in candidates')
    nodes, merged = load_taxonomy(args.taxdump_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    print(f'Auditing {len(selected):,} assemblies / {len({r["species_taxid"] for r in selected}):,} original species', flush=True)
    audited, signatures, record_groups = [], defaultdict(list), defaultdict(list)
    for i, original in enumerate(selected, 1):
        row = dict(original)
        row['issues'] = []
        species, status = check_taxonomy(int(row['taxid']), int(row['species_taxid']), nodes, merged)
        row['resolved_species_taxid'] = species
        if status != 'ok':
            row['issues'].append(status)
        if row['assembly_level'] != 'Complete Genome' or row['file_status'] != 'found':
            row['issues'].append('not_found_complete')
        if row.get('version_status') != 'latest':
            row['issues'].append('not_latest')
        if row.get('excluded_from_refseq', '').strip().lower() not in ('', 'na'):
            row['issues'].append('excluded_from_refseq')
        try:
            old_counts = {k: row.get(k) for k in ('bases', 'sequences', 'estimated_windows')}
            stats, records = fasta_fingerprint(row['fna_path'], args.window_size, args.stride)
            row.update(stats)
            if any(v not in ('', None) and int(v) != stats[k] for k, v in old_counts.items()):
                row['issues'].append('counts_changed_since_inventory')
            if not stats['estimated_windows']:
                row['issues'].append('no_full_windows')
            signatures[stats['sequence_signature']].append(row)
            for length, digest in set(records):
                if length >= args.window_size:
                    record_groups[(length, digest)].append((row['assembly_accession'], species))
        except (OSError, EOFError, ValueError) as exc:
            row['issues'].append('fasta_error: ' + str(exc))
        audited.append(row)
        if i % 100 == 0:
            print(f'Checked {i:,}/{len(selected):,}', flush=True)
    duplicates, unique = [], defaultdict(list)
    for signature, members in signatures.items():
        conflict = len({r['resolved_species_taxid'] for r in members}) > 1
        if len(members) > 1:
            duplicates.append(dict(signature=signature, cross_species=conflict,
                                   assemblies=[r['assembly_accession'] for r in members]))
        if conflict:
            for r in members:
                r['issues'].append('identical_assembly_cross_species')
        clean = sorted((r for r in members if not r['issues']), key=lambda r: r['assembly_accession'])
        if clean:
            unique[clean[0]['resolved_species_taxid']].append(clean[0])
    eligible = {s: rs for s, rs in unique.items() if len(rs) >= args.min_assemblies}
    fields = list(rows[0]) + ['resolved_species_taxid', 'sequence_signature', 'ambiguous_bases', 'issues'] if rows else []
    for filename, data in [('assembly_audit.tsv', audited),
                           ('eligible_unique_assemblies.tsv', [r for rs in eligible.values() for r in rs])]:
        with open(out / filename, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fields, delimiter='\t', extrasaction='ignore')
            writer.writeheader()
            for r in data:
                writer.writerow({**r, 'issues': '; '.join(r['issues'])})
    with open(out / 'shared_full_sequences.tsv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['length', 'sha256', 'cross_species', 'assemblies'])
        for (length, digest), members in record_groups.items():
            if len(members) > 1:
                writer.writerow([length, digest, len({s for _, s in members}) > 1,
                                 ','.join(a for a, _ in members)])
    report = dict(status='complete', original_species=len({r['species_taxid'] for r in selected}),
                  assemblies_checked=len(audited), issues=dict(Counter(x for r in audited for x in r['issues'])),
                  exact_duplicate_assembly_groups=len(duplicates),
                  cross_species_duplicate_groups=sum(d['cross_species'] for d in duplicates),
                  eligible_species=len(eligible), eligible_assemblies=sum(map(len, eligible.values())),
                  eligible_window_distribution=distribution({s: sum(r['estimated_windows'] for r in rs) for s, rs in eligible.items()}),
                  clean_unique_assembly_distribution=distribution({s: len(rs) for s, rs in unique.items()}),
                  shared_full_sequence_groups=sum(len(v) > 1 for v in record_groups.values()),
                  notes=['Exact duplicate checks ignore headers, record order, case and reverse complement.',
                         'Circular rotations, near-duplicates, shared subsequences and ANI NOT checked.',
                         'Shared full sequences (including plasmids) flagged, not automatically removed.',
                         'Eligible list is provisional, not a train/val/test split.',
                         'Only species with >=minimum assemblies in the original inventory were audited.'])
    for filename, data in [('report.json', report), ('duplicate_assemblies.json', duplicates)]:
        with open(out / filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
