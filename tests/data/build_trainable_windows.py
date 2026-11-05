#!/usr/bin/env python3
"""Build real finetune windows and validate final files; CPU only, no training."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import pickle
import random
import shutil
import sys

PROJECT = '/home/user/zhangguoli/species_identification'
sys.path.insert(0, PROJECT)
sys.path.insert(0, PROJECT + '/tests/data')
from clean_complete_manifest import read_tsv, write_tsv, records, sequence_hash, sha256, validate as validate_clean
from src.config import Config
from src.data.fna_reader import SequenceInfo, FNAReader
from src.data.taxonomy import TaxonomyDB, TAXONOMY_LEVELS


def select_small_rows(rows, species_limit, quotas, seed):
    """Sample exact per-species quotas from already isolated splits, without replacement."""
    groups = defaultdict(list)
    for row in rows:
        groups[(int(row['species_taxid']), row['split'])].append(row)
    eligible = sorted(s for s in {s for s, _ in groups}
                      if all(sum(int(r['windows']) for r in groups[(s, split)]) >= quota
                             for split, quota in quotas.items()))
    if len(eligible) < species_limit:
        raise ValueError(f'Only {len(eligible)} species meet all quotas; requested {species_limit}')
    rng = random.Random(seed)
    species = sorted(rng.sample(eligible, species_limit))
    selected = []
    for s in species:
        for split, quota in quotas.items():
            candidates = sorted(groups[(s, split)], key=lambda r: (r['assembly_accession'], r['sequence_accession']))
            total = sum(int(r['windows']) for r in candidates)
            positions = sorted(rng.sample(range(total), quota))
            cursor = offset = 0
            for row in candidates:
                end = offset + int(row['windows'])
                local = []
                while cursor < len(positions) and positions[cursor] < end:
                    local.append(positions[cursor] - offset)
                    cursor += 1
                if local:
                    selected.append({**row, 'selected_window_ordinals': json.dumps(local)})
                offset = end
    return selected, species


def expand_training_rows(rows, train_quota, seed, candidate_rows=None, oversample_shortfall=False):
    """Keep all old windows; sample extra positions only from existing train records."""
    result = [dict(r) for r in rows]
    if candidate_rows is not None:
        species_ids = {int(r['species_taxid']) for r in rows}
        existing = {(r['assembly_accession'], r['sequence_accession']) for r in rows}
        for r in candidate_rows:
            key = (r['assembly_accession'], r['sequence_accession'])
            if (r['split'] == 'train' and not r['reason']
                    and int(r['species_taxid']) in species_ids and key not in existing):
                result.append({**r, 'selected_window_ordinals': '[]'})
                existing.add(key)
    rng = random.Random(seed)
    by_species = defaultdict(list)
    for r in result:
        if r['split'] == 'train':
            by_species[int(r['species_taxid'])].append(r)
    for species, group in sorted(by_species.items()):
        group.sort(key=lambda r: (r['assembly_accession'], r['sequence_accession']))
        selected = [set(json.loads(r['selected_window_ordinals'])) for r in group]
        old_count = sum(map(len, selected))
        capacities = [int(r['windows']) - len(s) for r, s in zip(group, selected)]
        need = train_quota - old_count
        available = old_count + sum(capacities)
        if need < 0 or (need > sum(capacities) and not oversample_shortfall) or available == 0:
            raise ValueError(f'Species TaxID {species}: selected={old_count}, '
                             f'available={old_count + sum(capacities)}, requested={train_quota}; '
                             'insufficient train windows or quota below original count; no duplication allowed')
        repeats = max(0, train_quota - available)
        chosen = sorted(rng.sample(range(sum(capacities)), min(need, sum(capacities))))
        cursor = offset = 0
        for r, old, capacity in zip(group, selected, capacities):
            local = set()
            while cursor < len(chosen) and chosen[cursor] < offset + capacity:
                local.add(chosen[cursor] - offset)
                cursor += 1
            additions = []
            available_rank = 0
            for ordinal in range(int(r['windows'])):
                if ordinal not in old:
                    if available_rank in local:
                        additions.append(ordinal)
                    available_rank += 1
            r['selected_window_ordinals'] = json.dumps(sorted(old | set(additions)))
            offset += capacity
        if repeats:
            pool = [(r, ordinal) for r in group
                    for ordinal in json.loads(r['selected_window_ordinals'])]
            additions = defaultdict(list)
            for _ in range(repeats):
                r, ordinal = pool[rng.randrange(len(pool))]
                additions[id(r)].append(ordinal)
            for r in group:
                r['selected_window_ordinals'] = json.dumps(sorted(
                    json.loads(r['selected_window_ordinals']) + additions[id(r)]))
    return [r for r in result if json.loads(r['selected_window_ordinals'])]


def verify_expansion(original, expanded):
    def keys(rows, split):
        return Counter((r['assembly_accession'], r['sequence_accession'], int(r['species_taxid']),
                 r['sequence_sha256'], i)
                for r in rows if r['split'] == split
                for i in json.loads(r['selected_window_ordinals']))
    old_train, new_train = keys(original, 'train'), keys(expanded, 'train')
    if old_train - new_train:
        raise ValueError('Expansion lost original train windows')
    for split in ('val', 'test'):
        if keys(original, split) != keys(expanded, split):
            raise ValueError('Expansion changed holdout windows: ' + split)
    return dict(original_train_windows_retained=sum(old_train.values()),
                added_train_windows=sum((new_train - old_train).values()),
                validation_windows_unchanged=True, test_windows_unchanged=True)


def extend_species_rows(original, clean_rows, total_species, quotas, seed, oversample_shortfall=False):
    old_species = {int(r['species_taxid']) for r in original}
    if total_species <= len(old_species):
        raise ValueError('Target species count must exceed baseline')
    counts = Counter()
    for r in original:
        counts[(int(r['species_taxid']), r['split'])] += len(json.loads(r['selected_window_ordinals']))
    if any(counts[(s, split)] != n for s in old_species for split, n in quotas.items()):
        raise ValueError('Baseline quotas differ from requested extension quotas')
    candidates = [r for r in clean_rows if not r['reason']
                  and int(r['species_taxid']) not in old_species]
    if oversample_shortfall:
        # Select species using a positive train pool and full, unique holdout quotas.
        initial_quotas = dict(quotas, train=1)
        added, new_species = select_small_rows(candidates, total_species - len(old_species), initial_quotas, seed)
        added = expand_training_rows(added, quotas['train'], seed,
                                     candidate_rows=candidates, oversample_shortfall=True)
    else:
        added, new_species = select_small_rows(candidates, total_species - len(old_species), quotas, seed)
    return [dict(r) for r in original] + added, sorted(old_species | set(new_species))


def build(clean_dir, output, taxdump, base_config, species_limit=0, quotas=None, seed=42,
          expand_from=None, include_clean_train=False, oversample_shortfall=False, extend_species=0):
    clean_dir, output = Path(clean_dir).resolve(), Path(output).resolve()
    validate_clean(clean_dir, verify_sources=True)
    clean_meta = json.loads((clean_dir / 'metadata.json').read_text(encoding='utf-8'))
    window, stride = clean_meta['window_size'], clean_meta['stride']
    rows = [r for r in read_tsv(clean_dir / 'records.tsv') if not r['reason']]
    small = None
    original_rows = None
    if expand_from:
        baseline = Path(expand_from).resolve()
        validate_dataset(baseline)
        baseline_meta = json.loads((baseline / 'dataset_metadata.json').read_text(encoding='utf-8'))
        if (not baseline_meta.get('small_dataset') or baseline_meta['window_size'] != window
                or baseline_meta['stride'] != stride
                or baseline_meta['original_clean_manifest_sha256'] != sha256(clean_dir / 'records.tsv')):
            raise ValueError('Expansion requires matching small-dataset cleaning source and window settings')
        original_rows = read_tsv(baseline / 'split_manifest.tsv')
        if extend_species:
            rows, _ = extend_species_rows(original_rows, rows, extend_species, quotas, seed,
                                          oversample_shortfall=oversample_shortfall)
            oversample_shortfall = (oversample_shortfall or
                                   baseline_meta['small_dataset'].get('oversample_shortfall', False))
        else:
            rows = expand_training_rows(original_rows, quotas['train'], seed,
                                        candidate_rows=rows if include_clean_train else None,
                                        oversample_shortfall=oversample_shortfall)
        selected_species = sorted({int(r['species_taxid']) for r in rows})
        species_limit = len(selected_species)
        quotas = dict(baseline_meta['small_dataset']['per_species_windows'], train=quotas['train'])
        small = dict(species_count=species_limit, species_taxids=selected_species,
                     per_species_windows=quotas, seed=seed, expanded_from=str(baseline),
                     original_metadata_sha256=sha256(baseline / 'dataset_metadata.json'),
                     oversample_shortfall=oversample_shortfall,
                     extended_species=bool(extend_species),
                     baseline_species_taxids=sorted({int(r['species_taxid']) for r in original_rows}),
                     training_pool=('all_clean_train_records' if include_clean_train
                                    else 'previously_selected_train_records'),
                     expansion_checks=verify_expansion(original_rows, [r for r in rows
                         if int(r['species_taxid']) in {int(o['species_taxid']) for o in original_rows}]))
    elif species_limit:
        rows, selected_species = select_small_rows(rows, species_limit, quotas, seed)
        small = dict(species_count=species_limit, species_taxids=selected_species,
                     per_species_windows=quotas, seed=seed,
                     note='Fixed window subset, no replacement; original source-group split preserved.')
    species = {int(r['species_taxid']) for r in rows}
    taxonomy = TaxonomyDB()
    taxonomy.load_taxdump(str(taxdump))
    names = {}
    for s in sorted(species):
        lineage = taxonomy.get_lineage(s)
        if not lineage or lineage['domain'] != 'Bacteria':
            raise ValueError(f'Missing bacterial taxonomy for species TaxID {s}')
        name = lineage['species']
        if name in names and names[name] != s:
            raise ValueError(f'Different species TaxIDs share classifier name: {s}, {names[name]}')
        names[name] = s
    taxonomy.build_label_encoders(species)
    if expand_from:
        old_taxonomy = TaxonomyDB.load(str(Path(expand_from) / 'taxonomy_db.pkl'))
        if not extend_species and taxonomy.idx2label != old_taxonomy.idx2label:
            raise ValueError('Expansion changed taxonomy label encoding')
        if extend_species:
            for s in {int(r['species_taxid']) for r in original_rows}:
                if taxonomy.get_lineage(s) != old_taxonomy.get_lineage(s):
                    raise ValueError('Baseline lineage changed during species extension')
    if taxonomy.num_classes['species'] != len(species):
        raise ValueError('Species label count mismatch')
    for s in species:
        if any(v < 0 for v in taxonomy.encode_lineage(s).values()):
            raise ValueError(f'Invalid labels for {s}')
    config = Config.from_yaml(str(base_config))
    if expand_from and config.model.kmer_size != baseline_meta['kmer_size']:
        raise ValueError('Expansion changed kmer size')
    if config.model.kmer_size > window:
        raise ValueError('Kmer larger than window')
    output.mkdir(parents=True, exist_ok=False)
    fasta = output / 'bacteria_complete.genomic.fna'
    by_source = defaultdict(list)
    for r in rows:
        by_source[r['fna_path']].append(r)
    index = {s: [] for s in ('train', 'val', 'test')}
    checks, manifest = [], []
    with open(fasta, 'wb') as f:
        for number, (source, selected) in enumerate(sorted(by_source.items()), 1):
            wanted = {r['sequence_accession']: r for r in selected}
            if len(wanted) != len(selected):
                raise ValueError('Source path reused across assemblies or duplicate accession')
            for accession, data in records(source):
                if accession not in wanted:
                    continue
                r = wanted.pop(accession)
                if len(data) != int(r['length']) or sequence_hash(data) != r['sequence_sha256']:
                    raise ValueError('Source sequence differs from cleaning manifest')
                uid = r['assembly_accession'] + '__' + accession
                header = f'{uid} original={accession} assembly={r["assembly_accession"]} species_taxid={r["species_taxid"]}'
                f.write(('>' + header + '\n').encode('ascii'))
                offset = f.tell()
                for pos in range(0, len(data), 80):
                    f.write(data[pos:pos + 80] + b'\n')
                info = SequenceInfo(uid, header, str(fasta), offset, 81, 80, len(data), False)
                n = max(0, (len(data) - window) // stride + 1)
                if n != int(r['windows']):
                    raise ValueError('Manifest window count mismatch')
                ordinals = (json.loads(r['selected_window_ordinals']) if small else range(n))
                for ordinal in ordinals:
                    start = ordinal * stride
                    index[r['split']].append((info, start, -1, int(r['species_taxid'])))
                # First/middle/last actual windows, plus a read starting across an 80bp boundary.
                starts = sorted({0, ((n - 1) // 2) * stride, (n - 1) * stride,
                                 min(79, len(data) - window)})
                import hashlib
                for start in starts:
                    checks.append(dict(uid=uid, start=start,
                                       sha256=hashlib.sha256(data[start:start + window]).hexdigest()))
                manifest.append({**r, 'internal_id': uid})
            if wanted:
                raise ValueError('Missing retained FASTA records: ' + source)
            if number % 100 == 0:
                print(f'[Build] {number}/{len(by_source)} assemblies', flush=True)
    taxonomy.save(str(output / 'taxonomy_db.pkl'))
    with open(output / 'window_index_finetune.pkl', 'wb') as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    del index
    write_tsv(output / 'split_manifest.tsv', manifest, list(manifest[0]))
    if small:
        write_tsv(output / 'cleaning_records.tsv', rows, list(rows[0]))
    else:
        shutil.copyfile(clean_dir / 'records.tsv', output / 'cleaning_records.tsv')
    (output / 'source_window_checks.json').write_text(json.dumps(checks), encoding='utf-8')
    config.data.fna_file = str(fasta)
    config.data.index_dir = str(output)
    config.data.taxdump_dir = str(Path(taxdump).resolve())
    config.data.scores_file = config.data.scores_bin_file = config.data.phylo_file = ''
    config.data.dna_only = True
    config.data.window_size_bp, config.data.window_stride_bp = window, stride
    config.training.guidance_dropout, config.training.disable_phylo = 1.0, True
    config.experiment_name = 'bacteria_complete_dna_v1'
    if small:
        config.experiment_name = f'bacteria_complete_small{species_limit}_seed{seed}'
        if expand_from:
            config.experiment_name += f'_train{quotas["train"]}'
        config.seed = seed
        config.training.steps_per_epoch = None
        config.training.val_steps = None
        config.training.species_balanced_fraction = 0.0
        total_train = species_limit * quotas['train']
        if total_train % config.training.batch_size_per_gpu:
            config.training.batch_size_per_gpu = 1
        # Small fixed index: visit every window once each epoch, full validation.
        batches = total_train // config.training.batch_size_per_gpu
        accum = config.training.gradient_accumulation_steps
        updates_per_epoch = (batches + accum - 1) // accum
        config.training.warmup_steps = min(config.training.warmup_steps, updates_per_epoch)
    config.to_yaml(str(output / 'config_data.yaml'))
    files = ['bacteria_complete.genomic.fna', 'taxonomy_db.pkl', 'window_index_finetune.pkl',
             'split_manifest.tsv', 'cleaning_records.tsv', 'source_window_checks.json', 'config_data.yaml']
    if original_rows is not None:
        write_tsv(output / 'expansion_baseline.tsv', original_rows, list(original_rows[0]))
        files.append('expansion_baseline.tsv')
    metadata = dict(window_size=window, stride=stride, kmer_size=config.model.kmer_size, small_dataset=small,
                    original_clean_manifest_sha256=sha256(clean_dir / 'records.tsv'),
                    clean_report=validate_clean(clean_dir),
                    source_sha256={s: clean_meta['source_sha256'][s] for s in by_source},
                    file_sha256={name: sha256(output / name) for name in files},
                    taxdump_sha256={name: sha256(Path(taxdump) / name)
                                   for name in ('nodes.dmp', 'merged.dmp', 'rankedlineage.dmp')
                                   if (Path(taxdump) / name).exists()},
                    absent_score_offset=-1, input_mode='DNA only; U=None, P=None',
                    unknown_lineage_labels={s: [k for k, v in taxonomy.get_lineage(s).items()
                                                if v.startswith('Unknown_')] for s in sorted(species)
                                            if any(v.startswith('Unknown_') for v in taxonomy.get_lineage(s).values())})
    (output / 'dataset_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return output


def validate_dataset(output, verify_sources=True, dataset_smoke=True):
    """Inspect every window; compare sampled random reads with original-sequence digests."""
    import hashlib
    output = Path(output).resolve()
    meta = json.loads((output / 'dataset_metadata.json').read_text(encoding='utf-8'))
    for name, digest in meta['file_sha256'].items():
        if sha256(output / name) != digest:
            raise ValueError('Final file changed: ' + name)
    config = Config.from_yaml(str(output / 'config_data.yaml'))
    if (not config.data.dna_only or config.training.guidance_dropout != 1.0
            or not config.training.disable_phylo
            or Path(config.data.index_dir).resolve() != output
            or Path(config.data.fna_file).resolve() != output / 'bacteria_complete.genomic.fna'
            or config.data.window_size_bp != meta['window_size']
            or config.data.window_stride_bp != meta['stride']
            or config.model.kmer_size != meta['kmer_size']):
        raise ValueError('Generated data configuration inconsistent with dataset')
    if verify_sources:
        for path, digest in meta['source_sha256'].items():
            if sha256(path) != digest:
                raise ValueError('Original source changed: ' + path)
    rows = read_tsv(output / 'split_manifest.tsv')
    expansion_checks = None
    if (meta.get('small_dataset') or {}).get('expanded_from'):
        baseline_rows = read_tsv(output / 'expansion_baseline.tsv')
        baseline_species = {int(r['species_taxid']) for r in baseline_rows}
        baseline_new_rows = [r for r in rows if int(r['species_taxid']) in baseline_species]
        expansion_checks = verify_expansion(baseline_rows, baseline_new_rows if
            meta['small_dataset'].get('extended_species') else rows)
        if meta['small_dataset'].get('extended_species'):
            if expansion_checks['added_train_windows'] != 0:
                raise ValueError('Species extension changed baseline train windows')
            expansion_checks['baseline_species'] = len(baseline_species)
            expansion_checks['added_species'] = len({int(r['species_taxid']) for r in rows} - baseline_species)
    clean = [r for r in read_tsv(output / 'cleaning_records.tsv') if not r['reason']]
    if len(rows) != len(clean):
        raise ValueError('Cleaning/final manifest size mismatch')
    expected_clean = {(r['assembly_accession'], r['sequence_accession']): r for r in clean}
    expected = {}
    expected_starts = {}
    for r in rows:
        old = expected_clean.get((r['assembly_accession'], r['sequence_accession']))
        # Expansion mixes prior manifest rows (with internal_id) and newly selected
        # cleaning rows (without it). It is derived, not source metadata: validate
        # its canonical value independently, including for already built outputs.
        if old is None or any(r[k] != old[k] for k in old if k != 'internal_id'):
            raise ValueError('Final manifest differs from cleaning plan')
        if r['internal_id'] != r['assembly_accession'] + '__' + r['sequence_accession']:
            raise ValueError('Invalid derived internal sequence ID')
        if r['internal_id'] in expected:
            raise ValueError('Duplicate internal sequence ID')
        expected[r['internal_id']] = r
        if meta.get('small_dataset'):
            ordinals = json.loads(r['selected_window_ordinals'])
            allow_repeats = (meta['small_dataset'].get('oversample_shortfall', False)
                             and r['split'] == 'train')
            if (not ordinals or ordinals != sorted(ordinals if allow_repeats else set(ordinals))
                    or any(type(i) is not int or i < 0 or i >= int(r['windows']) for i in ordinals)):
                raise ValueError('Invalid small-dataset selected window ordinals')
            expected_starts[r['internal_id']] = [i * meta['stride'] for i in ordinals]
    taxonomy = TaxonomyDB.load(str(output / 'taxonomy_db.pkl'))
    species_ids = {int(r['species_taxid']) for r in rows}
    encoded = {s: taxonomy.encode_lineage(s) for s in species_ids}
    if any(any(v < 0 for v in labels.values()) for labels in encoded.values()):
        raise ValueError('Invalid taxonomy label')
    if len({v['species'] for v in encoded.values()}) != len(species_ids):
        raise ValueError('Species labels collapsed')
    with open(output / 'window_index_finetune.pkl', 'rb') as f:
        index = pickle.load(f)
    if set(index) != {'train', 'val', 'test'}:
        raise ValueError('Invalid index splits')
    counts, infos = Counter(), {}
    supports = {s: Counter() for s in index}
    assemblies, groups, hashes, hash_species = (defaultdict(set) for _ in range(4))
    for split, windows in index.items():
        for info, start, score_offset, taxid in windows:
            r = expected.get(info.accession)
            if (r is None or r['split'] != split or int(r['species_taxid']) != taxid
                    or score_offset != -1
                    or (meta.get('small_dataset') and (counts[info.accession] >= len(expected_starts[info.accession])
                        or start != expected_starts[info.accession][counts[info.accession]]))
                    or (not meta.get('small_dataset') and start != counts[info.accession] * meta['stride'])
                    or start < 0 or start + meta['window_size'] > info.num_bases
                    or info.num_bases != int(r['length'])
                    or Path(info.fna_file).resolve() != output / 'bacteria_complete.genomic.fna'
                    or info.line_len_bytes != 81 or info.line_len_bases != 80):
                raise ValueError('Invalid window metadata, order, bounds or label')
            if info.accession in infos and info != infos[info.accession]:
                raise ValueError('Conflicting sequence offsets')
            infos[info.accession] = info
            counts[info.accession] += 1
            supports[split][taxid] += 1
        print(f'[Validate] {split}: {len(windows):,} windows', flush=True)
    for uid, r in expected.items():
        n = (int(r['length']) - meta['window_size']) // meta['stride'] + 1
        expected_n = len(expected_starts[uid]) if meta.get('small_dataset') else n
        if counts[uid] != expected_n or n != int(r['windows']) or n <= 0:
            raise ValueError('Window count differs from cleaned record')
        assemblies[r['assembly_accession']].add(r['split'])
        groups[r['group_id']].add(r['split'])
        hashes[r['sequence_sha256']].add(r['split'])
        hash_species[r['sequence_sha256']].add(r['species_taxid'])
    if any(set(c) != species_ids for c in supports.values()):
        raise ValueError('Species coverage differs across splits')
    if meta.get('small_dataset'):
        small = meta['small_dataset']
        if species_ids != set(small['species_taxids']) or len(species_ids) != small['species_count']:
            raise ValueError('Small dataset species count mismatch')
        for split, counts_by_species in supports.items():
            if any(n != small['per_species_windows'][split] for n in counts_by_species.values()):
                raise ValueError('Small dataset per-species window quota mismatch')
    if any(len(v) != 1 for mapping in (assemblies, groups, hashes, hash_species) for v in mapping.values()):
        raise ValueError('Cross-split or cross-species exact-record overlap')
    seen_fasta = set()
    for uid, data in records(output / 'bacteria_complete.genomic.fna'):
        if uid not in expected or uid in seen_fasta or sequence_hash(data) != expected[uid]['sequence_sha256']:
            raise ValueError('Final FASTA sequence mismatch')
        seen_fasta.add(uid)
    if seen_fasta != set(expected):
        raise ValueError('Missing final FASTA record')
    reader = FNAReader(str(output / 'bacteria_complete.genomic.fna'))
    checks = json.loads((output / 'source_window_checks.json').read_text(encoding='utf-8'))
    checked = set()
    for check in checks:
        data = reader.read_bases(infos[check['uid']], check['start'], meta['window_size'])
        if hashlib.sha256(data.encode('ascii')).hexdigest() != check['sha256']:
            raise ValueError('Random reader differs from original source slice')
        checked.add(check['uid'])
    if checked != set(expected):
        raise ValueError('Not all records covered by source slice validation')
    reader.close()
    if dataset_smoke:
        from src.data.dataset import FinetuneDataset, collate_finetune
        from torch.utils.data import DataLoader
        for split in index:
            sample = [index[split][0], index[split][-1]]
            reader = FNAReader(str(output / 'bacteria_complete.genomic.fna'))
            dataset = FinetuneDataset(sample, reader, None, taxonomy, meta['window_size'], meta['kmer_size'])
            batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0, collate_fn=collate_finetune)))
            if (batch['uniqueness_scores'] is not None or 'phylo_vecs' in batch
                    or tuple(batch['bases_21'].shape) != (2, meta['window_size'] - meta['kmer_size'] + 1, meta['kmer_size'])):
                raise ValueError('Actual Dataset/collate incompatible with DNA-only windows')
            for level in TAXONOMY_LEVELS:
                if batch['labels'][level].tolist() != [encoded[w[3]][level] for w in sample]:
                    raise ValueError('Dataset label mismatch')
            reader.close()
    report = dict(status='passed' if dataset_smoke else 'structural_checks_only',
                  species=len(species_ids), assemblies=len(assemblies), records=len(expected),
                  windows={s: len(v) for s, v in index.items()},
                  per_species_windows={s: dict(v) for s, v in supports.items()},
                  source_read_checks=len(checks), source_checked_records=len(checked),
                  source_checksums_verified=verify_sources, actual_dataset_collate_checked=dataset_smoke,
                  invalid_labels=0, window_errors=0, source_read_mismatches=0, cross_split_assemblies=0,
                  cross_split_groups=0, cross_split_full_sequence_hashes=0,
                  note='CPU data validation, not model training. Near duplicates/ANI/circular rotations/partial sequence sharing not audited.')
    if expansion_checks is not None:
        report['expansion_checks'] = expansion_checks
        duplicates = Counter()
        unique = Counter()
        for r in rows:
            if r['split'] == 'train':
                values = json.loads(r['selected_window_ordinals'])
                taxid = int(r['species_taxid'])
                duplicates[taxid] += len(values) - len(set(values))
                unique[taxid] += len(set(values))
        report['train_unique_windows'] = sum(unique.values())
        report['train_repeated_entries'] = sum(duplicates.values())
        report['oversampled_species'] = {str(s): dict(unique_windows=unique[s],
            repeated_entries=n) for s, n in duplicates.items() if n}
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clean-dir', default=PROJECT + '/tests/data/complete_clean_v1')
    p.add_argument('--output-dir', default=PROJECT + '/index_bacteria_complete_v1')
    p.add_argument('--taxdump-dir', default=PROJECT + '/utils/taxonomy_data/new_taxdump')
    p.add_argument('--base-config', default=PROJECT + '/index_bacteria_stratified_v2/config_resplit.yaml')
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--species-limit', type=int, default=0, help='Small fixed balanced dataset; 0 keeps full data')
    p.add_argument('--expand-from', help='Validated small dataset; retain species/holdouts and expand train quota only')
    p.add_argument('--extend-species', type=int, default=0,
                   help='With --expand-from: retain baseline windows and add eligible species to this total')
    p.add_argument('--oversample-shortfall', action='store_true',
                   help='Only if train pool is insufficient: use all unique windows then sample repeats')
    p.add_argument('--include-clean-train', action='store_true',
                   help='Expansion may use additional records from the original clean TRAIN split only')
    p.add_argument('--train-per-species', type=int, default=200)
    p.add_argument('--val-per-species', type=int, default=25)
    p.add_argument('--test-per-species', type=int, default=25)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.extend_species and (not args.expand_from or args.include_clean_train):
        p.error('--extend-species requires --expand-from; omit --include-clean-train')
    if args.oversample_shortfall and not args.expand_from:
        p.error('--oversample-shortfall requires --expand-from')
    if args.include_clean_train and not args.expand_from:
        p.error('--include-clean-train requires --expand-from')
    if args.expand_from and (args.species_limit or args.output_dir == PROJECT + '/index_bacteria_complete_v1'):
        p.error('--expand-from requires explicit new --output-dir and no --species-limit')
    if args.species_limit < 0 or min(args.train_per_species, args.val_per_species, args.test_per_species) <= 0:
        p.error('Species limit must be nonnegative; window quotas must be positive')
    if args.species_limit and args.output_dir == PROJECT + '/index_bacteria_complete_v1':
        args.output_dir = PROJECT + f'/index_bacteria_complete_small{args.species_limit}_seed{args.seed}'
    # Fail before large data generation if the actual Dataset dependencies are unavailable.
    from src.data.dataset import FinetuneDataset, collate_finetune
    output = Path(args.output_dir)
    report_path = output / 'data_report.json'
    if args.validate_only:
        report_path.write_text(json.dumps({'status': 'validating'}), encoding='utf-8')
    else:
        build(args.clean_dir, output, args.taxdump_dir, args.base_config, args.species_limit,
              dict(train=args.train_per_species, val=args.val_per_species, test=args.test_per_species), args.seed,
              expand_from=args.expand_from, include_clean_train=args.include_clean_train,
              oversample_shortfall=args.oversample_shortfall, extend_species=args.extend_species)
    try:
        report = validate_dataset(output)
    except Exception as exc:
        report_path.write_text(json.dumps({'status': 'failed', 'error': str(exc)}), encoding='utf-8')
        raise
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('[DONE] Trainable windows and data validation:', output, flush=True)
    print(json.dumps({k: report[k] for k in ('status', 'species', 'assemblies', 'windows')}))


if __name__ == '__main__':
    main()
