"""Synthetic end-to-end final data tests; PyTorch Dataset is separately tested on server."""
import ast
import hashlib
import json
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_clean_complete_manifest import production_scope


class FakeConfig:
    @classmethod
    def from_yaml(cls, path):
        obj = cls()
        if Path(path).exists():
            values = json.loads(Path(path).read_text())
            for key in ('data', 'training', 'model'):
                setattr(obj, key, SimpleNamespace(**values[key]))
            return obj
        obj.data = SimpleNamespace()
        obj.training = SimpleNamespace(batch_size_per_gpu=4, gradient_accumulation_steps=8, warmup_steps=3000)
        obj.model = SimpleNamespace(kmer_size=21)
        return obj

    def to_yaml(self, path):
        Path(path).write_text(json.dumps({k: vars(getattr(self, k)) for k in ('data', 'training', 'model')}), encoding='utf-8')


def builder_scope(clean):
    path = Path(__file__).with_name('build_trainable_windows.py')
    root = path.resolve().parents[2]
    sys.path.insert(0, str(root))
    tree = ast.parse(path.read_text(encoding='utf-8'))
    tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom)
                 and n.module in ('src.config', 'clean_complete_manifest'))]
    scope = dict(__name__='test_builder', __file__=str(path), Config=FakeConfig,
                 validate_clean=clean['validate'], **{k: clean[k] for k in (
                     'read_tsv', 'write_tsv', 'records', 'sequence_hash', 'sha256')})
    exec(compile(tree, str(path), 'exec'), scope)
    return scope


class BuildWindowTests(unittest.TestCase):
    def test_extra_pool_only_uses_clean_train_and_preserves_holdouts(self):
        builder = builder_scope(production_scope())
        def row(asm, split, selected, species='10', reason=''):
            return dict(assembly_accession=asm, sequence_accession=asm, split=split,
                        species_taxid=species, reason=reason, windows='5',
                        sequence_sha256=asm, selected_window_ordinals=json.dumps(selected))
        original = [row('train', 'train', [0, 2]), row('val', 'val', [1]),
                    row('test', 'test', [3])]
        pool = [row('extra', 'train', []), row('forbidden_val', 'val', []),
                row('excluded', 'train', [], reason='bad'), row('other', 'train', [], species='20')]
        result = builder['expand_training_rows'](original, 8, 42, pool)
        self.assertEqual({r['assembly_accession'] for r in result}, {'train', 'extra', 'val', 'test'})
        self.assertEqual(builder['verify_expansion'](original, result)['added_train_windows'], 6)
        self.assertEqual(result, builder['expand_training_rows'](original, 8, 42, pool))
        with self.assertRaisesRegex(ValueError, 'available=10'):
            builder['expand_training_rows'](original, 11, 42, pool)
        repeated = builder['expand_training_rows'](original, 13, 42, pool, True)
        train = [r for r in repeated if r['split'] == 'train']
        self.assertEqual(sum(len(json.loads(r['selected_window_ordinals'])) for r in train), 13)
        self.assertEqual(sum(len(set(json.loads(r['selected_window_ordinals']))) for r in train), 10)
        self.assertEqual(builder['verify_expansion'](original, repeated)['added_train_windows'], 11)

    def test_real_fasta_index_labels_and_reader(self):
        clean = production_scope()
        builder = builder_scope(clean)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = []
            for i, motif in enumerate(('A', 'C', 'AC')):
                fna = root / f'input{i}.fna'
                fna.write_text('>original\n' + (motif * 8193)[:8193] + '\n', encoding='ascii')
                stats, _ = clean['fasta_fingerprint'](fna, 4096, 2048)
                candidates.append(dict(assembly_accession=f'asm{i}', resolved_species_taxid=10,
                                       fna_path=str(fna), sequence_signature=stats['sequence_signature'],
                                       issues='', assembly_level='Complete Genome', version_status='latest'))
            eligible = root / 'eligible.tsv'
            clean['write_tsv'](eligible, candidates, list(candidates[0]))
            clean_dir = root / 'clean'
            with patch('sys.argv', ['clean', '--eligible', str(eligible), '--output-dir', str(clean_dir)]):
                clean['main']()
            taxdump = root / 'taxonomy'
            taxdump.mkdir()
            (taxdump / 'rankedlineage.dmp').write_text(
                '10 | Test species | Test species | Testgenus | Testfamily | Testorder | Testclass | Testphylum | | Bacteria |\n',
                encoding='utf-8')
            output = root / 'final'
            builder['build'](clean_dir, output, taxdump, root / 'config')
            report = builder['validate_dataset'](output, verify_sources=True, dataset_smoke=False)
            self.assertEqual(report['windows'], {'train': 3, 'val': 3, 'test': 3})
            self.assertEqual(report['source_checked_records'], 3)
            self.assertEqual(report['species'], 1)
            small_output = root / 'small'
            builder['build'](clean_dir, small_output, taxdump, root / 'config',
                             species_limit=1, quotas=dict(train=2, val=1, test=2), seed=42)
            small_report = builder['validate_dataset'](small_output, True, False)
            self.assertEqual(small_report['windows'], {'train': 2, 'val': 1, 'test': 2})
            expanded_output = root / 'expanded'
            real_validate = builder['validate_dataset']
            with patch.dict(builder, validate_dataset=lambda path: real_validate(path, True, False)):
                builder['build'](clean_dir, expanded_output, taxdump, root / 'config',
                                 quotas=dict(train=3), expand_from=small_output)
            expanded_report = real_validate(expanded_output, True, False)
            self.assertEqual(expanded_report['windows'], {'train': 3, 'val': 1, 'test': 2})
            self.assertEqual(expanded_report['expansion_checks'], dict(
                original_train_windows_retained=2, added_train_windows=1,
                validation_windows_unchanged=True, test_windows_unchanged=True))
            repeated_output = root / 'repeated'
            with patch.dict(builder, validate_dataset=lambda path: real_validate(path, True, False)):
                builder['build'](clean_dir, repeated_output, taxdump, root / 'config',
                    quotas=dict(train=5), expand_from=small_output, oversample_shortfall=True)
            repeated_report = real_validate(repeated_output, True, False)
            self.assertEqual(repeated_report['windows'], dict(train=5, val=1, test=2))
            self.assertEqual(repeated_report['train_unique_windows'], 3)
            self.assertEqual(repeated_report['train_repeated_entries'], 2)
            # Existing outputs may mix prior-manifest internal IDs with empty IDs
            # on additional clean records; derived IDs are checked independently.
            mixed = builder['read_tsv'](repeated_output / 'cleaning_records.tsv')
            for i, r in enumerate(mixed):
                r['internal_id'] = '' if i == 0 else r['assembly_accession'] + '__' + r['sequence_accession']
            builder['write_tsv'](repeated_output / 'cleaning_records.tsv', mixed, list(mixed[0]))
            meta_path = repeated_output / 'dataset_metadata.json'
            mixed_meta = json.loads(meta_path.read_text())
            mixed_meta['file_sha256']['cleaning_records.tsv'] = builder['sha256'](repeated_output / 'cleaning_records.tsv')
            meta_path.write_text(json.dumps(mixed_meta))
            self.assertEqual(real_validate(repeated_output, True, False)['windows']['train'], 5)
            original_rows = builder['read_tsv'](small_output / 'split_manifest.tsv')
            with self.assertRaises(ValueError):
                builder['expand_training_rows'](original_rows, 4, 42)
            changed = [dict(r) for r in original_rows]
            for r in changed:
                if r['split'] == 'val':
                    r['selected_window_ordinals'] = '[]'
            with self.assertRaises(ValueError):
                builder['verify_expansion'](original_rows, changed)
            # Reject an incorrect index even when its storage hash has been refreshed.
            path = output / 'window_index_finetune.pkl'
            with path.open('rb') as f:
                index = pickle.load(f)
            info, _, score, taxid = index['train'][0]
            index['train'][0] = (info, 2048, score, taxid)
            with path.open('wb') as f:
                pickle.dump(index, f)
            metadata = json.loads((output / 'dataset_metadata.json').read_text())
            metadata['file_sha256'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            (output / 'dataset_metadata.json').write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'Invalid window'):
                builder['validate_dataset'](output, False, False)

    def test_exact_balancing_and_no_replacement(self):
        builder = builder_scope(production_scope())
        rows = [dict(species_taxid=str(s), split=split, assembly_accession=f'{s}_{split}',
                     sequence_accession='seq', windows='10')
                for s in range(4) for split in ('train', 'val', 'test')]
        selected, species = builder['select_small_rows'](rows, 2, dict(train=5, val=3, test=3), 42)
        self.assertEqual(len(species), 2)
        self.assertEqual((selected, species), builder['select_small_rows'](rows, 2, dict(train=5, val=3, test=3), 42))
        for row in selected:
            positions = json.loads(row['selected_window_ordinals'])
            self.assertEqual(len(positions), 5 if row['split'] == 'train' else 3)
            self.assertEqual(len(positions), len(set(positions)))
        with self.assertRaisesRegex(ValueError, 'Only 0 species'):
            builder['select_small_rows'](rows, 2, dict(train=11, val=3, test=3), 42)

    def test_requested_32_species_811_budget(self):
        builder = builder_scope(production_scope())
        rows = [dict(species_taxid=str(s), split=split, assembly_accession=f'{s}_{split}',
                     sequence_accession='seq', windows='1000')
                for s in range(40) for split in ('train', 'val', 'test')]
        chosen, species = builder['select_small_rows'](rows, 32, dict(train=200, val=25, test=25), 42)
        self.assertEqual(len(species), 32)
        totals = {split: sum(len(json.loads(r['selected_window_ordinals'])) for r in chosen if r['split'] == split)
                  for split in ('train', 'val', 'test')}
        self.assertEqual(totals, {'train': 6400, 'val': 800, 'test': 800})


if __name__ == '__main__':
    unittest.main()
