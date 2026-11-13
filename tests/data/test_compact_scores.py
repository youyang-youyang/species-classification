import json
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.compact_scores import CompactScoreReader, file_sha256
from src.data.dataset import FinetuneDataset


class FakeDNA:
    def read_bases(self, info, start, length): return 'A' * length


class FakeTaxonomy:
    def encode_lineage(self, taxid): return {'species': taxid}


class CompactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        info = SimpleNamespace(accession='record')
        self.rows = [(info, 2048, -1, 0), (info, 4096, -1, 0), (info, 2048, -1, 0)]
        self.index = self.root/'index.pkl'
        self.index.write_bytes(pickle.dumps({'train': self.rows}))
        values = np.full(4076*2, .25, dtype=np.float32)
        values[4076:] = .75
        values[0] = -1
        values.tofile(self.root/'train_window_scores.f32')
        np.save(self.root/'train_row_offsets.npy', np.array([0,4076,0], dtype=np.int64))
        self.report = dict(schema='compact_train_windows_v1',dtype='float32',byteorder=sys.byteorder,
                           window_bp=4096,k=21,values_per_window=4076,train_rows=3,unique_windows=2)
        self.refresh()
        (self.root/'SCORES_COMPLETE').touch()

    def refresh(self):
        for filename,key in [('index.pkl','source_index_sha256'),('train_window_scores.f32','score_sha256'),
                             ('train_row_offsets.npy','offsets_sha256')]:
            self.report[key] = file_sha256(self.root/filename)
        (self.root/'scores_manifest.json').write_text(json.dumps(self.report))

    def tearDown(self): self.temp.cleanup()
    def reader(self): return CompactScoreReader(self.root,self.index)

    def test_positions_repeats_and_dna_only(self):
        ds=FinetuneDataset(self.rows,FakeDNA(),self.reader(),FakeTaxonomy())
        a,b,c=[ds[i]['uniqueness_scores'] for i in range(3)]
        self.assertTrue(torch.equal(a,c))
        self.assertEqual(a[0].item(),0)
        self.assertTrue(torch.all(a[1:]==.25).item())
        self.assertTrue(torch.all(b==.75).item())
        dna=FinetuneDataset(self.rows,FakeDNA(),None,FakeTaxonomy())
        self.assertIsNone(dna[0]['uniqueness_scores'])

    def test_reject_unknown_window(self):
        with self.assertRaises(ValueError): self.reader().read_window_scores('record',0,4076)

    def test_checksum_failure(self):
        with (self.root/'train_window_scores.f32').open('ab') as stream: stream.write(b'x')
        with self.assertRaisesRegex(ValueError,'checksum'): self.reader()

    def test_wrong_index(self):
        self.index.write_bytes(pickle.dumps({'train':list(reversed(self.rows[:2]))}))
        with self.assertRaisesRegex(ValueError,'checksum'): self.reader()

    def test_inconsistent_repeated_offset(self):
        np.save(self.root/'train_row_offsets.npy',np.array([0,4076,4076],dtype=np.int64))
        self.refresh()
        with self.assertRaisesRegex(ValueError,'Repeated'): self.reader()

    def test_missing_complete_marker(self):
        (self.root/'SCORES_COMPLETE').unlink()
        with self.assertRaisesRegex(ValueError,'incomplete'): self.reader()


if __name__=='__main__': unittest.main()
