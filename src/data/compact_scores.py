"""Verified compact scores keyed by frozen (sequence ID, window start)."""
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class CompactScoreReader:
    def __init__(self, directory, index_path, window_size_bp=4096, kmer_size=21):
        root = Path(directory)
        if not (root / 'SCORES_COMPLETE').is_file():
            raise ValueError('Compact scores are incomplete: SCORES_COMPLETE missing')
        report = json.loads((root / 'scores_manifest.json').read_text())
        self.length = window_size_bp - kmer_size + 1
        if (report['schema'] != 'compact_train_windows_v1'
                or report['dtype'] != 'float32' or report['byteorder'] != sys.byteorder
                or report['window_bp'] != window_size_bp or report['k'] != kmer_size
                or report['values_per_window'] != self.length):
            raise ValueError('Compact score format/window parameters mismatch')
        score_path = root / 'train_window_scores.f32'
        offset_path = root / 'train_row_offsets.npy'
        for path, key in ((index_path, 'source_index_sha256'),
                          (score_path, 'score_sha256'), (offset_path, 'offsets_sha256')):
            if file_sha256(path) != report[key]:
                raise ValueError('Compact score checksum mismatch: ' + str(path))
        offsets = np.load(offset_path, allow_pickle=False)
        if offsets.dtype != np.int64 or offsets.shape != (report['train_rows'],):
            raise ValueError('Invalid compact row offsets')
        expected_values = report['unique_windows'] * self.length
        if score_path.stat().st_size != expected_values * 4:
            raise ValueError('Compact score binary size mismatch')
        if np.any(offsets < 0) or np.any(offsets % self.length) or np.any(offsets + self.length > expected_values):
            raise ValueError('Compact score offsets out of bounds or unaligned')
        with open(index_path, 'rb') as stream:
            windows = pickle.load(stream)['train']
        if len(windows) != len(offsets):
            raise ValueError('Compact scores and training index differ in length')
        self.offsets = {}
        for (info, start, old_offset, taxid), offset in zip(windows, offsets):
            if old_offset != -1:
                raise ValueError('Compact scores require the original DNA-only index')
            key = (info.accession, start)
            offset = int(offset)
            if key in self.offsets and self.offsets[key] != offset:
                raise ValueError('Repeated window has inconsistent score offsets')
            self.offsets[key] = offset
        if (len(self.offsets) != report['unique_windows']
                or len(set(self.offsets.values())) != report['unique_windows']):
            raise ValueError('Distinct windows share or lack score offsets')
        self._mmap = np.memmap(score_path, dtype=np.float32, mode='r')

    def read_window_scores(self, accession, start, length):
        if length != self.length:
            raise ValueError('Compact score token length mismatch')
        key = (accession, start)
        if key not in self.offsets:
            raise ValueError('No training scores for this window; use DNA-only validation/test')
        offset = self.offsets[key]
        scores = self._mmap[offset:offset + length].copy()
        if not np.isfinite(scores).all() or np.any((scores < 0) & (scores != -1)) or np.any(scores > 1):
            raise ValueError('Invalid compact score values')
        return np.clip(scores, 0.0, 1.0)
