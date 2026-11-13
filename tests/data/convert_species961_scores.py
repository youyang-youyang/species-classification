#!/usr/bin/env python3
"""Write compact train-window scores; explicit offsets are NOT legacy sequence offsets."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import pickle
import subprocess
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.fna_reader import FNAReader


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index-dir', type=Path, required=True)
    p.add_argument('--counts-dir', type=Path, required=True)
    p.add_argument('--query-exe', type=Path, required=True)
    args = p.parse_args()
    root = args.counts_dir
    if not (root/'COUNTS_COMPLETE').exists():
        raise ValueError('Count database is not complete')
    out = root/'window_scores_v1'
    out.mkdir(exist_ok=False)
    # Independent naive reference, including novel k-mers, N, reverse complement and saturation.
    reference = ['A'*300, 'T'*50, 'ACGT'*20+'NN'+'GATTACA'*8]
    query = reference + ['C'*35, 'A'*21+'N'+'T'*25]
    counts = collections.Counter(s[i:i+21] for s in reference for i in range(len(s)-20)
                                 if set(s[i:i+21]) <= set('ACGT'))
    expected=[]
    for s in query:
        for i in range(len(s)-20):
            k=s[i:i+21]
            c=min(255, counts[k])
            expected.append(-1 if set(k)-set('ACGT') else
                            (float(f'{math.exp(-5*(c-1)/254):.4f}') if c else 0))
    subprocess.run([str(args.query_exe),str(root/'smoke/db'),str(out/'smoke.bin'),
                    str(out/'smoke.json')],input=('\n'.join(query)+'\n').encode(),check=True)
    if not np.array_equal(np.fromfile(out/'smoke.bin',dtype=np.float32),
                          np.array(expected,dtype=np.float32)):
        raise ValueError('Independent scoring smoke test failed')
    print('SCORING SMOKE PASSED',flush=True)
    index_path=args.index_dir/'window_index_finetune.pkl'
    source_hash=sha(index_path)
    with index_path.open('rb') as stream: index=pickle.load(stream)
    train=index['train']
    exposure=collections.Counter(int(row[3]) for row in train)
    if len(train)!=961000 or len(exposure)!=961 or set(exposure.values())!={1000}:
        raise ValueError('Unexpected train quotas')
    if sha(args.index_dir/'split_manifest.tsv') != json.loads((root/'reference_report.json').read_text())['manifest_sha256']:
        raise ValueError('Count reference manifest differs')
    reader=FNAReader(str(args.index_dir/'bacteria_complete.genomic.fna'))
    unique={}; offsets=[]; sample=[]
    partial=out/'train_window_scores.f32.partial'
    proc=subprocess.Popen([str(args.query_exe),str(root/'train21'),str(partial),
                           str(out/'query_stats.json')],stdin=subprocess.PIPE)
    try:
        for rownum,(info,start,old_offset,taxid) in enumerate(train):
            if old_offset!=-1: raise ValueError('Expected frozen DNA-only index')
            key=(info.accession,start)
            if key not in unique:
                seq=reader.read_bases(info,start,4096)
                if len(seq)!=4096: raise ValueError('Short window')
                offset=len(unique)*4076
                unique[key]=offset
                proc.stdin.write((seq+'\n').encode('ascii'))
                if len(sample)<32:
                    sample.append(dict(accession=info.accession,start=start,offset=offset,
                                       sequence_sha256=hashlib.sha256(seq.encode()).hexdigest()))
            offsets.append(unique[key])
            if rownum%10000==0: print(f'INPUT {rownum}/{len(train)}',flush=True)
        proc.stdin.close()
        if proc.wait()!=0: raise RuntimeError('KMC query process failed')
    except BaseException:
        proc.kill(); proc.wait(); raise
    finally:
        reader.close()
    stats=json.loads((out/'query_stats.json').read_text())
    if stats['windows']!=len(unique) or stats['values']!=len(unique)*4076 or stats['absent']!=0:
        raise ValueError('Missing training k-mers or inconsistent output size')
    if partial.stat().st_size!=len(unique)*4076*4:
        raise ValueError('Truncated binary output')
    if sha(index_path)!=source_hash: raise ValueError('Source index changed')
    # One offset per original training row, preserving order and sampling multiplicity.
    np.save(out/'train_row_offsets.npy',np.array(offsets,dtype=np.int64))
    with (out/'unique_windows.tsv').open('w') as stream:
        stream.write('accession\twindow_start\tfloat32_offset\n')
        for (uid,start),offset in unique.items(): stream.write(f'{uid}\t{start}\t{offset}\n')
    final=out/'train_window_scores.f32'
    partial.rename(final)
    report=dict(schema='compact_train_windows_v1',dtype='float32',byteorder=sys.byteorder,
                window_bp=4096,k=21,values_per_window=4076,train_rows=len(train),
                unique_windows=len(unique),source_index_sha256=source_hash,
                score_sha256=sha(final),offsets_sha256=sha(out/'train_row_offsets.npy'),
                query_stats=stats,samples=sample,
                note='Offsets are absolute float32 positions. Do NOT add window_start. Validation/test retain DNA-only input.')
    (out/'scores_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'SCORES_COMPLETE').write_text('Generation and positional checks passed. Training loader adaptation required.\n')
    print('SCORES COMPLETE',json.dumps(stats),flush=True)


if __name__=='__main__': main()
