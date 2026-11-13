"""Check delivered phylogeny vectors and actual Dataset batches, without training."""
import argparse
import collections
import csv
import json
from pathlib import Path
import pickle
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.data.compact_scores import CompactScoreReader, file_sha256
from src.data.dataset import FinetuneDataset, collate_finetune
from src.data.fna_reader import FNAReader
from src.data.taxonomy import TaxonomyDB


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index-dir',type=Path,required=True)
    p.add_argument('--phylo-dir',type=Path,required=True)
    p.add_argument('--scores-dir',type=Path,required=True)
    args=p.parse_args()
    root=args.phylo_dir
    assert (root/'PHYLO_COMPLETE').is_file(), 'Generation incomplete'
    report=json.loads((root/'report.json').read_text())
    assert file_sha256(root/'phylo_species961_train.pt')==report['output_sha256']
    assert file_sha256(root/'mapping.tsv')==report['mapping_sha256']
    manifest_hash=file_sha256(args.index_dir/'split_manifest.tsv')
    assert any(str(k).endswith('split_manifest.tsv') and v==manifest_hash for k,v in report['inputs'].items())
    db=torch.load(root/'phylo_species961_train.pt',map_location='cpu',weights_only=False)
    with (root/'mapping.tsv').open() as f: mapping=list(csv.DictReader(f,delimiter='\t'))
    with (args.index_dir/'window_index_finetune.pkl').open('rb') as f: index=pickle.load(f)
    ids={row[0].accession for row in index['train']}
    assert set(db)==ids|{'UNKNOWN'}
    assert torch.count_nonzero(db['UNKNOWN']).item()==0
    assert len(mapping)==len(ids) and {r['internal_id'] for r in mapping}==ids
    actual=collections.Counter()
    for r in mapping:
        vector=db[r['internal_id']]
        assert vector.dtype==torch.float32 and vector.shape==(64,) and torch.isfinite(vector).all().item()
        norm=torch.linalg.vector_norm(vector).item()
        assert norm==0 if r['method']=='missing_zero' else 0<norm<1
        actual[r['method']]+=1
    assert dict(actual)==report['mapping_counts']
    print('VECTOR_CONTENT_CHECK_PASSED',dict(actual),flush=True)
    scores=CompactScoreReader(args.scores_dir,args.index_dir/'window_index_finetune.pkl')
    fna=FNAReader(str(args.index_dir/'bacteria_complete.genomic.fna'))
    taxonomy=TaxonomyDB.load(str(args.index_dir/'taxonomy_db.pkl'))
    summary={}
    for split in ('train','val','test'):
        ds=FinetuneDataset(index[split],fna,scores if split=='train' else None,taxonomy,
                           phylo_dict=db if split=='train' else None)
        positions=[0,len(ds)//3,len(ds)//2,len(ds)-1]
        batch=collate_finetune([ds[i] for i in positions])
        if split=='train':
            assert batch['phylo_vecs'].shape==(4,64)
            assert batch['uniqueness_scores'].shape==(4,4076)
            for i,pos in enumerate(positions):
                assert torch.equal(batch['phylo_vecs'][i],db[index[split][pos][0].accession])
        else:
            assert 'phylo_vecs' not in batch and batch['uniqueness_scores'] is None
        summary[split]=dict(rows=len(ds),phylo_shape=list(batch['phylo_vecs'].shape) if 'phylo_vecs' in batch else None)
    (root/'data_check.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('PHYLO_DATA_CHECK_PASSED',json.dumps(summary),flush=True)


if __name__=='__main__': main()
