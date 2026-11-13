"""CPU-only real-data check; does not train or modify data."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.config import Config
from src.data.compact_scores import CompactScoreReader
from src.data.dataset import FinetuneDataset, collate_finetune
from src.data.fna_reader import FNAReader
from src.data.taxonomy import TaxonomyDB


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    args=p.parse_args()
    cfg=Config.from_yaml(args.config)
    index_path=Path(cfg.data.index_dir)/'window_index_finetune.pkl'
    reader=CompactScoreReader(cfg.data.compact_scores_dir,index_path,
                              cfg.data.window_size_bp,cfg.model.kmer_size)
    with index_path.open('rb') as stream: index=pickle.load(stream)
    fna=FNAReader(cfg.data.fna_file)
    taxonomy=TaxonomyDB.load(str(Path(cfg.data.index_dir)/'taxonomy_db.pkl'))
    results={}
    for split in ('train','val','test'):
        ds=FinetuneDataset(index[split],fna,reader if split=='train' else None,taxonomy,
                           cfg.data.window_size_bp,cfg.model.kmer_size)
        rows=[0,len(ds)//3,len(ds)//2,len(ds)-1]
        batch=collate_finetune([ds[i] for i in rows])
        if split=='train':
            assert tuple(batch['uniqueness_scores'].shape)==(4,4076)
            assert torch.isfinite(batch['uniqueness_scores']).all().item()
        else:
            assert batch['uniqueness_scores'] is None and 'phylo_vecs' not in batch
        results[split]=dict(rows=len(ds),bases_shape=list(batch['bases_21'].shape),
                           scores='compact' if split=='train' else 'None')
    report=json.loads((Path(cfg.data.compact_scores_dir)/'scores_manifest.json').read_text())
    infos={info.accession:info for info,_,_,_ in index['train']}
    for sample in report['samples']:
        sequence=fna.read_bases(infos[sample['accession']],sample['start'],4096)
        assert hashlib.sha256(sequence.encode()).hexdigest()==sample['sequence_sha256']
        assert reader.offsets[(sample['accession'],sample['start'])]==sample['offset']
    print('COMPACT_DATA_CHECK_PASSED',json.dumps(results),flush=True)


if __name__=='__main__': main()
