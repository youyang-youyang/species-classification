#!/usr/bin/env python3
"""Fit train-target GTDB topology embeddings, with explicit mapping provenance.

Missing taxa receive zero vectors, never hash or random identity substitutes.
This exports TRAIN keys only: held-out evaluation must remain DNA-only.
"""
import argparse
import collections
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
from Bio import Phylo
from gensim.models.poincare import PoincareModel


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def assembly(name):
    return name.removeprefix('RS_').removeprefix('GB_')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--tree', type=Path, required=True)
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    out = args.output_dir
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    with args.manifest.open() as stream: rows=list(csv.DictReader(stream,delimiter='\t'))
    train = [r for r in rows if r['split']=='train']
    if len({r['species_taxid'] for r in train})!=961: raise ValueError('Expected 961 train species')
    if len({r['internal_id'] for r in rows})!=len(rows): raise ValueError('Duplicate internal IDs')
    sources=collections.defaultdict(set)
    for r in rows: sources[r['group_id']].add(r['split'])
    if any(len(v)!=1 for v in sources.values()): raise ValueError('Cross-split source groups')
    wanted={r['assembly_accession'] for r in train}
    bases={a.split('.')[0] for a in wanted}
    species={r['species_taxid'] for r in train}
    print('Loading GTDB tree',flush=True)
    tree=Phylo.read(args.tree,'newick')
    parents={}; order=[]; leaves=set(); branch_lengths={}
    stack=[(tree.root,None)]; number=0
    while stack:
        clade,parent=stack.pop()
        if clade.clades:
            name=f'_internal_{number}'; number+=1
        else:
            name=clade.name.strip("'\" \t\r\n")
            if name in leaves: raise ValueError('Duplicate GTDB leaf')
            leaves.add(name)
        parents[name]=parent; order.append(name); branch_lengths[name]=clade.branch_length
        for child in reversed(clade.clades): stack.append((child,name))
    exact=collections.defaultdict(set); version=collections.defaultdict(set)
    by_species=collections.defaultdict(set)
    print(f'Tree leaves={len(leaves)}; mapping train assemblies',flush=True)
    with args.metadata.open() as stream:
        for r in csv.DictReader(stream,delimiter='\t'):
            acc=r['accession']; representative=r['gtdb_genome_representative']
            target=acc if acc in leaves else representative
            if target not in leaves: continue
            for a in (assembly(acc),r['ncbi_genbank_assembly_accession']):
                if a in wanted: exact[a].add(target)
                if a.split('.')[0] in bases: version[a.split('.')[0]].add(target)
            if r['ncbi_species_taxid'] in species:
                by_species[r['ncbi_species_taxid']].add(target)
    mappings=[]; selected=set()
    for r in train:
        a=r['assembly_accession']; targets=exact.get(a,set()); method='assembly_exact'
        if not targets:
            targets=version.get(a.split('.')[0],set()); method='assembly_version_proxy'
        if not targets:
            targets=by_species.get(r['species_taxid'],set()); method='ncbi_species_proxy'
        if not targets: method='missing_zero'; target=''
        elif method!='ncbi_species_proxy' and len(targets)!=1:
            raise ValueError('Ambiguous assembly mapping: '+a)
        else:
            # Stable choice among SAME NCBI species representatives, explicitly reported as proxy.
            target=sorted(targets)[0]; selected.add(target)
        mappings.append(dict(internal_id=r['internal_id'],assembly_accession=a,
                             species_taxid=r['species_taxid'],method=method,target_node=target,
                             candidate_nodes=len(targets)))
    mapping_path=out/'mapping.tsv'
    with mapping_path.open('w') as stream:
        w=csv.DictWriter(stream,fieldnames=list(mappings[0]),delimiter='\t'); w.writeheader(); w.writerows(mappings)
    coverage=collections.Counter(r['method'] for r in mappings)
    print('MAPPING',json.dumps(coverage),flush=True)
    if not selected: raise ValueError('No real mapped nodes')
    # Retain complete ancestor paths for train-target leaves, without extra leaves.
    retained=set()
    for leaf in selected:
        node=leaf
        while node is not None and node not in retained:
            retained.add(node); node=parents[node]
    relations=[(parents[node],node) for node in order
               if node in retained and parents[node] is not None]
    with (out/'subtree_edges.tsv').open('w') as stream:
        stream.write('parent\tchild\toriginal_branch_length\n')
        for parent,child in relations: stream.write(f'{parent}\t{child}\t{branch_lengths[child]}\n')
    print(f'TRAIN nodes={len(retained)} leaves={len(selected)} edges={len(relations)} epochs={args.epochs}',flush=True)
    # Same topology-only parent/child relation convention as original generator.
    # Branch lengths are audited above but are NOT used as edge weights.
    model=PoincareModel(relations,size=64,burn_in=2,seed=args.seed,workers=1)
    model.train(epochs=args.epochs,print_every=1000)
    vectors={key:torch.tensor(model.kv[key].tolist(),dtype=torch.float32) for key in selected}
    for key,value in vectors.items():
        if value.shape!=(64,) or not torch.isfinite(value).all() or not (0<float(torch.linalg.vector_norm(value))<1):
            raise ValueError('Invalid Poincare vector: '+key)
    zero=torch.zeros(64,dtype=torch.float32)
    lookup={r['internal_id']:vectors[r['target_node']] if r['target_node'] else zero for r in mappings}
    lookup['UNKNOWN']=zero
    vector_file=out/'phylo_species961_train.pt'
    torch.save(lookup,str(vector_file)+'.partial')
    check=torch.load(str(vector_file)+'.partial',map_location='cpu',weights_only=False)
    if set(check)!={r['internal_id'] for r in train}|{'UNKNOWN'}: raise ValueError('Key coverage mismatch')
    for key in lookup:
        if not torch.equal(lookup[key],check[key]): raise ValueError('Serialization mismatch')
    os.replace(str(vector_file)+'.partial',vector_file)
    species_status=collections.defaultdict(collections.Counter)
    for r in mappings: species_status[r['species_taxid']][r['method']]+=1
    missing_species=sorted(s for s,v in species_status.items() if set(v)=={'missing_zero'})
    report=dict(seed=args.seed,dimension=64,epochs=args.epochs,burn_in=2,workers=1,
                method='Poincare parent-child topology embedding on train-target ancestral subtree',
                branch_lengths_used=False,split='train',train_records=len(train),species=961,
                real_vector_species=961-len(missing_species),missing_species=missing_species,
                mapping_counts=dict(coverage),species_mapping={s:dict(v) for s,v in species_status.items()},
                subtree_nodes=len(retained),subtree_leaves=len(selected),subtree_edges=len(relations),
                inputs={str(path):digest(path) for path in (args.manifest,args.tree,args.metadata)},
                output_sha256=digest(vector_file),mapping_sha256=digest(mapping_path),
                numpy_version=np.__version__,torch_version=torch.__version__,
                warning='Missing taxa have zero vectors. No hash fallback. Train-only lookup; validation/test must use DNA only.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'PHYLO_COMPLETE').write_text('Vectors generated and reloaded; inspect report for missing/proxy mappings.\n')
    print('PHYLO_COMPLETE',json.dumps(dict(records=len(train),real_vector_species=961-len(missing_species),
                                         missing_species=missing_species,coverage=dict(coverage))),flush=True)


if __name__=='__main__': main()
