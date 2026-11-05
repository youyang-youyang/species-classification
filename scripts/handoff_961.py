"""Package, verify and smoke-test this experiment without changing training code."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = 'index_bacteria_species961_train1000'
RUN = 'bacteria_species961_train1000_seed42_v1_finetune'
CONFIG = 'finetune_species961_train1000_seed42.yaml'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def check_archive(path):
    """Verify a downloaded tar on Mac/Linux with Python stdlib only; no extraction."""
    seen = {}
    manifest = None
    with tarfile.open(path, 'r|*') as tar:
        for member in tar:
            if not member.isfile() or member.name in seen:
                raise ValueError(f'Unexpected/duplicate archive entry: {member.name}')
            stream = tar.extractfile(member)
            if member.name == 'handoff_961/manifest.json':
                if manifest is not None:
                    raise ValueError('Duplicate manifest')
                manifest = json.load(stream)
                continue
            h = hashlib.sha256()
            size = 0
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                size += len(block)
                h.update(block)
            seen[member.name] = dict(bytes=size, sha256=h.hexdigest())
            print('Read', member.name, flush=True)
    if manifest is None or seen != manifest['files']:
        raise ValueError('Archive incomplete or checksum mismatch')
    print('ARCHIVE_CHECK_PASSED', path)


def stable_copy(source, dest):
    before = source.stat()
    shutil.copy2(source, dest)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f'File changed while copying: {source}; retry after saving ends')


def load_checkpoint(path):
    import torch
    ck = torch.load(path, map_location='cpu', weights_only=False)
    required = {'epoch', 'global_step', 'model_state', 'optimizer_state',
                'scheduler_state', 'species_exposure', 'best_metric'}
    if required - set(ck):
        raise ValueError(f'Incomplete resume checkpoint: {required-set(ck)}')
    if ck['global_step'] != (ck['epoch'] + 1) * 30032:
        raise ValueError('Unexpected epoch/update budget; inspect before migration')
    return ck


def package(output):
    import torch
    import yaml
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = ROOT / 'configs' / CONFIG
    cfg = yaml.safe_load(config.read_text())
    if not (cfg['data']['dna_only'] and cfg['training']['disable_phylo']
            and cfg['training']['guidance_dropout'] == 1.0):
        raise ValueError('This handoff supports the DNA-only experiment only')
    if Path(cfg['data']['index_dir']).resolve() != ROOT / DATA:
        raise ValueError('Unexpected index directory')
    if Path(cfg['data']['fna_file']).resolve() != ROOT / DATA / 'bacteria_complete.genomic.fna':
        raise ValueError('Unexpected FASTA dependency')
    stage = output / 'stage'
    handoff = stage / 'handoff_961'
    handoff.mkdir(parents=True)
    candidates = sorted((ROOT / 'runs' / RUN).glob('checkpoint_epoch[0-9][0-9][0-9].pt'))
    if not candidates:
        raise FileNotFoundError('No epoch checkpoint')
    source = candidates[-1]
    stable_copy(source, handoff / 'resume.pt')
    ck = load_checkpoint(handoff / 'resume.pt')
    epoch, updates, best_metric = int(ck['epoch']), int(ck['global_step']), float(ck['best_metric'])
    if epoch >= 9:
        raise ValueError('Training already completed 10 epochs; no remaining work')
    del ck
    # Use an immutable epoch checkpoint matching the best score at the resume boundary.
    best_source = None
    for p in reversed(candidates):
        c = load_checkpoint(p)
        score = c.get('metrics', {}).get('val_species_acc')
        matches = int(c['epoch']) <= epoch and score is not None and abs(float(score)-best_metric) < 1e-10
        del c
        if matches:
            best_source = p
            break
    if best_source is None:
        raise ValueError('Cannot locate historical best checkpoint at resume boundary')
    stable_copy(best_source, handoff / 'best_model.pt')
    stable_copy(config, handoff / 'source_config.yaml')
    files = {f'{DATA}/{p.name}': p for p in (ROOT / DATA).iterdir() if p.is_file()}
    required = ['bacteria_complete.genomic.fna', 'taxonomy_db.pkl', 'window_index_finetune.pkl']
    for name in required:
        if f'{DATA}/{name}' not in files:
            raise FileNotFoundError(name)
    from src.data.taxonomy import TaxonomyDB
    from src.data.fna_reader import FNAReader
    db = TaxonomyDB.load(ROOT / DATA / 'taxonomy_db.pkl')
    if db.num_classes['species'] != 961:
        raise ValueError('Expected 961 labels')
    with (ROOT / DATA / 'window_index_finetune.pkl').open('rb') as f:
        index = pickle.load(f)
    reader = FNAReader(str(ROOT / DATA / 'bacteria_complete.genomic.fna'))
    examples = []
    for split, expected in [('train',961000), ('val',24025), ('test',24025)]:
        if len(index[split]) != expected:
            raise ValueError(f'Unexpected {split} count')
        for i in sorted({0, expected//2, expected-1}):
            info, start, offset, taxid = index[split][i]
            seq = reader.read_bases(info, start, 4096)
            if len(seq) != 4096 or offset != -1 or db.encode_lineage(taxid)['species'] < 0:
                raise ValueError('Invalid example window')
            examples.append(dict(split=split, row=i, sha256=hashlib.sha256(seq.encode()).hexdigest()))
    reader.close()
    del index, db
    # Verify the uncommitted source code does not diverge from the transferred commit.
    subprocess.run(['git','diff','--exit-code','HEAD','--','src','scripts/finetune.py','configs/'+CONFIG],
                   cwd=ROOT, check=True)
    commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    runtime = dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda)
    print(f'Packaging epoch {epoch}, next epoch {epoch+1}, updates {updates}', flush=True)
    files.update({f'handoff_961/{p.name}': p for p in handoff.iterdir() if p.is_file()})
    for rel, path in files.items():
        if path.is_symlink():
            raise ValueError(f'Symlink not supported: {path}')
    needed = sum(p.stat().st_size for p in files.values())
    if shutil.disk_usage(output).free < needed * 1.05:
        raise RuntimeError('Insufficient disk space for uncompressed payload')
    manifest = dict(source_commit=commit, epoch=epoch, next_epoch=epoch+1,
                    global_step=updates, best_metric=best_metric,
                    source_checkpoint=str(source), best_checkpoint=str(best_source),
                    runtime=runtime, examples=examples, files={})
    archive = output / 'species961_payload.tar'
    # Hash exactly the bytes fed to tar, avoiding a second full read of the 47G FASTA.
    with tarfile.open(archive, 'w') as tar:
        for rel, p in sorted(files.items()):
            before = p.stat()
            h = hashlib.sha256()
            class HashReader:
                def __init__(self, stream): self.stream = stream
                def read(self, n=-1):
                    b = self.stream.read(n)
                    h.update(b)
                    return b
            with p.open('rb') as stream:
                tar.addfile(tar.gettarinfo(str(p), arcname=rel), HashReader(stream))
            after = p.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                raise RuntimeError(f'Changed during packaging: {p}')
            manifest['files'][rel] = dict(bytes=before.st_size,sha256=h.hexdigest())
            print(f'Packed {rel}: {before.st_size:,} bytes', flush=True)
        meta = handoff / 'manifest.json'
        meta.write_text(json.dumps(manifest, indent=2))
        tar.add(meta, arcname='handoff_961/manifest.json')
    shutil.copy2(meta, output / 'manifest.json')
    (output / 'PACKAGING_COMPLETE').write_text(str(archive)+'\n')
    print('PACKAGING_COMPLETE', archive, flush=True)


def verify():
    import yaml
    from src.data.fna_reader import FNAReader
    manifest = json.loads((ROOT/'handoff_961/manifest.json').read_text())
    for rel, info in manifest['files'].items():
        p = ROOT / rel
        if p.stat().st_size != info['bytes'] or digest(p) != info['sha256']:
            raise ValueError(f'Checksum mismatch: {rel}')
        print('Verified',rel,flush=True)
    with (ROOT/DATA/'window_index_finetune.pkl').open('rb') as f:
        index = pickle.load(f)
    reader = FNAReader(str(ROOT/DATA/'bacteria_complete.genomic.fna'))
    for example in manifest['examples']:
        info,start,_,_ = index[example['split']][example['row']]
        seq = reader.read_bases(info,start,4096)
        if hashlib.sha256(seq.encode()).hexdigest() != example['sha256']:
            raise ValueError('Window read differs from source server')
    reader.close()
    ck = load_checkpoint(ROOT/'handoff_961/resume.pt')
    if int(ck['epoch']) != manifest['epoch']:
        raise ValueError('Checkpoint epoch mismatch')
    cfg = yaml.safe_load((ROOT/'handoff_961/source_config.yaml').read_text())
    cfg['data'].update(index_dir=str(ROOT/DATA),fna_file=str(ROOT/DATA/'bacteria_complete.genomic.fna'))
    cfg['output_dir'] = str(ROOT/'runs_h100')
    target = ROOT/'configs/finetune_species961_h100.yaml'
    target.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False))
    run = ROOT/'runs_h100'/RUN
    run.mkdir(parents=True,exist_ok=False)
    shutil.copy2(ROOT/'handoff_961/best_model.pt',run/'best_model.pt')
    (ROOT/'handoff_961/VERIFIED').write_text(digest(ROOT/'handoff_961/manifest.json'))
    print(f'VERIFIED: resume at Epoch {manifest["next_epoch"]}/9; config {target}')


def smoke():
    import torch
    from functools import partial
    from torch.utils.data import DataLoader
    from src.config import Config
    from src.data.taxonomy import TaxonomyDB
    from src.data.fna_reader import FNAReader
    from src.data.dataset import FinetuneDataset, collate_finetune
    from src.model.model import build_model
    from src.model.losses import SpeciesClassificationLoss
    from src.training.trainer import Trainer
    from scripts.finetune import finetune_forward_fn
    if not (ROOT/'handoff_961/VERIFIED').is_file():
        raise RuntimeError('Run verify first')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A CUDA GPU supporting BF16 is required')
    cfg = Config.from_yaml(ROOT/'configs/finetune_species961_h100.yaml')
    print('GPU:',torch.cuda.get_device_name(0),'PyTorch:',torch.__version__,flush=True)
    db = TaxonomyDB.load(ROOT/DATA/'taxonomy_db.pkl')
    with (ROOT/DATA/'window_index_finetune.pkl').open('rb') as f: index=pickle.load(f)
    reader=FNAReader(cfg.data.fna_file)
    def loader(split,n):
        ds=FinetuneDataset(index[split][:n],reader,None,db,
                           window_size_bp=4096,kmer_size=cfg.model.kmer_size)
        return DataLoader(ds,batch_size=4,num_workers=0,collate_fn=collate_finetune)
    train,val=loader('train',32),loader('val',8)
    del index
    model=build_model(config=cfg.model,mode='finetune',num_classes_per_level=db.num_classes)
    old,new=[],[]
    for name,p in model.named_parameters():
        (old if any(prefix in name for prefix in ['encoder.','embedding.base_embedding.','pos_encoding.']) else new).append(p)
    opt=torch.optim.AdamW([{'params':old,'lr':cfg.training.finetune_lr},
                          {'params':new,'lr':cfg.training.finetune_lr*10}],
                         weight_decay=cfg.training.finetune_weight_decay)
    loss=SpeciesClassificationLoss(classification_only=cfg.training.classification_loss_only,
        lambda_hier=cfg.training.lambda_hier,lambda_contrast=cfg.training.lambda_contrast,
        lambda_focus=cfg.training.lambda_focus,delta_focus=cfg.training.delta_focus,
        tau_contrast=cfg.training.tau_contrast)
    with tempfile.TemporaryDirectory(prefix='species961_smoke_') as tmp:
        trainer=Trainer(model,cfg.training,train,val,optimizer=opt,loss_fn=loss,
                        output_dir=tmp,device=torch.device('cuda:0'))
        trainer.build_cosine_scheduler(300320,cfg.training.warmup_steps)
        trainer.species_exposure={}
        trainer.load_checkpoint(str(ROOT/'handoff_961/resume.pt'))
        before=trainer.global_step
        fn=partial(finetune_forward_fn,guidance_dropout=1.0,disable_phylo=True)
        train_metrics=trainer.train_epoch(trainer.current_epoch+1,fn)
        val_metrics=trainer.validate(fn)
        if trainer.global_step != before+1 or not all(math.isfinite(v) for v in [*train_metrics.values(),*val_metrics.values()]):
            raise RuntimeError('Nonfinite metrics or failed optimizer update')
        # Only a disposable diagnostic checkpoint; never used for real resume.
        trainer.save_checkpoint(trainer.current_epoch,{'smoke_only':True})
        saved=Path(trainer.output_dir)/f'checkpoint_epoch{trainer.current_epoch:03d}.pt'
        trainer.load_checkpoint(str(saved))
    reader.close()
    (ROOT/'handoff_961/GPU_SMOKE_PASSED').write_text(json.dumps(dict(
        gpu=torch.cuda.get_device_name(0),torch=torch.__version__,optimizer_updates=1,
        train=train_metrics,val=val_metrics),indent=2))
    print('GPU_SMOKE_PASSED: resume, 8 microbatches, optimizer update, validation, save/reload')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['package','verify','smoke','check-archive'])
    p.add_argument('--output',type=Path)
    p.add_argument('--archive',type=Path)
    a=p.parse_args()
    if a.action=='package':
        if a.output is None: p.error('package requires --output (a new directory)')
        package(a.output)
    elif a.action=='check-archive':
        if a.archive is None: p.error('check-archive requires --archive')
        check_archive(a.archive)
    elif a.action=='verify': verify()
    else: smoke()
