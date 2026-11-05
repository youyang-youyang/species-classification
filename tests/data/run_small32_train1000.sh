#!/usr/bin/env bash
# Usage: bash /home/user/zhangguoli/species_identification/tests/data/run_small32_train1000.sh /absolute/pretrained.pt
set -euo pipefail
cd /home/user/zhangguoli/species_identification
: "${CONDA_PREFIX:?Activate evo2_hf first}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
pretrained=${1:?Pass the same pretrained checkpoint FILE used by the 200/500-window experiments}
test -f "$pretrained" || { echo "Missing checkpoint file: $pretrained"; exit 1; }

# Report shortages; only training windows may be repeated to fill them.
python - <<'PY'
import csv, json
from pathlib import Path
from collections import Counter, defaultdict
root = Path('/home/user/zhangguoli/species_identification')
baseline = root / 'index_bacteria_small32_train500'
clean = root / 'tests/data/complete_clean_v1'
with (baseline / 'split_manifest.tsv').open() as f:
    species = {int(r['species_taxid']) for r in csv.DictReader(f, delimiter='\t')}
if len(species) != 32:
    raise SystemExit('Expected exactly 32 baseline species')
available = Counter()
assemblies = defaultdict(set)
with (clean / 'records.tsv').open() as f:
    for r in csv.DictReader(f, delimiter='\t'):
        taxid = int(r['species_taxid'])
        if taxid in species and r['split'] == 'train' and not r['reason']:
            available[taxid] += int(r['windows'])
            assemblies[taxid].add(r['assembly_accession'])
print('TaxID\ttrain_available_windows\ttrain_assemblies', flush=True)
for taxid in sorted(species):
    print(taxid, available[taxid], len(assemblies[taxid]), sep='\t', flush=True)
short = {s: available[s] for s in sorted(species) if available[s] < 1000}
if short:
    if any(n == 0 for n in short.values()):
        raise SystemExit('STOP: species with zero train windows: ' + json.dumps(short))
    print('Training-only oversampling required: ' + json.dumps(
        {s: dict(unique_windows=n, repeated_entries=1000-n) for s, n in short.items()}), flush=True)
PY

out=/home/user/zhangguoli/species_identification/index_bacteria_small32_train1000
if [ -e "$out" ]; then
  echo "Existing output: run full validation before continuing: $out"
  python -u tests/data/build_trainable_windows.py --validate-only --output-dir "$out"
else
  python -u tests/data/build_trainable_windows.py \
  --expand-from /home/user/zhangguoli/species_identification/index_bacteria_small32_train500 \
  --include-clean-train --oversample-shortfall \
  --base-config /home/user/zhangguoli/species_identification/configs/finetune_small32_train500_equal_budget.yaml \
  --train-per-species 1000 --seed 42 --output-dir "$out"
fi

python - <<'PY'
from pathlib import Path
import json
import yaml
root = Path('/home/user/zhangguoli/species_identification')
out = root / 'index_bacteria_small32_train1000'
report = json.loads((out / 'data_report.json').read_text())
assert report['status'] == 'passed'
assert report['windows'] == dict(train=32000, val=800, test=800)
checks = report['expansion_checks']
assert checks['original_train_windows_retained'] == 16000
assert checks['added_train_windows'] == 16000
assert checks['validation_windows_unchanged'] and checks['test_windows_unchanged']
cfg = yaml.safe_load((out / 'config_data.yaml').read_text())
cfg.update(experiment_name='bacteria_small32_train1000_equal_budget_v1', output_dir='./runs', seed=42)
cfg['training'].update(finetune_epochs=10, batch_size_per_gpu=4,
    gradient_accumulation_steps=8, warmup_steps=200, steps_per_epoch=None,
    val_steps=None, species_balanced_fraction=0.0, guidance_dropout=1.0,
    disable_phylo=True, classification_loss_only=False, track_species_exposure=True,
    early_stop_patience=0)
config = root / 'configs/finetune_small32_train1000_equal_budget.yaml'
config.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
run = root / 'runs/bacteria_small32_train1000_equal_budget_v1_finetune'
if run.exists():
    raise SystemExit('Training output already exists: ' + str(run))
print('Data passed; 10 epochs / 10000 updates / 200 warmup updates', flush=True)
PY

python -u scripts/finetune.py \
  --config configs/finetune_small32_train1000_equal_budget.yaml \
  --pretrained "$pretrained" --gpus 0
echo TRAIN1000_COMPLETED
