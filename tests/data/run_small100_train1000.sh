#!/usr/bin/env bash
set -euo pipefail
cd /home/user/zhangguoli/species_identification
: "${CONDA_PREFIX:?Activate evo2_hf first}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
pretrained=/home/user/zhangguoli/species_identification/runs/bacteria_species_cls_v1_pretrain_v2/best_model.pt
test -f "$pretrained"
run=runs/bacteria_small100_train1000_seed42_v1_finetune
if [ -e "$run" ]; then
  echo "Training output already exists: $run"
  exit 1
fi
out=/home/user/zhangguoli/species_identification/index_bacteria_small100_train1000
if [ -e "$out" ]; then
  python -u tests/data/build_trainable_windows.py --validate-only --output-dir "$out"
else
  python -u tests/data/build_trainable_windows.py \
    --expand-from /home/user/zhangguoli/species_identification/index_bacteria_small32_train1000 \
    --extend-species 100 --train-per-species 1000 --val-per-species 25 --test-per-species 25 \
    --base-config /home/user/zhangguoli/species_identification/configs/finetune_small32_train1000_equal_budget.yaml \
    --seed 42 --output-dir "$out"
fi
python - <<'PY'
from pathlib import Path
import json
import yaml
root = Path('/home/user/zhangguoli/species_identification')
out = root / 'index_bacteria_small100_train1000'
report = json.loads((out / 'data_report.json').read_text())
assert report['status'] == 'passed' and report['species'] == 100
assert report['windows'] == dict(train=100000, val=2500, test=2500)
checks = report['expansion_checks']
assert checks['baseline_species'] == 32 and checks['added_species'] == 68
assert checks['original_train_windows_retained'] == 32000 and checks['added_train_windows'] == 0
assert checks['validation_windows_unchanged'] and checks['test_windows_unchanged']
assert report['train_unique_windows'] == 99731 and report['train_repeated_entries'] == 269
cfg = yaml.safe_load((out / 'config_data.yaml').read_text())
cfg.update(experiment_name='bacteria_small100_train1000_seed42_v1', output_dir='./runs', seed=42)
cfg['training'].update(finetune_epochs=10, batch_size_per_gpu=4,
    gradient_accumulation_steps=8, warmup_steps=200, steps_per_epoch=None,
    val_steps=None, species_balanced_fraction=0.0, guidance_dropout=1.0,
    disable_phylo=True, classification_loss_only=False, track_species_exposure=True,
    early_stop_patience=0)
path = root / 'configs/finetune_small100_train1000_seed42.yaml'
path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
print('Data passed; 100 species / 10 epochs / 31250 updates / 200 warmup updates', flush=True)
PY
python -u scripts/finetune.py \
  --config configs/finetune_small100_train1000_seed42.yaml \
  --pretrained "$pretrained" --gpus 0
echo SMALL100_TRAINING_COMPLETED
