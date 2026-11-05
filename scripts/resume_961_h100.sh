#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
test -f handoff_961/VERIFIED
test -f handoff_961/GPU_SMOKE_PASSED
if [ "${WORLD_SIZE:-1}" != 1 ]; then
  echo 'This experiment must resume on a single GPU.' >&2
  exit 1
fi
if compgen -G 'runs_h100/bacteria_species961_train1000_seed42_v1_finetune/checkpoint_epoch*.pt' > /dev/null; then
  echo 'H100 run already contains a checkpoint. Select the new resume point explicitly; refusing to restart from the transfer snapshot.' >&2
  exit 1
fi
exec python -u scripts/finetune.py \
  --config configs/finetune_species961_h100.yaml \
  --resume handoff_961/resume.pt --gpus 0
