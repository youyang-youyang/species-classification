#!/bin/bash
#SBATCH --job-name=species961_counts
#SBATCH --partition=q03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/user/zhangguoli/species_identification/tests/data/species961_counts_%j.log
set -euo pipefail
cd /home/user/zhangguoli/species_identification
OUT="$PWD/auxiliary_species961_train_counts_${SLURM_JOB_ID}"
echo "START $(date -Is) host=$(hostname) output=$OUT"
/home/user/zhangguoli/miniconda3/envs/evo2_hf/bin/python -u \
  tests/data/build_species961_counts.py \
  --index-dir "$PWD/index_bacteria_species961_train1000" \
  --output-dir "$OUT" \
  --kmc-bin-dir /home/user/zhangguoli/miniconda3/envs/ont_project/bin \
  --threads "$SLURM_CPUS_PER_TASK" --memory-gb 64
echo "FINISHED $(date -Is)"
