#!/bin/bash
#SBATCH --job-name=rn50_smoke
#SBATCH --output=./logs/rn50_smoke_%j.out
#SBATCH --error=./logs/rn50_smoke_%j.err
#SBATCH --account=dsi_dgx_iacc
#SBATCH --partition=interactive_gpu
#SBATCH --qos=dgx_iacc
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marren.jenkins.1@vanderbilt.edu
# =============================================================================
# ResNet50 ImageNet pipeline smoke test
# =============================================================================
#
# Shrinks every dial (1 GPU, 4 CPUs, 16G RAM, 30 min) so this queues fast
# and fails fast. Runs preflight checks, then a 1-epoch training pass over
# 256 train / 128 val samples to exercise the pipeline end-to-end.
#
# Submit from the repo root:
#   cd /path/to/cvpr_perturbations
#   mkdir -p logs
#   sbatch resnet50_imagenet_smoke.slurm
# =============================================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

# --- Environment setup ------------------------------------------------------
module --force purge
module load StdEnv/2023 gcc python/3.11 arrow/23.0.1
source ~/envs/clip_hba_cc/bin/activate


python -c "import numpy; print(numpy.__version__); import numpy as np; print(np.dot(np.ones(4), np.ones(4)))"
python -c "import pyarrow, datasets, torch; print(pyarrow.__version__, datasets.__version__, torch.__version__)"

echo "=========================================="
echo "ResNet50 ImageNet SMOKE TEST"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $(hostname)"
echo "Partition    : $SLURM_JOB_PARTITION"
echo "GPUs         : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start time   : $(date)"
echo "Working dir  : $(pwd)"
echo "Python       : $(which python)"
echo "=========================================="

# --- HuggingFace token ------------------------------------------------------
export HF_HOME=/tmp/hf_cache
export HF_DATASETS_CACHE=/tmp/hf_cache/datasets
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.cache/huggingface/token" ]]; then
  export HF_TOKEN="$(cat "${HOME}/.cache/huggingface/token")"
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN not set and ~/.cache/huggingface/token missing."
  exit 1
fi

# --- Preflight checks -------------------------------------------------------
echo ""
echo "---- Preflight: imports + GPU + HF stream ----"
python - <<'PY'
import os, sys
print(f"python           : {sys.version.split()[0]}")

import pyarrow, datasets, torch, torchvision
print(f"pyarrow          : {pyarrow.__version__}")
print(f"datasets         : {datasets.__version__}")
print(f"torch            : {torch.__version__}")
print(f"torchvision      : {torchvision.__version__}")

assert torch.cuda.is_available(), "CUDA not visible to torch"
print(f"cuda device count: {torch.cuda.device_count()}")
print(f"cuda device name : {torch.cuda.get_device_name(0)}")

from datasets import load_dataset
token = os.environ.get("HF_TOKEN")
ds = load_dataset("ILSVRC/imagenet-1k", split="validation",
                  streaming=True, token=token)
sample = next(iter(ds))
print(f"hf sample keys   : {list(sample.keys())}")
print("preflight OK")
PY

# --- Full pipeline smoke run ------------------------------------------------
SMOKE_SAVE_PATH="${SLURM_SUBMIT_DIR}/training_runs/resnet50_imagenet_smoke"
mkdir -p "${SMOKE_SAVE_PATH}"

echo ""
echo "---- Smoke training run (1 epoch, 256 train / 128 val) ----"
python "${SLURM_SUBMIT_DIR}/scripts/run_training.py" \
    --config "${SLURM_SUBMIT_DIR}/configs/resnet50_imagenet_smoke.yaml" \
    --no_wandb

EXIT_CODE=$?

echo "=========================================="
echo "Finished at  : $(date)"
echo "Exit code    : ${EXIT_CODE}"
echo "=========================================="

exit "${EXIT_CODE}"
