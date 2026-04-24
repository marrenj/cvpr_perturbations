#!/bin/bash
#SBATCH --job-name=resnet50_imagenet_debug
#SBATCH --output=./logs/resnet50_imagenet_debug_%j.out
#SBATCH --error=./logs/resnet50_imagenet_debug_%j.err
#SBATCH --account=dsi_dgx_iacc
#SBATCH --partition=interactive_gpu
#SBATCH --qos=dgx_iacc
#SBATCH --gres=gpu:nvidia_a100-sxm4-40gb:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marren.jenkins.1@vanderbilt.edu
# =============================================================================
# ResNet50 ImageNet DEBUG run (2 epochs) — verify HF-stream truncation fix
# =============================================================================
#
# Submit from the repo root:
#   cd /path/to/cvpr_perturbations
#   sbatch resnet50_imagenet_debug.slurm
#
# What to look for in the log:
#   * Train progress bar reaches 5005/5005 (not 637/5005).
#   * Val progress bar reaches  196/196  (not  28/196).
#   * No "Too many dataloader workers" warning (or if present, lower
#     num_workers in configs/resnet50_imagenet_debug.yaml to match n_shards).
# =============================================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
if [[ ! -f "${SLURM_SUBMIT_DIR}/scripts/run_training.py" ]]; then
  echo "ERROR: scripts/run_training.py not found under SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}"
  echo "cd to the repo root, then: sbatch resnet50_imagenet_debug.slurm"
  exit 1
fi

# --- Environment setup ------------------------------------------------------
module --force purge
module load StdEnv/2023 gcc python/3.11 arrow/23.0.1
source ~/envs/clip_hba_cc/bin/activate

python -c "import pyarrow, datasets, torch; print(pyarrow.__version__, datasets.__version__, torch.__version__)"

echo "=========================================="
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "GPUs         : $CUDA_VISIBLE_DEVICES"
echo "Start time   : $(date)"
echo "Working dir  : $(pwd)"
echo "=========================================="

SAVE_PATH="${SLURM_SUBMIT_DIR}/training_runs/resnet50_imagenet_debug"

# --- HuggingFace streaming setup -------------------------------------------
export HF_HOME=/tmp/hf_cache
export HF_DATASETS_CACHE=/tmp/hf_cache/datasets
unset HF_HUB_ENABLE_HF_TRANSFER
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.cache/huggingface/token" ]]; then
  export HF_TOKEN="$(cat "${HOME}/.cache/huggingface/token")"
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN not set and ~/.cache/huggingface/token missing."
  echo "Run 'huggingface-cli login' once and accept the ILSVRC/imagenet-1k license."
  exit 1
fi

mkdir -p "${SAVE_PATH}"

# --- Launch ---------------------------------------------------------------
python "${SLURM_SUBMIT_DIR}/scripts/run_training.py" \
    --config "${SLURM_SUBMIT_DIR}/configs/resnet50_imagenet_debug.yaml" \
    --no_wandb

EXIT_CODE=$?

exit "${EXIT_CODE}"
