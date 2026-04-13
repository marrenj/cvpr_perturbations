#!/bin/bash
#SBATCH --job-name=resnet50_imagenet_baseline
#SBATCH --output=./logs/resnet50_imagenet_baseline_%j.out
#SBATCH --error=./logs/resnet50_imagenet_baseline_%j.err
#SBATCH --account=dsi_dgx_iacc
#SBATCH --partition=interactive_gpu
#SBATCH --qos=dgx_iacc
#SBATCH --gres=gpu:nvidia_a100-sxm4-40gb:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=30:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marren.jenkins.1@vanderbilt.edu
# =============================================================================
# ResNet50 ImageNet baseline training — Vanderbilt ACCRE DGX A100
# =============================================================================
#
# Confirm location: sacct -j <JOBID> --format=WorkDir -P | tail -1
#
# Submit (must cd to repo root first so SLURM_SUBMIT_DIR is the checkout):
#   cd /path/to/cvpr_perturbations
#   sbatch slurm/resnet50_imagenet_baseline.slurm
#
# Monitor:
#   squeue -u $USER
#   tail -f resnet50_imagenet_baseline_<JOBID>.out
# =============================================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
if [[ ! -f "${SLURM_SUBMIT_DIR}/scripts/run_training.py" ]]; then
  echo "ERROR: scripts/run_training.py not found under SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}"
  echo "cd to the repo root, then: sbatch slurm/resnet50_imagenet_baseline.slurm"
  exit 1
fi

# --- Environment setup ------------------------------------------------------
# conda activate clip-hba
module load python/3.11.5
source ~/envs/clip_hba/bin/activate

echo "=========================================="
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "GPUs         : $CUDA_VISIBLE_DEVICES"
echo "Start time   : $(date)"
echo "Working dir  : $(pwd)"
echo "=========================================="

# =============================================================================
# Paths — edit these before submitting
# =============================================================================

IMAGENET_SRC="/home/jenkm22/imagenet"          # source on shared NFS
SAVE_PATH="${SLURM_SUBMIT_DIR}/training_runs/resnet50_imagenet_baseline"

# =============================================================================
# Job info
# =============================================================================

echo "=========================================="
echo "ResNet50 ImageNet Baseline Training"
echo "=========================================="
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $(hostname)"
echo "Partition : $SLURM_JOB_PARTITION"
echo "GPUs      : $SLURM_JOB_GPUS"
echo "Started   : $(date)"
echo "=========================================="

# =============================================================================
# Extract ImageNet tars to node-local scratch (not counted against quota)
# =============================================================================

echo "Extracting ImageNet to /tmp/ directory..."
mkdir -p /tmp/imagenet
tar -xf /nobackup/userspace/jenkm22/imagenet/imagenet_train.tar -C /tmp/imagenet/
tar -xf /nobackup/userspace/jenkm22/imagenet/imagenet_val.tar   -C /tmp/imagenet/

# =============================================================================
# Create output directories on shared storage
# =============================================================================

mkdir -p "${SAVE_PATH}"

# =============================================================================
# Two A100s are used via DataParallel (cuda: -1).
# =============================================================================

python "${SLURM_SUBMIT_DIR}/scripts/run_training.py" \
    --config "${SLURM_SUBMIT_DIR}/configs/resnet50_imagenet.yaml" \
    --override img_dir=/tmp/imagenet

EXIT_CODE=$?

exit "${EXIT_CODE}"
