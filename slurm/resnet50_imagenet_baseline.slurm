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
# Submit:
#   sbatch slurm/resnet50_imagenet_baseline.slurm
#
# Monitor:
#   squeue -u $USER
#   tail -f ./logs/resnet50_baseline_<JOBID>.out
# =============================================================================

set -euo pipefail
 
# --- Environment setup ------------------------------------------------------
# conda activate clip-hba
module load python/3.11.5
source ~/envs/clip_hba/bin/activate
 
# cd /panfs/accrepfs.vampire/home/jenkm22/cvpr_perturbations
 
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

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
IMAGENET_SRC="/home/jenkm22/imagenet"          # source on shared NFS
SAVE_PATH="./training_runs/resnet50_imagenet_baseline"
CONTAINER="./pytorch_25.01-py3.sif"

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
# Create output directories on shared storage
# =============================================================================

mkdir -p "${SAVE_PATH}"
mkdir -p "./logs"

# =============================================================================
# Two A100s are used via DataParallel (cuda: -1).
# =============================================================================

python ./scripts/run_training.py \
    --config ./configs/resnet50_imagenet.yaml 

EXIT_CODE=$?

exit "${EXIT_CODE}"
