#!/bin/bash
#SBATCH --job-name=resnet50_imagenet_baseline
#SBATCH --output=./logs/resnet50_imagenet_baseline_%j.out
#SBATCH --error=./logs/resnet50_imagenet_baseline_%j.err
#SBATCH --account=dsi_dgx_iacc
#SBATCH --partition=interactive_gpu
#SBATCH --qos=dgx_iacc
#SBATCH --gres=gpu:nvidia_a100-sxm4-40gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=10:00:00
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
 
cd /panfs/accrepfs.vampire/home/jenkm22/perturbations
 
echo "=========================================="
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "GPUs         : ${CUDA_VISIBLE_DEVICES:-none}"
echo "Start time   : $(date)"
echo "Working dir  : $(pwd)"
echo "=========================================="

# =============================================================================
# Paths — edit these before submitting
# =============================================================================

PROJECT_DIR="./cvpr_perturbations"
TAR_SRC="/nobackup/userspace/jenkm22/imagenet"  # contains train.tar and val.tar
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
# Copy ImageNet to local /tmp for fast I/O
# (NFS throughput is the bottleneck for ImageNet; /tmp is local NVMe)
# =============================================================================

LOCAL_DATA=/tmp/imagenet_${SLURM_JOB_ID}

if [ ! -d "${LOCAL_DATA}/train" ]; then
    echo "Extracting ImageNet to local storage: ${LOCAL_DATA}..."
    mkdir -p "${LOCAL_DATA}/train" "${LOCAL_DATA}/val"

    # Count entries first (reads only tar headers — fast)
    echo "Counting archive contents..."
    TRAIN_TOTAL=$(tar -tf "${TAR_SRC}/train.tar" | wc -l)
    VAL_TOTAL=$(tar -tf "${TAR_SRC}/val.tar" | wc -l)
    echo "  train.tar : ${TRAIN_TOTAL} files"
    echo "  val.tar   : ${VAL_TOTAL} files"

    # Extract train split — sample every 50 000 files for progress
    echo ""
    echo "Extracting train.tar..."
    tar -xvf "${TAR_SRC}/train.tar" -C "${LOCAL_DATA}/train/" 2>&1 | \
        awk -v total="${TRAIN_TOTAL}" \
            'NR % 50000 == 0 { printf "  [train] %d / %d files (%.1f%%)\n", NR, total, NR*100.0/total; fflush() }'
    echo "  [train] Done — ${TRAIN_TOTAL} files extracted."

    # Extract val split — sample every 5 000 files for progress
    echo ""
    echo "Extracting val.tar..."
    tar -xvf "${TAR_SRC}/val.tar" -C "${LOCAL_DATA}/val/" 2>&1 | \
        awk -v total="${VAL_TOTAL}" \
            'NR % 5000 == 0 { printf "  [val]   %d / %d files (%.1f%%)\n", NR, total, NR*100.0/total; fflush() }'
    echo "  [val] Done — ${VAL_TOTAL} files extracted."
else
    echo "Using cached ImageNet data at ${LOCAL_DATA}"
fi

echo "=========================================="

EXIT_CODE=$?

# =============================================================================
# Cleanup
# =============================================================================

echo "=========================================="
echo "ImageNet transfer finished with exit code: ${EXIT_CODE}"
echo "Finished  : $(date)"
echo "✓ Cleanup complete"
echo "=========================================="

exit "${EXIT_CODE}"