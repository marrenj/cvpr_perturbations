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

TAR_SRC="/nobackup/userspace/jenkm22/imagenet"  # contains imagenet_train.tar and imagenet_val.tar
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
# Create output directories on shared storage
# =============================================================================

mkdir -p "${SAVE_PATH}"

# =============================================================================
# Extract ImageNet to node-local /tmp for fast I/O
# (not counted against quota; cleaned up automatically when the job ends)
# =============================================================================

LOCAL_DATA="/tmp/imagenet_${SLURM_JOB_ID}"

if [ ! -d "${LOCAL_DATA}/train" ]; then
    echo "Extracting ImageNet to local storage: ${LOCAL_DATA}..."
    mkdir -p "${LOCAL_DATA}"

    # Count entries first (reads only tar headers — fast)
    echo "Counting archive contents..."
    TRAIN_TOTAL=$(tar -tf "${TAR_SRC}/imagenet_train.tar" | wc -l)
    VAL_TOTAL=$(tar -tf "${TAR_SRC}/imagenet_val.tar" | wc -l)
    echo "  imagenet_train.tar : ${TRAIN_TOTAL} entries"
    echo "  imagenet_val.tar   : ${VAL_TOTAL} entries"

    # Extract train split — tar contains train/ prefix, so extract to LOCAL_DATA/
    # Result: ${LOCAL_DATA}/train/n01234567/img.JPEG
    echo ""
    echo "Extracting imagenet_train.tar..."
    tar -xvf "${TAR_SRC}/imagenet_train.tar" -C "${LOCAL_DATA}/" 2>&1 | \
        awk -v total="${TRAIN_TOTAL}" \
            'NR % 50000 == 0 { printf "  [train] %d / %d entries (%.1f%%)\n", NR, total, NR*100.0/total; fflush() }'
    echo "  [train] Done — ${TRAIN_TOTAL} entries extracted."

    # Extract val split — sample every 5 000 entries for progress
    echo ""
    echo "Extracting imagenet_val.tar..."
    tar -xvf "${TAR_SRC}/imagenet_val.tar" -C "${LOCAL_DATA}/" 2>&1 | \
        awk -v total="${VAL_TOTAL}" \
            'NR % 5000 == 0 { printf "  [val]   %d / %d entries (%.1f%%)\n", NR, total, NR*100.0/total; fflush() }'
    echo "  [val] Done — ${VAL_TOTAL} entries extracted."
else
    echo "Using cached ImageNet data at ${LOCAL_DATA}"
fi

echo "=========================================="

# Stamp the job-specific LOCAL_DATA path into a temporary config so the
# training script sees the extracted directory rather than the tar source.
TMP_CONFIG="/tmp/resnet50_config_${SLURM_JOB_ID}.yaml"
sed "s|img_dir:.*|img_dir: \"${LOCAL_DATA}\"|" \
    "${SLURM_SUBMIT_DIR}/configs/resnet50_imagenet.yaml" > "${TMP_CONFIG}"
echo "Config img_dir set to: ${LOCAL_DATA}"

# =============================================================================
# Two A100s are used via DataParallel (cuda: -1).
# =============================================================================

python "${SLURM_SUBMIT_DIR}/scripts/run_training.py" \
    --config "${TMP_CONFIG}"

EXIT_CODE=$?

exit "${EXIT_CODE}"
