#!/bin/bash
# =============================================================================
# ResNet50 ImageNet perturbation sweep — Vanderbilt ACCRE DGX A100
#
# Submits 93 array tasks: 3 perturbation types × 31 epoch positions.
# Each task applies one perturbation type for exactly one epoch, then
# resumes normal training.
#
# Prerequisites:
#   1. A completed baseline run (perturb_type=none) whose per-epoch
#      checkpoints are saved under BASELINE_CHECKPOINT_PATH.
#   2. ImageNet data available at IMAGENET_SRC.
#
# Submit:
#   sbatch slurm/resnet50_imagenet_perturbation_sweep.slurm
#
# Monitor:
#   squeue -u $USER
#   tail -f /data/p_dsi/<YOUR_VUNETID>/logs/perturbation_sweep_<JOBID>_<TASKID>.out
#
# List all 93 tasks without running anything:
#   python scripts/run_perturbation_sweep.py \
#     --config configs/resnet50_imagenet.yaml --list-tasks
# =============================================================================

#SBATCH --job-name=resnet50_perturbation_sweep
#SBATCH --account=
#SBATCH --partition=interactive_gpu
#SBATCH --qos=dgx_iacc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:nvidia_a100-sxm4-40gb:2
#SBATCH --cpus-per-task=16              # 16 CPUs × 2 tasks = 32 total; 8 workers/GPU
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --array=0-92%10                 # 93 tasks; max 10 running concurrently
#SBATCH --output=/data/<YOUR_VUNETID>/logs/perturbation_sweep_%A_%a.out
#SBATCH --error=/data/<YOUR_VUNETID>/logs/perturbation_sweep_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marren.jenkins.1@vanderbilt.edu

# =============================================================================
# Paths — edit these before submitting
# =============================================================================

VUNETID="<YOUR_VUNETID>"
PROJECT_DIR="/data/p_dsi/${VUNETID}/cvpr_perturbations"
IMAGENET_SRC="/data/p_dsi/${VUNETID}/imagenet"
SAVE_PATH="/data/p_dsi/${VUNETID}/training_runs/resnet50_imagenet_perturbation_sweep"
BASELINE_CHECKPOINT_PATH="/data/p_dsi/${VUNETID}/training_runs/resnet50_imagenet_baseline/baseline_seed1"
CONTAINER="/data/p_dsi/singularity-containers/pytorch_25.01-py3.sif"

# =============================================================================
# Job info
# =============================================================================

echo "=========================================="
echo "ResNet50 ImageNet Perturbation Sweep"
echo "=========================================="
echo "Job ID    : $SLURM_ARRAY_JOB_ID"
echo "Task ID   : $SLURM_ARRAY_TASK_ID"
echo "Node      : $(hostname)"
echo "Partition : $SLURM_JOB_PARTITION"
echo "GPUs      : $SLURM_JOB_GPUS"
echo "Started   : $(date)"
echo "=========================================="

# =============================================================================
# Copy ImageNet to local /tmp for fast I/O
# =============================================================================

LOCAL_DATA=/tmp/imagenet_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${LOCAL_DATA}"

echo "Copying ImageNet to local storage: ${LOCAL_DATA} ..."
rsync -a --info=progress2 "${IMAGENET_SRC}/" "${LOCAL_DATA}/"
echo "✓ Data copy complete"
echo "=========================================="

# =============================================================================
# Create output directories
# =============================================================================

mkdir -p "${SAVE_PATH}"
mkdir -p "/data/p_dsi/${VUNETID}/logs"

# =============================================================================
# Launch training inside the Singularity container
# =============================================================================

singularity exec --nv \
    --bind /data/p_dsi:/data/p_dsi \
    --bind "${LOCAL_DATA}:${LOCAL_DATA}" \
    "${CONTAINER}" \
    python "${PROJECT_DIR}/scripts/run_perturbation_sweep.py" \
        --config "${PROJECT_DIR}/configs/resnet50_imagenet.yaml" \
        --task-id "${SLURM_ARRAY_TASK_ID}" \
        --img-dir "${LOCAL_DATA}" \
        --save-path "${SAVE_PATH}" \
        --baseline-checkpoint-path "${BASELINE_CHECKPOINT_PATH}"

EXIT_CODE=$?

# =============================================================================
# Cleanup
# =============================================================================

echo "=========================================="
echo "Task ${SLURM_ARRAY_TASK_ID} finished with exit code: ${EXIT_CODE}"
echo "Finished  : $(date)"
echo "Cleaning up ${LOCAL_DATA} ..."
rm -rf "${LOCAL_DATA}"
echo "✓ Cleanup complete"
echo "=========================================="

exit "${EXIT_CODE}"
