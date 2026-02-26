#!/bin/bash
# =============================================================================
# ResNet50 ImageNet baseline training — Vanderbilt ACCRE DGX A100
# Two A100s via DistributedDataParallel (torchrun --nproc_per_node=2)
# =============================================================================
#
# Submit:
#   sbatch slurm/resnet50_imagenet_baseline.slurm
#
# Monitor:
#   squeue -u $USER
#   tail -f /data/p_dsi/<YOUR_VUNETID>/logs/resnet50_baseline_<JOBID>.out
# =============================================================================

#SBATCH --job-name=resnet50_imagenet_baseline
#SBATCH --account=dsi_dgx_iacc
#SBATCH --partition=interactive_gpu
#SBATCH --qos=dgx_iacc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2            # one process per GPU (torchrun DDP)
#SBATCH --gres=gpu:nvidia_a100-sxm4-40gb:2
#SBATCH --cpus-per-task=16             # 8 DataLoader workers + overhead, per rank
                                        # Keep in sync with num_workers in config
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/data/p_dsi/<YOUR_VUNETID>/logs/resnet50_baseline_%j.out
#SBATCH --error=/data/p_dsi/<YOUR_VUNETID>/logs/resnet50_baseline_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<YOUR_EMAIL>@vanderbilt.edu

# =============================================================================
# Paths — edit these before submitting
# =============================================================================

VUNETID="<YOUR_VUNETID>"
PROJECT_DIR="/data/p_dsi/${VUNETID}/cvpr_perturbations"
IMAGENET_SRC="/data/p_dsi/${VUNETID}/imagenet"          # source on shared NFS
SAVE_PATH="/data/p_dsi/${VUNETID}/training_runs/resnet50_imagenet_baseline"
CONTAINER="/data/p_dsi/singularity-containers/pytorch_25.01-py3.sif"

# =============================================================================
# Job info
# =============================================================================

echo "=========================================="
echo "ResNet50 ImageNet Baseline Training (DDP)"
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
mkdir -p "${LOCAL_DATA}"

echo "Copying ImageNet to local storage: ${LOCAL_DATA} ..."
rsync -a --info=progress2 "${IMAGENET_SRC}/" "${LOCAL_DATA}/"
echo "✓ Data copy complete"
echo "=========================================="

# =============================================================================
# Create output directories on shared storage
# =============================================================================

mkdir -p "${SAVE_PATH}"
mkdir -p "/data/p_dsi/${VUNETID}/logs"

# =============================================================================
# Launch training via torchrun (2 processes, one per A100)
#
# torchrun sets LOCAL_RANK / RANK / WORLD_SIZE automatically.
# The config has use_ddp: true, so run_training_experiment initialises DDP.
# --set overrides redirect img_dir → local /tmp copy and save_path → shared
# storage without editing the YAML config file.
# =============================================================================

singularity exec --nv \
    --bind /data/p_dsi:/data/p_dsi \
    --bind "${LOCAL_DATA}:${LOCAL_DATA}" \
    "${CONTAINER}" \
    torchrun \
        --nproc_per_node=2 \
        --master_addr=localhost \
        --master_port=29500 \
        "${PROJECT_DIR}/scripts/run_training.py" \
            --config "${PROJECT_DIR}/configs/resnet50_imagenet.yaml" \
            --set img_dir="${LOCAL_DATA}" \
            --set save_path="${SAVE_PATH}"

EXIT_CODE=$?

# =============================================================================
# Cleanup
# =============================================================================

echo "=========================================="
echo "Training finished with exit code: ${EXIT_CODE}"
echo "Finished  : $(date)"
echo "Cleaning up ${LOCAL_DATA} ..."
rm -rf "${LOCAL_DATA}"
echo "✓ Cleanup complete"
echo "=========================================="

exit "${EXIT_CODE}"
