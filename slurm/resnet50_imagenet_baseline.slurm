#!/bin/bash
# =============================================================================
# ResNet50 ImageNet baseline training — Vanderbilt ACCRE DGX A100
# =============================================================================
#
# Submit:
#   sbatch slurm/resnet50_imagenet_baseline.slurm
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/resnet50_baseline_<JOBID>.out
#
# Verify the partition name for DGX A100 nodes on your allocation:
#   sinfo -s
# =============================================================================

#SBATCH --job-name=resnet50_imagenet_baseline

# ---- Allocation ----------------------------------------------------------
#SBATCH --account=<YOUR_ACCOUNT>        # Your PI's ACCRE group account
#SBATCH --partition=turing              # DGX A100 partition — verify with sinfo

# ---- Resources -----------------------------------------------------------
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10             # 8 DataLoader workers + 1 main + 1 buffer
                                        # Keep in sync with num_workers in config
#SBATCH --gres=gpu:A100:1              # 1x A100 80 GB
#SBATCH --mem=128G                      # ImageNet images buffered by 8 workers

# ---- Time ----------------------------------------------------------------
# ResNet50 on ImageNet, 1x A100 (no AMP): ~8-10 min/epoch × 100 ep ≈ 14 h
# 24 h gives headroom for I/O variance and early-stopping tail.
#SBATCH --time=24:00:00

# ---- Output --------------------------------------------------------------
# The logs/ directory is gitignored; create it before submitting:
#   mkdir -p logs
#SBATCH --output=logs/resnet50_baseline_%j.out
#SBATCH --error=logs/resnet50_baseline_%j.err

# ---- Notifications -------------------------------------------------------
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<YOUR_EMAIL>

# =============================================================================
# Environment
# =============================================================================

echo "========================================"
echo "Job ID        : $SLURM_JOB_ID"
echo "Node          : $SLURMD_NODENAME"
echo "Partition     : $SLURM_JOB_PARTITION"
echo "GPUs          : $SLURM_JOB_GPUS"
echo "CPUs per task : $SLURM_CPUS_PER_TASK"
echo "Start time    : $(date)"
echo "========================================"

# Activate conda environment
source ~/.bashrc
conda activate cvpr_perturbations

# Make the project root importable regardless of working directory
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"

# Run W&B in offline mode by default; remove or override to log to cloud
export WANDB_MODE=offline

# Navigate to project root (sbatch sets CWD to submission directory by default,
# but being explicit avoids surprises)
cd "$SLURM_SUBMIT_DIR"

# =============================================================================
# Training
# =============================================================================

echo ""
echo "Launching ResNet50 ImageNet baseline training..."
echo "Config : configs/resnet50_imagenet.yaml"
echo ""

python scripts/run_training.py --config configs/resnet50_imagenet.yaml

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Training finished with exit code: $EXIT_CODE"
echo "End time : $(date)"
echo "========================================"

exit $EXIT_CODE
