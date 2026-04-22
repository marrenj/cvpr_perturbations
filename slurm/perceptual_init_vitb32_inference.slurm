#!/bin/bash
#SBATCH --mail-user=marren.jenkins.1@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=pi_vitb32_inference
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --partition=interactive_gpu
#SBATCH --account=dsi_dgx_iacc
#SBATCH --qos=dgx_iacc
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1

#SBATCH --output=logs/pi_vitb32_inference.log
#SBATCH --error=logs/pi_vitb32_inference.log

set -euo pipefail

echo "Host: $(hostname)"
nvidia-smi

module --force purge
module load StdEnv/2023 gcc python/3.11
source ~/envs/cvpr_perturbations/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

CHECKPOINT_DIR="/home/wallacelab/teba/multimodal_brain_inspired/PI_dk/pct_checkpoints/perceptual_init_yfcc3m_vitb32"
SAVE_DIR="/home/wallacelab/teba/multimodal_brain_inspired/PI_dk/inference/perceptual_init_yfcc3m_vitb32"
RUN_NAME="perceptual_init_yfcc3m_vitb32"

mkdir -p "${SAVE_DIR}"
mkdir -p logs

echo "Running perceptual-init ViT-B/32 inference sweep..."
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_perceptual_init_inference.py \
    --config         configs/perceptual_init_vitb32_inference.yaml \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --save_dir       "${SAVE_DIR}" \
    --run_name       "${RUN_NAME}"

echo "Inference sweep finished."
