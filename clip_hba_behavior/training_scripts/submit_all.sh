#!/bin/bash
#SBATCH --job-name=perturb_sweep
#SBATCH --output=/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior_loops/logs/%x_%A_%a.out
#SBATCH --error=/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior_loops/logs/%x_%A_%a.err
#SBATCH --array=1-32
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# --- Load environment ---
module load anaconda
conda activate adaptive-clip

# --- Create logs directory ---
mkdir -p /home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior_loops/logs

# --- Fetch the command corresponding to this SLURM array index ---
COMMAND=$(sed -n "${SLURM_ARRAY_TASK_ID}p" experiment_runs.tsv | cut -f2-)
echo "Running job ${SLURM_ARRAY_TASK_ID}: ${COMMAND}"

# --- Execute training ---
eval "${COMMAND}"