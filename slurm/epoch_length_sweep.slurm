#!/bin/bash
#SBATCH --job-name=epoch_length_sweep
#SBATCH --output=slurm/logs/epoch_length_sweep_%j.out
#SBATCH --error=slurm/logs/epoch_length_sweep_%j.err
#SBATCH --account=dsi_dgx_iacc
#SBATCH --partition=interactive_gpu
#SBATCH --qos=dgx_iacc
#SBATCH --gres=gpu:nvidia_a100-sxm4-40gb:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=marren.jenkins.1@vanderbilt.edu

# =============================================================================
# Epoch-length perturbation sweep — Vanderbilt ACCRE DGX A100
#
# Runs two sweep scripts in parallel on separate GPUs:
#   GPU 0 — random_target, label_shuffle   (run_epoch_length_sweep_cuda0.sh)
#   GPU 1 — image_noise, uniform_images    (run_epoch_length_sweep_cuda1.sh)
#
# Submit from the repo root:
#   cd /path/to/cvpr_perturbations
#   mkdir -p slurm/logs
#   sbatch slurm/epoch_length_sweep.slurm
#
# Monitor:
#   squeue -u $USER
#   tail -f slurm/logs/epoch_length_sweep_<JOBID>.out
# =============================================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
if [[ ! -f "${SLURM_SUBMIT_DIR}/scripts/run_training.py" ]]; then
  echo "ERROR: scripts/run_training.py not found under SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}"
  echo "cd to the repo root, then: sbatch slurm/epoch_length_sweep.slurm"
  exit 1
fi

# --- Environment setup -------------------------------------------------------
module --force purge
module load StdEnv/2023 gcc python/3.11 arrow/23.0.1
source ~/envs/clip_hba_cc/bin/activate

export PYTHON_CMD=python

# --- Paths -------------------------------------------------------------------
export SAVE_ROOT="/home/jenkm22/length_sweep_checkpoints"
mkdir -p "${SAVE_ROOT}"

# --- Job info ----------------------------------------------------------------
echo "=========================================="
echo "Epoch-Length Perturbation Sweep"
echo "=========================================="
echo "Job ID    : ${SLURM_JOB_ID}"
echo "Node      : $(hostname)"
echo "Partition : ${SLURM_JOB_PARTITION}"
echo "GPUs      : ${CUDA_VISIBLE_DEVICES}"
echo "Save root : ${SAVE_ROOT}"
echo "Started   : $(date)"
echo "=========================================="

# --- Launch both sweeps in parallel on separate GPUs -------------------------
# Redirect each worker's stdout/stderr to its own log for easier debugging.
LOG_DIR="slurm/logs"
mkdir -p "${LOG_DIR}"

CUDA=0 bash "${SLURM_SUBMIT_DIR}/run_epoch_length_sweep_cuda0.sh" \
  > "${LOG_DIR}/epoch_sweep_cuda0_${SLURM_JOB_ID}.out" \
  2> "${LOG_DIR}/epoch_sweep_cuda0_${SLURM_JOB_ID}.err" &
PID0=$!

CUDA=1 bash "${SLURM_SUBMIT_DIR}/run_epoch_length_sweep_cuda1.sh" \
  > "${LOG_DIR}/epoch_sweep_cuda1_${SLURM_JOB_ID}.out" \
  2> "${LOG_DIR}/epoch_sweep_cuda1_${SLURM_JOB_ID}.err" &
PID1=$!

echo "cuda0 sweep PID: ${PID0}"
echo "cuda1 sweep PID: ${PID1}"

# Wait for both workers and capture their exit codes.
EXIT0=0
EXIT1=0
wait "${PID0}" || EXIT0=$?
wait "${PID1}" || EXIT1=$?

echo "=========================================="
echo "Finished   : $(date)"
echo "cuda0 exit : ${EXIT0}"
echo "cuda1 exit : ${EXIT1}"
echo "=========================================="

# Fail the job if either worker failed.
if [[ "${EXIT0}" -ne 0 || "${EXIT1}" -ne 0 ]]; then
  echo "ERROR: one or both sweeps exited with a non-zero status."
  exit 1
fi
