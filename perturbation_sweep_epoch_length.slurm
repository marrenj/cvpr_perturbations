#!/bin/bash
#SBATCH --job-name=perturbation_sweep_epoch_length
#SBATCH --output=./logs/perturbation_sweep_epoch_length_%j.out
#SBATCH --error=./logs/perturbation_sweep_epoch_length_%j.err
#SBATCH --account=dsi_dgx_iacc
#SBATCH --partition=interactive_gpu
#SBATCH --qos=dgx_iacc
#SBATCH --gres=gpu:nvidia_a100-sxm4-40gb:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marren.jenkins.1@vanderbilt.edu
# =============================================================================
# Perturbation epoch-length sweep — Vanderbilt ACCRE DGX A100
# =============================================================================
#
# Runs run_epoch_length_sweep_cuda0.sh and run_epoch_length_sweep_cuda1.sh
# concurrently, each pinned to a single A100 inside the SLURM allocation.
#
# Submit (must cd to repo root first so SLURM_SUBMIT_DIR is the checkout):
#   cd /path/to/cvpr_perturbations
#   sbatch perturbation_sweep_epoch_length.slurm
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/perturbation_sweep_epoch_length_<JOBID>.out
#   tail -f logs/perturbation_sweep_cuda0_<JOBID>.out
#   tail -f logs/perturbation_sweep_cuda1_<JOBID>.out
#
# BEFORE SUBMITTING — edit these in BOTH run_epoch_length_sweep_cuda{0,1}.sh
# to ACCRE-resident paths (the defaults point at the lab workstation):
#   SAVE_ROOT          (writable scratch / project storage on ACCRE)
#   IMG_DIR            (THINGS images on ACCRE)
#   IMG_ANNOTATIONS_FILE (lives in repo, usually fine as-is)
# Also confirm that the baseline checkpoints referenced as
#   ${SAVE_ROOT}/baseline_seed{1,2,3}/vit_l_14_rank32_perturb-type-none_init-seed{1,2,3}behavioral-rsa-True
# exist on ACCRE before launching.
# =============================================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
if [[ ! -f "${SLURM_SUBMIT_DIR}/run_epoch_length_sweep_cuda0.sh" || \
      ! -f "${SLURM_SUBMIT_DIR}/run_epoch_length_sweep_cuda1.sh" ]]; then
  echo "ERROR: sweep scripts not found under SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}"
  echo "cd to the repo root, then: sbatch perturbation_sweep_epoch_length.slurm"
  exit 1
fi

# --- Environment setup ------------------------------------------------------
module --force purge
module load StdEnv/2023 gcc python/3.11 arrow/23.0.1
source ~/envs/clip_hba_cc/bin/activate

python -c "import pyarrow, datasets, torch; print(pyarrow.__version__, datasets.__version__, torch.__version__)"

echo "=========================================="
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "Partition    : $SLURM_JOB_PARTITION"
echo "GPUs (slurm) : $SLURM_JOB_GPUS"
echo "GPUs (visi.) : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start time   : $(date)"
echo "Working dir  : $(pwd)"
echo "=========================================="

mkdir -p "${SLURM_SUBMIT_DIR}/logs"

# =============================================================================
# Concurrent launch — each .sh script is pinned to one of the two allocated
# A100s. SLURM exposes the allocation as device indices 0..N-1 inside the job
# (CUDA_VISIBLE_DEVICES remap), which matches the hardcoded CUDA=0 / CUDA=1 in
# the sweep scripts. Each script's stdout/stderr lands in its own log file so
# the parent job log stays readable.
# =============================================================================

LOG_CUDA0="${SLURM_SUBMIT_DIR}/logs/perturbation_sweep_cuda0_${SLURM_JOB_ID}.out"
LOG_CUDA1="${SLURM_SUBMIT_DIR}/logs/perturbation_sweep_cuda1_${SLURM_JOB_ID}.out"

PYTHON_CMD="$(command -v python)"
export PYTHON_CMD

bash "${SLURM_SUBMIT_DIR}/run_epoch_length_sweep_cuda0.sh" \
    > "${LOG_CUDA0}" 2>&1 &
PID_0=$!
echo "Launched cuda0 sweep (pid ${PID_0}) -> ${LOG_CUDA0}"

bash "${SLURM_SUBMIT_DIR}/run_epoch_length_sweep_cuda1.sh" \
    > "${LOG_CUDA1}" 2>&1 &
PID_1=$!
echo "Launched cuda1 sweep (pid ${PID_1}) -> ${LOG_CUDA1}"

# Wait for both even if one fails so the surviving sweep still finishes.
set +e
wait "${PID_0}"; EXIT_0=$?
wait "${PID_1}"; EXIT_1=$?
set -e

echo "cuda0 sweep exit=${EXIT_0}"
echo "cuda1 sweep exit=${EXIT_1}"
echo "Finished     : $(date)"

if [[ "${EXIT_0}" -ne 0 || "${EXIT_1}" -ne 0 ]]; then
  exit 1
fi
exit 0
