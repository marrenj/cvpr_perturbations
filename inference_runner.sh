#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root (directory containing this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}" && pwd)"

BASE_CONFIG="${BASE_CONFIG:-${REPO_ROOT}/configs/inference_config.yaml}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

BASE="/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/test"

# ---------------------------------------------------------------------------
# run_sweep <checkpoint_dir> <save_dir> <run_name>
#   Calls the single-process sweep script for one set of checkpoints.
# ---------------------------------------------------------------------------
run_sweep() {
  local ckpt_dir="$1"
  local save_dir="$2"
  local run_name="$3"

  if ! compgen -G "${ckpt_dir}/epoch*_dora_params.pth" > /dev/null 2>&1; then
    echo "  [SKIP] No checkpoints in ${ckpt_dir}"
    return 0
  fi

  echo "  [RUN]  run_name=${run_name}"
  echo "         ckpt_dir=${ckpt_dir}"
  echo "         save_dir=${save_dir}"

  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python3 scripts/run_inference_sweep.py \
    --config         "${BASE_CONFIG}" \
    --checkpoint_dir "${ckpt_dir}" \
    --save_dir       "${save_dir}" \
    --run_name       "${run_name}"
}

# ---------------------------------------------------------------------------
# sweep_run_root <run_root>
#   For a flat run (e.g. baseline_seed1), finds the one training-run subdir
#   that contains dora_params/ and sweeps it.
# ---------------------------------------------------------------------------
sweep_run_root() {
  local run_root="$1"
  local training_run_dir
  training_run_dir="$(find "$run_root" -maxdepth 1 -type d -name 'vit_l_14*' | head -1)"
  if [[ -z "$training_run_dir" ]]; then
    echo "[WARN] No vit_l_14* subdir found under ${run_root}, skipping."
    return 0
  fi
  local ckpt_dir="${training_run_dir}/dora_params"
  local run_name
  run_name="$(basename "$training_run_dir")"
  run_sweep "$ckpt_dir" "$run_root" "$run_name"
}

# ---------------------------------------------------------------------------
# sweep_perturbation_root <perturb_root>
#   For a folder containing epoch*_length1 subdirs, iterates over each one
#   in numerical order and sweeps its dora_params/.
# ---------------------------------------------------------------------------
sweep_perturbation_root() {
  local perturb_root="$1"

  # Collect epoch*_length1 dirs, sorted numerically by epoch number.
  mapfile -t EPOCH_DIRS < <(
    for d in "${perturb_root}"/epoch*_length1; do
      [[ -d "$d" ]] || continue
      base="$(basename "$d")"
      num="${base#epoch}"
      num="${num%_length1}"
      printf '%020d\t%s\n' "$num" "$d"
    done | sort -n | cut -f2
  )

  if [[ ${#EPOCH_DIRS[@]} -eq 0 ]]; then
    echo "[WARN] No epoch*_length1 dirs found under ${perturb_root}, skipping."
    return 0
  fi

  echo "=== Perturbation root: ${perturb_root} (${#EPOCH_DIRS[@]} epoch dirs) ==="

  for epoch_dir in "${EPOCH_DIRS[@]}"; do
    local epoch_name
    epoch_name="$(basename "$epoch_dir")"   # e.g. epoch14_length1

    # Find the single vit_l_14* training-run subdir inside this epoch dir.
    local training_run_dir
    training_run_dir="$(find "$epoch_dir" -maxdepth 1 -type d -name 'vit_l_14*' | head -1)"
    if [[ -z "$training_run_dir" ]]; then
      echo "  [SKIP] No vit_l_14* subdir in ${epoch_dir}"
      continue
    fi

    local ckpt_dir="${training_run_dir}/dora_params"
    local run_name
    run_name="$(basename "$training_run_dir")"
    # Save outputs under <perturb_root>/<epoch_name>/
    local save_dir="${perturb_root}/${epoch_name}"

    echo "--- ${epoch_name} ---"
    run_sweep "$ckpt_dir" "$save_dir" "$run_name"
  done
}

# ---------------------------------------------------------------------------
# Define which runs to process.
# ---------------------------------------------------------------------------

# Baseline runs (flat structure: no epoch*_length1 nesting)
BASELINE_ROOTS=(
  "${BASE}/baseline_seed1"
)

# Perturbation runs (contain epoch*_length1 subdirs)
PERTURB_ROOTS=(
  "${BASE}/random_target_perturb_seed1"
  "${BASE}/random_target_perturb_seed2"
  "${BASE}/random_target_perturb_seed3"
  "${BASE}/image_noise_perturb_seed1"
  "${BASE}/image_noise_perturb_seed2"
  "${BASE}/image_noise_perturb_seed3"
  "${BASE}/label_shuffle_perturb_seed1"
  "${BASE}/label_shuffle_perturb_seed2"
  "${BASE}/label_shuffle_perturb_seed3"
  "${BASE}/uniform_images_perturb_seed1"
  "${BASE}/uniform_images_perturb_seed2"
  "${BASE}/uniform_images_perturb_seed3"
)

# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------
# for root in "${BASELINE_ROOTS[@]}"; do
#   echo "=== Baseline: ${root} ==="
#   sweep_run_root "$root"
# done

for root in "${PERTURB_ROOTS[@]}"; do
  sweep_perturbation_root "$root"
done
