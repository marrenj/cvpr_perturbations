#!/usr/bin/env bash
set -euo pipefail

# REQUIRED: set these before running
: "${SAVE_ROOT:?Set SAVE_ROOT to the base output directory for runs}"
: "${BASE_CONFIG:=configs/training_config.yaml}"
: "${BASELINE_CHECKPOINT_PATH:?Set BASELINE_CHECKPOINT_PATH to your baseline run root}"
: "${IMG_ANNOTATIONS_FILE:?Set IMG_ANNOTATIONS_FILE to your annotations CSV}"
: "${IMG_DIR:?Set IMG_DIR to your images directory}"
# CUDA devices for the two parallel sweeps
: "${CUDA_SEED43:=0}"
: "${CUDA_SEED44:=1}"

# Temp dir for generated configs
TMP_DIR="$(mktemp -d)"
cleanup() { rm -rf "$TMP_DIR"; }
trap cleanup EXIT

run_sweep() {
  local perturb_seed="$1"
  local random_seed="$2"
  local cuda_dev="$3"
  local label="$4"

  local epochs_43=(0 1 2 3 4 5 6 7 8 9 19 29 39 49 59 69 79 89)
  local epochs_44=(0 1 2 3 4 5 6 7 8 9 19 29 39 49 59 69 79 89 99 107)
  local epochs=("${epochs_43[@]}")
  if [[ "$perturb_seed" -eq 44 ]]; then
    epochs=("${epochs_44[@]}")
  fi

  for EPOCH in "${epochs[@]}"; do
    for LENGTH in 2 5 10 20 30 40 50; do
      TMP_CFG="${TMP_DIR}/${label}_epoch${EPOCH}_length${LENGTH}.yaml"
      CUDA="$cuda_dev" EPOCH="$EPOCH" LENGTH="$LENGTH" perturb_seed="$perturb_seed" random_seed="$random_seed" TMP_CFG="$TMP_CFG" python - <<'PY'
import os, yaml, copy
base_cfg = yaml.safe_load(open(os.environ["BASE_CONFIG"]))
cfg = copy.deepcopy(base_cfg)

cfg["save_path"] = os.environ["SAVE_ROOT"]
cfg["baseline_checkpoint_path"] = os.environ["BASELINE_CHECKPOINT_PATH"]
cfg["img_annotations_file"] = os.environ["IMG_ANNOTATIONS_FILE"]
cfg["img_dir"] = os.environ["IMG_DIR"]

cfg["perturb_type"] = "random_target"
cfg["perturb_seed"] = int(os.environ["perturb_seed"])
cfg["perturb_epoch"] = int(os.environ["EPOCH"])
cfg["perturb_length"] = int(os.environ["LENGTH"])
cfg["random_seed"] = int(os.environ["random_seed"])
cfg["cuda"] = int(os.environ["CUDA"])

with open(os.environ["TMP_CFG"], "w") as f:
    yaml.safe_dump(cfg, f)
PY
      CUDA="$cuda_dev" python scripts/run_training.py --config "$TMP_CFG" &
    done
  done
}

# Launch two sweeps in parallel on different GPUs
run_sweep 43 1 "$CUDA_SEED43" "seed43" &
run_sweep 44 3 "$CUDA_SEED44" "seed44" &

wait