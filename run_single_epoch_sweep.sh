#!/usr/bin/env bash
set -eu
set -o pipefail

export SAVE_ROOT="/mnt/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/test/experiments/label_shuffle_perturb_seed43"
export BASE_CONFIG="configs/training_config.yaml"
export BASELINE_CHECKPOINT_PATH="/mnt/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/test/cliphba_baseline_seed2"
export IMG_ANNOTATIONS_FILE="/mnt/c/Users/BrainInspired/Documents/GitHub/cvpr_perturbations/data/spose_embedding66d_rescaled_1806train.csv"
export IMG_DIR="/mnt/teba/multimodal_brain_inspired/marren/cvpr_perturbations/clip_hba_behavior/data/Things1854"
export CUDA=1
export PERTURB_SEED=43
export RANDOM_SEED=2

# REQUIRED: set these before running
: "${SAVE_ROOT:?Set SAVE_ROOT to the base output directory for runs}"
: "${BASE_CONFIG:=configs/training_config.yaml}"
: "${BASELINE_CHECKPOINT_PATH:?Set BASELINE_CHECKPOINT_PATH to your baseline run root}"
: "${IMG_ANNOTATIONS_FILE:?Set IMG_ANNOTATIONS_FILE to your annotations CSV}"
: "${IMG_DIR:?Set IMG_DIR to your images directory}"
: "${CUDA:=0}"
: "${PERTURB_SEED:=42}"
: "${RANDOM_SEED:=1}"

TMP_DIR="$(mktemp -d)"
cleanup() { rm -rf "$TMP_DIR"; }
trap cleanup EXIT

# Detect Python command (prefer python3, fallback to python)
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo "Error: python or python3 not found in PATH" >&2
    exit 1
fi

for EPOCH in $(seq 53 97); do
  TMP_CFG="${TMP_DIR}/epoch${EPOCH}.yaml"
  CUDA="$CUDA" EPOCH="$EPOCH" PERTURB_SEED="$PERTURB_SEED" RANDOM_SEED="$RANDOM_SEED" TMP_CFG="$TMP_CFG" "$PYTHON_CMD" - <<'PY'
import os, yaml, copy
base_cfg = yaml.safe_load(open(os.environ["BASE_CONFIG"]))
cfg = copy.deepcopy(base_cfg)

cfg["save_path"] = os.environ["SAVE_ROOT"]
cfg["baseline_checkpoint_path"] = os.environ["BASELINE_CHECKPOINT_PATH"]
cfg["img_annotations_file"] = os.environ["IMG_ANNOTATIONS_FILE"]
cfg["img_dir"] = os.environ["IMG_DIR"]

cfg["perturb_type"] = "label_shuffle"
cfg["perturb_seed"] = int(os.environ["PERTURB_SEED"])
cfg["perturb_epoch"] = int(os.environ["EPOCH"])
cfg["perturb_length"] = 1
cfg["random_seed"] = int(os.environ["RANDOM_SEED"])
cfg["cuda"] = int(os.environ["CUDA"])

with open(os.environ["TMP_CFG"], "w") as f:
    yaml.safe_dump(cfg, f)
PY
  CUDA="$CUDA" "$PYTHON_CMD" scripts/run_training.py --config "$TMP_CFG"
done
