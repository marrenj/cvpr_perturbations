#!/usr/bin/env bash
set -eu
set -o pipefail

export SAVE_ROOT="/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/test/perturb_sweep_replication"
export BASE_CONFIG="configs/training_config.yaml"
export BASELINE_CHECKPOINT_PATH="/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/test/perturb_sweep_replication/baseline_seed2"
export IMG_ANNOTATIONS_FILE="/home/wallacelab/Documents/GitHub/cvpr_perturbations/data/spose_embedding66d_rescaled_1806train.csv"
export IMG_DIR="/home/wallacelab/investigating-complexity/Images/THINGS"
export CUDA=0
export PERTURB_SEED=43
export RANDOM_SEED=2
export PYTHON_CMD="${PYTHON_CMD:-python3}"

# REQUIRED: set these before running
: "${SAVE_ROOT:?Set SAVE_ROOT to the base output directory for runs}"
: "${BASE_CONFIG:=configs/training_config.yaml}"
: "${BASELINE_CHECKPOINT_PATH:?Set BASELINE_CHECKPOINT_PATH to your baseline run root}"
: "${IMG_ANNOTATIONS_FILE:?Set IMG_ANNOTATIONS_FILE to your annotations CSV}"
: "${IMG_DIR:?Set IMG_DIR to your images directory}"
: "${CUDA:=0}"
: "${PERTURB_SEED:=43}"
: "${RANDOM_SEED:=2}"
: "${PYTHON_CMD:=python3}"

TMP_DIR="$(mktemp -d)"
cleanup() { rm -rf "$TMP_DIR"; }
trap cleanup EXIT

for EPOCH in $(seq 1 91); do
  TMP_CFG="${TMP_DIR}/epoch${EPOCH}.yaml"
  CUDA="$CUDA" EPOCH="$EPOCH" PERTURB_SEED="$PERTURB_SEED" RANDOM_SEED="$RANDOM_SEED" TMP_CFG="$TMP_CFG" "$PYTHON_CMD" - <<'PY'
import os, yaml, copy
base_cfg = yaml.safe_load(open(os.environ["BASE_CONFIG"]))
cfg = copy.deepcopy(base_cfg)

cfg["save_path"] = os.environ["SAVE_ROOT"]
cfg["baseline_checkpoint_path"] = os.environ["BASELINE_CHECKPOINT_PATH"]
cfg["img_annotations_file"] = os.environ["IMG_ANNOTATIONS_FILE"]
cfg["img_dir"] = os.environ["IMG_DIR"]

cfg["perturb_type"] = "random_target"
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
