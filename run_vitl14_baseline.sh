#!/usr/bin/env bash
set -eu
set -o pipefail

# Simple helper to launch a single ViT-L/14 training run with overrides.
# Fill in the required paths below before running.

# Required paths (edit these)
export BASE_CONFIG="configs/training_config.yaml"
export SAVE_ROOT="/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/vit_l_14"                 # e.g., /path/to/output/runs
export IMG_ANNOTATIONS_FILE="./data/spose_embedding66d_rescaled_1806train.csv"      # e.g., /path/to/spose_embedding66d_rescaled_*.csv
export IMG_DIR="/home/wallacelab/investigating-complexity/Images/THINGS"                   # e.g., /path/to/THINGS/images

# Optional overrides
export CUDA=-1                       # -1 all GPUs, 0/1 specific GPU, 2 for CPU
export RANDOM_SEED=1
export PERTURB_TYPE="none"
export PERTURB_EPOCH=
export PERTURB_LEN=
export PERTURB_SEED=
export PYTHON_CMD="${PYTHON_CMD:-python3}"

# Sanity checks
: "${BASE_CONFIG:?Set BASE_CONFIG to the base config YAML}"
: "${SAVE_ROOT:?Set SAVE_ROOT to the output root directory}"
: "${IMG_ANNOTATIONS_FILE:?Set IMG_ANNOTATIONS_FILE to your annotations CSV}"
: "${IMG_DIR:?Set IMG_DIR to your images directory}"

TMP_CFG="$(mktemp)"
trap 'rm -f "$TMP_CFG"' EXIT

# Build a minimal config override with ViT-L/14 backbone
"$PYTHON_CMD" - <<'PY'
import os, yaml, copy
base_cfg = yaml.safe_load(open(os.environ["BASE_CONFIG"]))
cfg = copy.deepcopy(base_cfg)

cfg["architecture"] = "ViT-L/14"
cfg["save_path"] = os.environ["SAVE_ROOT"]
cfg["img_annotations_file"] = os.environ["IMG_ANNOTATIONS_FILE"]
cfg["img_dir"] = os.environ["IMG_DIR"]
cfg["cuda"] = int(os.environ["CUDA"])
cfg["random_seed"] = int(os.environ["RANDOM_SEED"])
cfg["perturb_type"] = os.environ["PERTURB_TYPE"]
cfg["perturb_epoch"] = int(os.environ["PERTURB_EPOCH"])
cfg["perturb_length"] = int(os.environ["PERTURB_LEN"])
cfg["perturb_seed"] = int(os.environ["PERTURB_SEED"])

with open(os.environ["TMP_CFG"], "w") as f:
    yaml.safe_dump(cfg, f)
PY

echo "Running ViT-L/14 training with config: $TMP_CFG"
"$PYTHON_CMD" scripts/run_training.py --config "$TMP_CFG"