#!/usr/bin/env bash
set -eu
set -o pipefail

export SAVE_ROOT="/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/test/perturb_length_replication"
export BASE_CONFIG="configs/training_config.yaml"
export BASELINE_CHECKPOINT_PATH="/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/test/perturb_sweep_replication"
export IMG_ANNOTATIONS_FILE="/home/wallacelab/Documents/GitHub/cvpr_perturbations/data/spose_embedding66d_rescaled_1806train.csv"
export IMG_DIR="/home/wallacelab/investigating-complexity/Images/THINGS"
export CUDA=0
export PERTURB_TYPE="random_target"
export PERTURB_SEED=42
export RANDOM_SEED=1
export PYTHON_CMD="${PYTHON_CMD:-python3}"

# REQUIRED: set these before running
: "${SAVE_ROOT:?Set SAVE_ROOT to the base output directory for runs}"
: "${BASE_CONFIG:=configs/training_config.yaml}"
: "${BASELINE_CHECKPOINT_PATH:?Set BASELINE_CHECKPOINT_PATH to your baseline run root}"
: "${IMG_ANNOTATIONS_FILE:?Set IMG_ANNOTATIONS_FILE to your annotations CSV}"
: "${IMG_DIR:?Set IMG_DIR to your images directory}"
: "${CUDA:=0}"
: "${PERTURB_TYPE:?Set PERTURB_TYPE to the perturbation type}"
: "${PERTURB_SEED:?Set PERTURB_SEED to the perturbation seed}"
: "${RANDOM_SEED:?Set RANDOM_SEED to the random seed}"
: "${PYTHON_CMD:?Set PYTHON_CMD to the Python command}"

START_EPOCHS=(0 1 2 3 4 5 6 7 8 9 19 29 39 49 59 69 79 89)
PERTURB_LENGTHS=(2 5 10 20 30 40 50)

TMP_DIR="$(mktemp -d)"
cleanup() { rm -rf "$TMP_DIR"; }
trap cleanup EXIT

for EPOCH in "${START_EPOCHS[@]}"; do
  for PERTURB_LEN in "${PERTURB_LENGTHS[@]}"; do
    RUN_SAVE_PATH="${SAVE_ROOT}/${PERTURB_TYPE}_epoch${EPOCH}_len${PERTURB_LEN}"
    TMP_CFG="${TMP_DIR}/${PERTURB_TYPE}_epoch${EPOCH}_len${PERTURB_LEN}.yaml"
    CUDA="$CUDA" PERTURB_TYPE="$PERTURB_TYPE" EPOCH="$EPOCH" PERTURB_LEN="$PERTURB_LEN" PERTURB_SEED="$PERTURB_SEED" RANDOM_SEED="$RANDOM_SEED" RUN_SAVE_PATH="$RUN_SAVE_PATH" TMP_CFG="$TMP_CFG" "$PYTHON_CMD" - <<'PY'
import os, yaml, copy
base_cfg = yaml.safe_load(open(os.environ["BASE_CONFIG"]))
cfg = copy.deepcopy(base_cfg)

cfg["save_path"] = os.environ["RUN_SAVE_PATH"]
cfg["baseline_checkpoint_path"] = os.environ["BASELINE_CHECKPOINT_PATH"]
cfg["img_annotations_file"] = os.environ["IMG_ANNOTATIONS_FILE"]
cfg["img_dir"] = os.environ["IMG_DIR"]

cfg["perturb_type"] = os.environ["PERTURB_TYPE"]
cfg["perturb_seed"] = int(os.environ["PERTURB_SEED"])
cfg["perturb_epoch"] = int(os.environ["EPOCH"])
cfg["perturb_length"] = int(os.environ["PERTURB_LEN"])
cfg["random_seed"] = int(os.environ["RANDOM_SEED"])
cfg["cuda"] = int(os.environ["CUDA"])

with open(os.environ["TMP_CFG"], "w") as f:
    yaml.safe_dump(cfg, f)
PY
    CUDA="$CUDA" "$PYTHON_CMD" scripts/run_training.py --config "$TMP_CFG"
  done
done

