#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root (directory containing this script's parent)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}" && pwd)"

# Allow overriding via environment; fall back to absolute defaults.
BASE_CONFIG="${BASE_CONFIG:-/home/wallacelab/Documents/GitHub/cvpr_perturbations/configs/inference_config.yaml}"
PTH_ROOT="${PTH_ROOT:-/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/test/perturb_sweep_replication/random_target_perturb_seed42}"
BASE_INFERENCE_SAVE_DIR="${BASE_INFERENCE_SAVE_DIR:-/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/test/perturb_sweep_replication/random_target_perturb_seed42}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

find "${PTH_ROOT}" -maxdepth 2 -type d -name 'training_run*' | sort -V | while read -r training_dir; do
  run_name="$(basename "$training_dir")"           # e.g., training_run42
  run_num="${run_name#training_run}"               # e.g., 42
  params_dir="$(find "${training_dir}" -maxdepth 1 -type d -name "dora_params_run${run_num}" | head -n1)"
  if [[ -z "$params_dir" ]]; then
    echo "No dora_params_run${run_num} found under $training_dir, skipping"
    continue
  fi
  ckpt="$(find "${params_dir}" -maxdepth 1 -type f -name "epoch${run_num}_dora_params.pth" | head -n1)"
  if [[ -z "$ckpt" ]]; then
    echo "No epoch${run_num}_dora_params.pth found under $params_dir, skipping"
    continue
  fi

  ckpt_base="$(basename "$ckpt")"
  parent_run="$(basename "$params_dir")"    # e.g., dora_params_run42
  grandparent="$run_name"                   # e.g., training_run42

  run_dir="$(python - "$BASE_CONFIG" "$ckpt" "$BASE_INFERENCE_SAVE_DIR" "$grandparent" "$parent_run" <<'PY'
import os, sys, yaml
base_conf = yaml.safe_load(open(sys.argv[1], "r"))
ckpt_path = sys.argv[2]
base_inference_save_dir = sys.argv[3].rstrip("/")
training_run = sys.argv[4]
parent_run = sys.argv[5]
ckpt_base = os.path.basename(ckpt_path)
epoch = ckpt_base.split("_")[0]
print(os.path.join(base_inference_save_dir, training_run, "test_things_neural_inference", epoch))
PY
)"

  tmp_conf="$(mktemp)"
  python - "$BASE_CONFIG" "$ckpt" "$run_dir" "$tmp_conf" "$CUDA_DEVICE" <<'PY'
import sys, yaml
base_conf = yaml.safe_load(open(sys.argv[1], "r"))
ckpt_path = sys.argv[2]
run_dir = sys.argv[3]
tmp_conf = sys.argv[4]
cuda_device = int(sys.argv[5])

base_conf["model_weights_path"] = ckpt_path
base_conf["inference_save_dir"] = run_dir
base_conf["cuda"] = cuda_device

with open(tmp_conf, "w") as f:
    yaml.safe_dump(base_conf, f)
PY
  python scripts/run_inference.py --config "$tmp_conf"
  rm -f "$tmp_conf"
done