#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root (directory containing this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}" && pwd)"

# Allow overriding via environment; fall back to repo-relative defaults.
BASE_CONFIG="${BASE_CONFIG:-${REPO_ROOT}/configs/inference_config.yaml}"

# Root where run directories live (override via env)
PTH_ROOT="${PTH_ROOT:-/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/test/perturb_sweep_replication/random_target_perturb_seed42}"

# Where to save inference outputs (override via env)
BASE_INFERENCE_SAVE_DIR="${BASE_INFERENCE_SAVE_DIR:-/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/test/perturb_sweep_replication/random_target_perturb_seed42}"

CUDA_DEVICE="${CUDA_DEVICE:-0}"

# Derive dataset and evaluation_type from the base config to name the output subdir
INFERENCE_DATASET="$(python3 - "$BASE_CONFIG" <<'PY'
import sys, yaml
conf = yaml.safe_load(open(sys.argv[1], "r"))
print(conf.get("dataset", "things"))
PY
)"
INFERENCE_TYPE="$(python3 - "$BASE_CONFIG" <<'PY'
import sys, yaml
conf = yaml.safe_load(open(sys.argv[1], "r"))
print(conf.get("evaluation_type", "neural"))
PY
)"
INFERENCE_SUBDIR="test_${INFERENCE_DATASET}_${INFERENCE_TYPE}_inference"

# Iterate over run directories matching epoch*_length*
find "${PTH_ROOT}" -maxdepth 1 -type d -name 'epoch*_length*' | sort -V | while read -r run_dir_path; do
  run_name="$(basename "$run_dir_path")"  # e.g., epoch9_length1
  # Extract the start epoch (number after 'epoch' before '_length')
  if [[ "$run_name" =~ ^epoch([0-9]+)_length[0-9]+$ ]]; then
    run_num="${BASH_REMATCH[1]}"
  else
    echo "Could not parse run name $run_name, skipping"
    continue
  fi

  ckpt="${run_dir_path}/epoch${run_num}_dora_params.pth"
  if [[ ! -f "$ckpt" ]]; then
    echo "No epoch${run_num}_dora_params.pth found under $run_dir_path, skipping"
    continue
  fi

# Build inference save dir: <BASE_INFERENCE_SAVE_DIR>/<run_name>/<test_dataset_eval_inference>/epoch<run_num>
  inf_dir="$(INFERENCE_SUBDIR="$INFERENCE_SUBDIR" python3 - "$BASE_INFERENCE_SAVE_DIR" "$run_name" "$run_num" <<'PY'
import os, sys
base = sys.argv[1].rstrip("/")
run_name = sys.argv[2]
epoch = sys.argv[3]
subdir = os.environ.get("INFERENCE_SUBDIR", "test_things_neural_inference")
print(os.path.join(base, run_name, subdir, f"epoch{epoch}"))
PY
)"

  tmp_conf="$(mktemp)"
  python3 - "$BASE_CONFIG" "$ckpt" "$inf_dir" "$tmp_conf" "$CUDA_DEVICE" <<'PY'
import sys, yaml, os
base_conf = yaml.safe_load(open(sys.argv[1], "r"))
ckpt_path = sys.argv[2]
run_dir = sys.argv[3]
tmp_conf = sys.argv[4]
cuda_device = int(sys.argv[5])
os.makedirs(run_dir, exist_ok=True)
base_conf["model_weights_path"] = ckpt_path
base_conf["inference_save_dir"] = run_dir
base_conf["cuda"] = cuda_device
with open(tmp_conf, "w") as f:
    yaml.safe_dump(base_conf, f)
PY
  python3 scripts/run_inference.py --config "$tmp_conf"
  rm -f "$tmp_conf"
done