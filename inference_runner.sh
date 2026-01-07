#!/usr/bin/env bash
set -euo pipefail

BASE_CONFIG="configs/inference_config.yaml"
PTH_DIR="/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/baseline_runs/clip_hba_behavior_seed1/training_artifacts/dora_params/dora_params_seed1"
BASE_INFERENCE_SAVE_DIR="/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/baseline_runs/clip_hba_behavior_seed1/test_things_neural_inference"
CUDA_DEVICE=1

for ckpt in $(find "${PTH_DIR}" -maxdepth 1 -type f -name 'epoch*_dora_params.pth' | sort -V); do
  [ -e "$ckpt" ] || { echo "No checkpoints found in $PTH_DIR"; exit 1; }
  ckpt_base="$(basename "$ckpt")"
  run_dir="$(python - "$BASE_CONFIG" "$ckpt" "$BASE_INFERENCE_SAVE_DIR" <<'PY'
import os, sys, yaml
base_conf = yaml.safe_load(open(sys.argv[1], "r"))
ckpt_path = sys.argv[2]
base_inference_save_dir = sys.argv[3].rstrip("/")
ckpt_base = os.path.basename(ckpt_path)
epoch = ckpt_base.split("_")[0]
print(os.path.join(base_inference_save_dir, epoch))
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