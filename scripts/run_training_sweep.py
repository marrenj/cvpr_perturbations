from __future__ import annotations

import argparse
from pathlib import Path
import sys
import copy

# Ensure repository root is on sys.path when invoked via an absolute path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

import scripts.run_training as rt
from src.training.trainer import run_training_experiment
from src.utils.load_yaml_config import load_yaml_config

# Reuse the numeric coercion rules from run_training.py
TRAINING_INT_KEYS = [
    "epochs",
    "batch_size",
    "rank",
    "vision_layers",
    "transformer_layers",
    "early_stopping_patience",
    "random_seed",
    "cuda",
    "start_epoch",
    "end_epoch",
]
TRAINING_FLOAT_ONLY = ["lr", "train_portion"]


def _parse_csv_ints(text: str) -> list[int]:
    return [int(t.strip()) for t in text.split(",") if t.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-platform sweep driver that generates per-run configs and calls run_training."
    )
    parser.add_argument("--base-config", type=Path, required=True, help="Path to base YAML config.")
    parser.add_argument("--save-root", type=Path, required=True, help="Root directory for all runs.")
    parser.add_argument(
        "--baseline-checkpoint-path",
        type=Path,
        required=True,
        help="Baseline checkpoint path used when perturb_epoch > 0.",
    )
    parser.add_argument("--img-annotations-file", type=Path, required=True, help="Annotations CSV.")
    parser.add_argument("--img-dir", type=Path, required=True, help="Images directory.")
    parser.add_argument("--perturb-type", default="random_target", help="Perturbation type.")
    parser.add_argument(
        "--perturb-seeds",
        type=str,
        default="42",
        help="Comma-separated perturbation seeds (e.g., '42,44').",
    )
    parser.add_argument(
        "--random-seeds",
        type=str,
        default="1",
        help="Comma-separated random seeds aligned with perturb-seeds (e.g., '1,3').",
    )
    parser.add_argument(
        "--epochs",
        type=str,
        default="24-97",
        help="Epochs list as comma-separated ints or a range 'start-end' (inclusive).",
    )
    parser.add_argument(
        "--lengths",
        type=str,
        default="1",
        help="Comma-separated perturbation lengths (ints).",
    )
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device index (int).")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without executing.")
    return parser.parse_args()


def _expand_epochs(expr: str) -> list[int]:
    expr = expr.strip()
    if "-" in expr and "," not in expr:
        start, end = expr.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return _parse_csv_ints(expr)


def main() -> None:
    args = parse_args()

    perturb_seeds = _parse_csv_ints(args.perturb_seeds)
    random_seeds = _parse_csv_ints(args.random_seeds)
    if len(random_seeds) not in (1, len(perturb_seeds)):
        raise ValueError("random-seeds must be length 1 or match perturb-seeds length")
    if len(random_seeds) == 1:
        random_seeds = random_seeds * len(perturb_seeds)

    epochs = _expand_epochs(args.epochs)
    lengths = _parse_csv_ints(args.lengths)

    base_config = load_yaml_config(
        args.base_config,
        numeric_keys=TRAINING_INT_KEYS,
        float_only=TRAINING_FLOAT_ONLY,
    )

    planned = []
    for seed_idx, perturb_seed in enumerate(perturb_seeds):
        rseed = random_seeds[seed_idx]
        for epoch in epochs:
            for length in lengths:
                planned.append((perturb_seed, rseed, epoch, length))

    if args.dry_run:
        print("Planned runs:")
        for pseed, rseed, epoch, length in planned:
            print(f"  perturb_seed={pseed}, random_seed={rseed}, epoch={epoch}, length={length}")
        return

    for perturb_seed, rseed, epoch, length in planned:
        cfg = copy.deepcopy(base_config)
        cfg["save_path"] = str(args.save_root)
        cfg["baseline_checkpoint_path"] = str(args.baseline_checkpoint_path)
        cfg["img_annotations_file"] = str(args.img_annotations_file)
        cfg["img_dir"] = str(args.img_dir)

        cfg["perturb_type"] = args.perturb_type
        cfg["perturb_seed"] = int(perturb_seed)
        cfg["perturb_epoch"] = int(epoch)
        cfg["perturb_length"] = int(length)
        cfg["random_seed"] = int(rseed)
        cfg["cuda"] = int(args.cuda)

        prepared = rt._prepare_single_run(cfg, args.base_config)
        run_training_experiment(prepared)


if __name__ == "__main__":
    main()