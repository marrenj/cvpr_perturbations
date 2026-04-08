from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import shared infrastructure from the single-run script
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_training import _prepare_single_run, TRAINING_INT_KEYS, TRAINING_FLOAT_ONLY
from src.training.trainer import run_training_experiment
from src.utils.load_yaml_config import load_yaml_config

# ---------------------------------------------------------------------------
# Sweep definition
# ---------------------------------------------------------------------------

PERTURB_TYPES = ["label_shuffle", "image_noise", "uniform_images"]

SWEEP_EPOCHS = (
    list(range(0, 9, 2))           # 0, 2, 4, 6, 8  (every 2)
    + list(range(9, 39, 5))  # 9, 14, 19, 24, 29, 34  (every 5)
    + list(range(39, 100, 10))  # 39, 49, 59, 69, 79, 89, 99  (every 10)
)

# Outer loop: perturbation type; inner loop: epoch.
# Index i → ALL_TASKS[i] = (perturb_type, perturb_epoch)
ALL_TASKS: list[tuple[str, int]] = [
    (pt, ep) for pt in PERTURB_TYPES for ep in SWEEP_EPOCHS
]  # 3 × 18 = 54 entries


def _list_tasks() -> None:
    print(f"{'Task ID':>7}  {'Perturb Type':<20}  {'Perturb Epoch':>13}")
    print("-" * 46)
    for i, (pt, ep) in enumerate(ALL_TASKS):
        print(f"{i:>7}  {pt:<20}  {ep:>13}")
    print(f"\nTotal tasks: {len(ALL_TASKS)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a single task from the ResNet50 ImageNet perturbation sweep. "
            "Use --list-tasks to see all combinations (perturb_type, perturb_epoch)."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to base YAML config (e.g. configs/resnet50_imagenet.yaml)",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="0-indexed task ID. Required unless --list-tasks is set.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Override save_path in config.",
    )
    parser.add_argument(
        "--img-dir",
        type=str,
        default=None,
        help="Override img_dir in config.",
    )
    parser.add_argument(
        "--baseline-checkpoint-path",
        type=str,
        default=None,
        help=(
            "Path to a completed baseline run directory. Required for any task "
            "where perturb_epoch > 0."
        ),
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print all (task_id, perturb_type, perturb_epoch) combinations and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.list_tasks:
        _list_tasks()
        return

    if args.task_id is None:
        raise SystemExit("--task-id is required when not using --list-tasks")

    if not (0 <= args.task_id < len(ALL_TASKS)):
        raise SystemExit(
            f"--task-id must be between 0 and {len(ALL_TASKS) - 1}, "
            f"got {args.task_id}"
        )

    perturb_type, perturb_epoch = ALL_TASKS[args.task_id]

    print(f"Task {args.task_id}: perturb_type={perturb_type}, perturb_epoch={perturb_epoch}")

    config_path = args.config.resolve()
    config = load_yaml_config(
        config_path,
        numeric_keys=TRAINING_INT_KEYS,
        float_only=TRAINING_FLOAT_ONLY,
    )

    # Apply path overrides
    if args.save_path:
        config["save_path"] = args.save_path
    if args.img_dir:
        config["img_dir"] = args.img_dir

    # Apply sweep parameters
    config["perturb_type"] = perturb_type
    config["perturb_epoch"] = perturb_epoch
    config["perturb_length"] = 1

    # baseline_checkpoint_path is required by the trainer when perturb_epoch > 0
    if perturb_epoch > 0:
        if not args.baseline_checkpoint_path:
            raise SystemExit(
                f"--baseline-checkpoint-path is required for perturb_epoch={perturb_epoch} > 0"
            )
        config["baseline_checkpoint_path"] = args.baseline_checkpoint_path

    prepared_config = _prepare_single_run(config, config_path)
    run_training_experiment(prepared_config)


if __name__ == "__main__":
    main()
