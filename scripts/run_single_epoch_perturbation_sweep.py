from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
import yaml

# Ensure repository root is on sys.path when the script is invoked via an absolute path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.trainer import run_training_experiment
from src.utils.load_yaml_config import load_yaml_config
from src.utils.save_config import save_config

# Keys that should be coerced to numeric values when loading the YAML config
TRAINING_INT_KEYS = [
    "epochs",
    "batch_size",
    "rank",
    "vision_layers",
    "transformer_layers",
    "early_stopping_patience",
    "random_seed",
    "cuda",
]
TRAINING_FLOAT_ONLY = ["lr", "train_portion"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a single-epoch perturbation sweep. For each epoch in the sweep "
            "range, launch a training run where the perturbation is active for "
            "exactly that epoch."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Base YAML config to use for all runs.",
    )
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=None,
        help="First epoch (0-indexed) to perturb.",
    )
    parser.add_argument(
        "--end_epoch",
        type=int,
        default=None,
        help=(
            "Epoch after the last perturbed epoch (exclusive). Defaults to the "
            "total number of epochs in the base config."
        ),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help=(
            "Root directory where per-epoch run folders will be created. "
            "Defaults to the checkpoint_path in the base config."
        ),
    )
    parser.add_argument(
        "--run_name_template",
        type=str,
        default="training_run{epoch}",
        help="Template for run subfolders. Must include '{epoch}'.",
    )
    parser.add_argument(
        "--perturb_seed",
        type=int,
        default=None,
        help="Override perturb_seed (defaults to the value in the base config).",
    )
    parser.add_argument(
        "--baseline_checkpoint_path",
        type=Path,
        default=None,
        help=(
            "Path to a baseline run that contains checkpoints and random states. "
            "Required when sweeping epochs beyond 0."
        ),
    )
    return parser.parse_args()


def _serialize_run_config(run_config: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as handle:
        yaml.safe_dump(run_config, handle)


def main() -> None:
    args = _parse_args()

    if "{epoch}" not in args.run_name_template:
        raise ValueError("run_name_template must include the '{epoch}' placeholder")

    base_config = load_yaml_config(
        args.config,
        numeric_keys=TRAINING_INT_KEYS,
        float_only=TRAINING_FLOAT_ONLY,
    )

    if "epochs" not in base_config:
        raise ValueError("Base config is missing required key: 'epochs'")

    total_epochs = int(base_config["epochs"])
    start_epoch = (
        int(args.start_epoch)
        if args.start_epoch is not None
        else int(base_config.get("start_epoch", 0))
    )
    end_epoch = (
        int(args.end_epoch)
        if args.end_epoch is not None
        else int(base_config.get("end_epoch", total_epochs))
    )

    if start_epoch < 0:
        raise ValueError("start_epoch must be non-negative")
    if end_epoch > total_epochs:
        raise ValueError("end_epoch cannot exceed the total epochs in the base config")
    if end_epoch <= start_epoch:
        raise ValueError("end_epoch must be greater than start_epoch")

    baseline_checkpoint = (
        str(args.baseline_checkpoint_path)
        if args.baseline_checkpoint_path
        else base_config.get("baseline_checkpoint_path")
    )
    if start_epoch > 0 and not baseline_checkpoint:
        raise ValueError(
            "baseline_checkpoint_path is required when perturbing epochs beyond 0"
        )

    perturb_seed = (
        int(args.perturb_seed)
        if args.perturb_seed is not None
        else base_config.get("perturb_seed")
    )
    if perturb_seed is None:
        raise ValueError("Provide perturb_seed via --perturb_seed or the base config")

    output_root = (
        Path(args.output_root)
        if args.output_root is not None
        else Path(base_config["checkpoint_path"])
    )
    output_root.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, end_epoch):
        run_dir = output_root / args.run_name_template.format(epoch=epoch)
        run_dir.mkdir(parents=True, exist_ok=True)

        run_config = deepcopy(base_config)
        run_config["perturb_epoch"] = epoch
        run_config["perturb_length"] = 1
        run_config["perturb_seed"] = perturb_seed
        run_config["checkpoint_path"] = str(run_dir)
        if baseline_checkpoint:
            run_config["baseline_checkpoint_path"] = baseline_checkpoint

        # Save both the original config (for provenance) and the resolved config
        save_config(args.config, run_dir)
        _serialize_run_config(run_config, run_dir / "resolved_config.yaml")

        run_training_experiment(run_config)


if __name__ == "__main__":
    main()

