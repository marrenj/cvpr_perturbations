from __future__ import annotations

from pathlib import Path
import sys
import shutil
import yaml

# Ensure repository root is on sys.path when the script is invoked via an absolute path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.trainer import run_training_experiment
from src.utils.parse_config_cli import parse_config_cli
from src.utils.load_yaml_config import load_yaml_config
from src.utils.save_config import save_config

# Keys that should be coerced to numeric values when loading the YAML config
TRAINING_INT_KEYS = [
    "epochs",
    "batch_size",
    "num_classes",
    "rank",
    "vision_layers",
    "transformer_layers",
    "early_stopping_patience",
    "random_seed",
    "cuda",
    "start_epoch",
    "end_epoch",
]
TRAINING_FLOAT_ONLY = ["lr", "train_portion", "weight_decay", "momentum"]


def _write_resolved_config(resolved_cfg: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as handle:
        yaml.safe_dump(resolved_cfg, handle)


def _find_previous_run_with_same_start(run_root: Path, start_epoch: int, current_length: int):
    """
    Look for sibling run directories that share the same start epoch and have
    a shorter perturbation length. Assumes run dir names contain tokens
    'epoch{start_epoch}' and 'length{length}'.
    """
    parent = run_root.parent
    if not parent.exists():
        return None, None

    candidates = []
    for entry in parent.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        if f"epoch{start_epoch}" not in name:
            continue
        # extract length from pattern length{num}
        length_val = int(name.split("length")[1])
        if length_val < current_length:
            candidates.append((length_val, entry))

    if not candidates:
        return None, None
    best_length, best_dir = max(candidates, key=lambda t: t[0])
    return best_dir, best_length


def _resume_from_previous(config: dict, run_root: Path) -> dict:
    """
    If a previous run with the same start epoch exists in the save_path
    parent directory (and has a shorter perturb_length), reuse it to resume.
    """
    perturb_epoch = int(config.get("perturb_epoch", 0))
    perturb_length = int(config.get("perturb_length", 0))

    # Only applicable for perturbation runs with length > 0
    if perturb_length <= 0:
        return config

    prev_dir, prev_length = _find_previous_run_with_same_start(
        run_root, start_epoch=perturb_epoch, current_length=perturb_length
    )
    if not prev_dir or prev_length is None:
        return config

    resume_from_epoch = perturb_epoch + prev_length - 1
    dora_file = (
        prev_dir 
        / "dora_params"
        / f"epoch{resume_from_epoch}_dora_params.pth"
    )
    random_state_file = (
        prev_dir 
        / "random_states" 
        / f"epoch{resume_from_epoch}_random_states.pth"
    )

    if random_state_file.exists() and dora_file.exists():
        config["resume_checkpoint_path"] = str(prev_dir)
        config["resume_from_epoch"] = resume_from_epoch

        prev_training_res = prev_dir / "training_res.csv"
        dest_training_res = run_root / "training_res.csv"
        if prev_training_res.exists():
            dest_training_res.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(prev_training_res, dest_training_res)
            config["previous_training_res_path"] = str(prev_training_res)
    return config


def _prepare_single_run(config: dict, config_path: Path) -> dict:
    """
    Validate a single training run (baseline or perturbation) and ensure paths are set.

    Works for both ``training_mode='scratch'`` and ``training_mode='finetune'``.
    """
    base_save_path = config.get("save_path")
    if not base_save_path:
        raise ValueError("Config must set 'save_path'.")

    training_mode = config.get("training_mode", "finetune")

    # Validate mode-specific required fields
    if training_mode == "scratch":
        if not config.get("img_dir"):
            raise ValueError("Config must set 'img_dir' for training_mode='scratch'.")
        if config.get("perturb_type") == "random_target":
            raise ValueError(
                "perturb_type='random_target' requires continuous embedding targets; "
                "use training_mode='finetune' for this perturbation type."
            )
    elif training_mode == "finetune":
        if not config.get("img_annotations_file"):
            raise ValueError(
                "Config must set 'img_annotations_file' for training_mode='finetune'."
            )

    perturb_type = str(config.get("perturb_type") or "none")
    random_seed = config.get("random_seed")
    perturb_seed = config.get("perturb_seed")
    # Default to zeroed perturb settings for baseline (none) runs
    perturb_epoch = int(config.get("perturb_epoch") or 0)
    perturb_length = int(config.get("perturb_length") or 0)
    perturb_seed = int(perturb_seed if perturb_seed is not None else (random_seed or 0))

    if perturb_type and perturb_type.lower() != "none":
        # Include epoch/length tokens for perturbation runs
        seed_suffix = f"perturb_seed{perturb_seed}"
        run_root = Path(base_save_path) / f"{perturb_type}_{seed_suffix}" / f"epoch{perturb_epoch}_length{perturb_length}"
    else:
        seed_suffix = f"baseline_seed{random_seed}"
        run_root = Path(base_save_path) / seed_suffix
    run_root.mkdir(parents=True, exist_ok=True)
    save_path = run_root
    if perturb_length < 0:
        raise ValueError("perturb_length must be non-negative")
    if perturb_epoch > 0 and not config.get("baseline_checkpoint_path"):
        raise ValueError("baseline_checkpoint_path is required when perturb_epoch > 0")

    config["perturb_epoch"] = perturb_epoch
    config["perturb_length"] = perturb_length
    config["save_path"] = str(save_path)

    # Optionally reuse a previous run with the same start epoch
    config = _resume_from_previous(config, save_path)

    # Save the input config alongside the run artifacts
    save_config(config_path, save_path)
    _write_resolved_config(config, save_path / "resolved_config.yaml")
    return config


def main():
    """Entrypoint: load config specified via CLI and launch a single run."""
    args = parse_config_cli("Run training with an external config.")
    config_path = Path(args.config)
    config = load_yaml_config(
        config_path,
        numeric_keys=TRAINING_INT_KEYS,
        float_only=TRAINING_FLOAT_ONLY,
    )

    # Regardless of experiment_type, we run a single training job; perturbation
    # behavior is controlled by perturb_epoch/perturb_length.
    prepared_config = _prepare_single_run(config, config_path)
    run_training_experiment(prepared_config)


if __name__ == "__main__":
    main()