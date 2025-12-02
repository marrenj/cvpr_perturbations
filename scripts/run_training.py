import argparse
from pathlib import Path
import shutil
import yaml

from src.training.trainer import run_training_experiment
from src.utils.parse_args import parse_config_cli
from src.utils.load_yaml_config import load_yaml_config

TRAINING_INT_KEYS = ["epochs", "batch_size", "rank",
                     "vision_layers", "transformer_layers",
                     "early_stopping_patience",
                     "random_seed", "cuda"]
TRAINING_FLOAT_ONLY = ["lr", "train_portion"]


def save_config(src: Path, checkpoint_path: str):
    """Copy the YAML config into the checkpoint directory for provenance."""
    if not checkpoint_path:
        return

    checkpoint_dir = Path(checkpoint_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    destination = checkpoint_dir / src.name

    if src.resolve() == destination.resolve():
        return

    shutil.copy2(src, destination)


def main():
    """Entrypoint: load config specified via CLI and launch training."""
    args = parse_config_cli("Run training with an external config.")
    config = load_yaml_config(
        Path(args.config),
        numeric_keys=TRAINING_INT_KEYS,
        float_only=TRAINING_FLOAT_ONLY,
    )
    save_config(Path(args.config), config.get('checkpoint_path'))
    run_training_experiment(config)

if __name__ == '__main__':
    main()