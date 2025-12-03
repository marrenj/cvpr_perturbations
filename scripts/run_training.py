from pathlib import Path

from src.training.trainer import run_training_experiment
from src.utils.parse_config_cli import parse_config_cli
from src.utils.load_yaml_config import load_yaml_config
from src.utils.save_config import save_config

TRAINING_INT_KEYS = ["epochs", "batch_size", "rank",
                     "vision_layers", "transformer_layers",
                     "early_stopping_patience",
                     "random_seed", "cuda"]
TRAINING_FLOAT_ONLY = ["lr", "train_portion"]


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