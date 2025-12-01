import argparse
from pathlib import Path
import shutil
import yaml

from src.training.trainer import run_training_experiment

def parse_args():
    """Return parsed CLI arguments for selecting a YAML config file."""
    parser = argparse.ArgumentParser(description="Run training with an external config.")
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to YAML config file (e.g., configs/run.yaml)")
    return parser.parse_args()


def load_yaml_config(path: Path):
    """Load and parse the YAML file located at ``path``."""
    with path.open("r") as f:
        config = yaml.safe_load(f)

    # Convert numeric string values to proper types
    numeric_keys = ['lr', 'epochs', 'batch_size', 'train_portion', 'rank', 
                    'vision_layers', 'transformer_layers', 'early_stopping_patience',
                    'random_seed', 'cuda']
    
    for key in numeric_keys:
        if key in config and isinstance(config[key], str):
            try:
                # Try to convert to float first (handles scientific notation like 3e-4)
                config[key] = float(config[key])
                # Convert to int if it's a whole number
                if key in ['epochs', 'batch_size', 'rank', 'vision_layers', 
                           'transformer_layers', 'early_stopping_patience', 
                           'random_seed', 'cuda']:
                    config[key] = int(config[key])
            except (ValueError, TypeError):
                pass  # Keep original value if conversion fails

    return config


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
    args = parse_args()
    config = load_yaml_config(args.config)
    save_config(args.config, config.get('checkpoint_path'))
    run_training_experiment(config)

if __name__ == '__main__':
    main()