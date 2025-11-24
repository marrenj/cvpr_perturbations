import argparse
from pathlib import Path
import yaml

from src.training.train import run_training

def parse_args():
    """Return parsed CLI arguments for selecting a YAML config file."""
    parser = argparse.ArgumentParser(description="Run training with an external config.")
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to YAML config file (e.g., configs/run.yaml)")
    return parser.parse_args()


def load_yaml_config(path: Path):
    """Load and parse the YAML file located at ``path``."""
    with path.open("r") as f:
        return yaml.safe_load(f)


def main():
    """Entrypoint: load config specified via CLI and launch training."""
    args = parse_args()
    config = load_yaml_config(args.config)
    run_training(config)

if __name__ == '__main__':
    main()