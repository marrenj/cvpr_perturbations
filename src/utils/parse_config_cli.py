import argparse
from pathlib import Path

def parse_config_cli(description: str) -> argparse.Namespace:
    """
    Build the standard CLI for scripts that take a single --config YAML file.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config file (e.g., configs/run.yaml)",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        default=False,
        help="Disable Weights & Biases logging entirely.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Override random_seed from the config (e.g. for multi-seed parallel runs).",
    )
    return parser.parse_args()