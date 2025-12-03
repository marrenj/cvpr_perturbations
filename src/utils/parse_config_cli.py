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
    return parser.parse_args()