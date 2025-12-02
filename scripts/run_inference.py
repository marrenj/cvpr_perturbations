import argparse
from pathlib import Path
import yaml
import shutil

from src.inference.inference_core import run_inference
from src.utils.parse_args import parse_config_cli
from src.utils.load_yaml_config import load_yaml_config
import argparse
import yaml


# Parse CLI for --config and load YAML
args = parse_config_cli("Run inference with an external config.")
config_path = Path(args.config)
config = yaml.safe_load(config_path.read_text())

INFERENCE_NUMERIC_KEYS = ["epochs", "batch_size", "rank",
                          "vision_layers", "transformer_layers",
                          "random_seed", "cuda"]

def save_config(src: Path, inference_save_path: str):
    """Copy the YAML config into the checkpoint directory for provenance."""
    if not inference_save_path:
        return

    inference_save_dir = Path(inference_save_path)
    inference_save_dir.mkdir(parents=True, exist_ok=True)

    destination = inference_save_dir / src.name

    if src.resolve() == destination.resolve():
        return

    shutil.copy2(src, destination)


def main():
    """Entrypoint: load config specified via CLI and launch inference."""
    args = parse_config_cli("Run inference with an external config.")
    config = load_yaml_config(
        Path(args.config),
        numeric_keys=INFERENCE_NUMERIC_KEYS,
    )
    save_config(Path(args.config), config.get('inference_save_path'))
    run_inference(config)

if __name__ == '__main__':
    main()