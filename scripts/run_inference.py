from importlib import import_module
from pathlib import Path
import sys

# Ensure repository root is on sys.path when the script is invoked via an absolute path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.load_yaml_config import load_yaml_config
from src.utils.parse_config_cli import parse_config_cli
from src.utils.save_config import save_config


INFERENCE_NUMERIC_KEYS = [
    "epochs",
    "batch_size",
    "rank",
    "vision_layers",
    "transformer_layers",
    "random_seed",
    "cuda",
]


def main():
    """Entrypoint: load config specified via CLI and launch inference."""
    ## PARSE COMMAND LINE ARGUMENTS
    args = parse_config_cli("Run inference with an external config.")
    ## LOAD YAML CONFIG
    config_path = Path(args.config)
    config = load_yaml_config(
        config_path,
        numeric_keys=INFERENCE_NUMERIC_KEYS,
    )
    ## SAVE A COPY OF THE CONFIG INTO THE INFERENCE OUTPUT DIR
    save_config(config_path, config["inference_save_dir"])
    ## RUN INFERENCE
    inference_module = import_module("src.inference.inference_core")
    run_inference = inference_module.run_inference
    run_inference(config)


if __name__ == "__main__":
    main()