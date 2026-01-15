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
    config_dir = config_path.parent.resolve()
    config = load_yaml_config(
        config_path,
        numeric_keys=INFERENCE_NUMERIC_KEYS,
    )

    # Resolve all file/dir paths relative to the config location to make execution path-agnostic.
    def _resolve_path(p: str | Path | None) -> str:
        if p is None:
            return None
        pth = Path(p)
        return str(pth if pth.is_absolute() else (config_dir / pth))

    for key in ("inference_save_dir", "model_weights_path", "img_dir", "annotations_file"):
        if key in config and config[key]:
            config[key] = _resolve_path(config[key])

    if "reference_rdm_paths" in config and config["reference_rdm_paths"]:
        config["reference_rdm_paths"] = {
            roi: _resolve_path(path) for roi, path in config["reference_rdm_paths"].items()
        }

    ## SAVE A COPY OF THE CONFIG INTO THE INFERENCE OUTPUT DIR
    save_config(config_path, config["inference_save_dir"])
    ## RUN INFERENCE
    inference_module = import_module("src.inference.inference_core")
    run_inference = inference_module.run_inference
    run_inference(config)


if __name__ == "__main__":
    main()