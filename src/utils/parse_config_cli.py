import argparse
from pathlib import Path


def parse_config_cli(description: str) -> argparse.Namespace:
    """
    Build the standard CLI for scripts that take a single --config YAML file.

    Optional ``--set key=value`` arguments override individual config keys
    after the YAML is loaded.  Values are kept as strings here; numeric
    coercion happens in ``load_yaml_config`` via the caller's key lists.

    Example::

        python scripts/run_training.py \\
            --config configs/resnet50_imagenet.yaml \\
            --set img_dir=/tmp/imagenet_12345 \\
            --set cuda=-1
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config file (e.g., configs/run.yaml)",
    )
    parser.add_argument(
        "--set",
        metavar="KEY=VALUE",
        action="append",
        default=[],
        dest="overrides",
        help=(
            "Override a config key at runtime, e.g. --set img_dir=/tmp/data. "
            "Can be repeated for multiple keys."
        ),
    )
    return parser.parse_args()