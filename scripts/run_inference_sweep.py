"""
Single-process behavioral RSA sweep over all DoRA checkpoints in a run directory.

Compared to calling run_inference.py once per epoch from a shell loop, this
script pays the Python/PyTorch startup cost and image-loading cost exactly
once, then iterates over checkpoints by swapping only the DoRA state dict and
re-running the forward pass.

Usage
-----
    python scripts/run_inference_sweep.py --config configs/inference_config.yaml \
        --checkpoint_dir  /path/to/run/dora_params \
        --save_dir        /path/to/output_root \
        --run_name        vit_l_14_rank32_perturb-type-none_init-seed1
"""

from importlib import import_module
from pathlib import Path
import argparse
import json
import os
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

from src.utils.load_yaml_config import load_yaml_config
from src.utils.logging import setup_logger
from src.models.clip_hba.clip_hba_utils import initialize_cliphba_model
from src.data.spose_dimensions import classnames66
from src.inference.extract_embeddings import extract_embeddings
from src.inference.inference_core import (
    compute_model_rdm,
    compute_rdm_similarity,
    prepare_reference_rdms,
)

INFERENCE_INT_KEYS = [
    "batch_size", "num_classes", "rank", "vision_layers",
    "transformer_layers", "random_seed", "cuda", "num_workers",
    "max_images_per_category",
]


def _parse_args():
    p = argparse.ArgumentParser(description="RSA sweep over all DoRA checkpoints.")
    p.add_argument("--config", required=True, type=Path,
                   help="Base inference config YAML.")
    p.add_argument("--checkpoint_dir", type=Path, default=None,
                   help="Directory containing epoch*_dora_params.pth files. "
                        "Overrides the PTH_ROOT in the shell script.")
    p.add_argument("--save_dir", type=Path, default=None,
                   help="Root output directory. Overrides inference_save_dir in config.")
    p.add_argument("--run_name", type=str, default=None,
                   help="Human-readable label for this run (used in output paths and JSON).")
    return p.parse_args()


def _device_from_config(config):
    cuda = config.get("cuda", 0)
    if cuda == -1:
        return torch.device("cuda")
    return torch.device(f"cuda:{cuda}")


def _sorted_checkpoints(ckpt_dir: Path):
    """Return epoch*_dora_params.pth files sorted by epoch number."""
    files = list(ckpt_dir.glob("epoch*_dora_params.pth"))
    def _epoch_num(p):
        stem = p.stem  # e.g. epoch42_dora_params
        num = stem.replace("epoch", "").replace("_dora_params", "")
        return int(num)
    return sorted(files, key=_epoch_num)


def main():
    args = _parse_args()
    cwd = Path.cwd()

    config = load_yaml_config(args.config, numeric_keys=INFERENCE_INT_KEYS)

    # Resolve relative paths against cwd (the repo root).
    def _resolve(p):
        if p is None:
            return None
        pth = Path(p)
        return pth if pth.is_absolute() else cwd / pth

    for key in ("img_dir", "annotations_file"):
        if config.get(key):
            config[key] = str(_resolve(config[key]))

    if config.get("reference_rdm_paths"):
        config["reference_rdm_paths"] = {
            k: str(_resolve(v)) for k, v in config["reference_rdm_paths"].items()
        }

    # CLI overrides for paths.
    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else _resolve(config.get("model_weights_path"))
    if ckpt_dir.is_file():
        # If a single file was given, treat its parent as the checkpoint dir.
        ckpt_dir = ckpt_dir.parent

    base_save_dir = Path(args.save_dir) if args.save_dir else _resolve(config.get("inference_save_dir"))
    run_name = args.run_name or ckpt_dir.parent.name

    dataset      = config.get("dataset", "things")
    eval_type    = config["evaluation_type"]
    subdir       = f"test_{dataset}_{eval_type}_inference"
    agg_json     = base_save_dir / f"{subdir}_rsa_summary.json"

    os.makedirs(base_save_dir, exist_ok=True)
    log_file = base_save_dir / f"sweep_log_{dataset}_{eval_type}.txt"
    logger = setup_logger(str(log_file))
    logger.info("=== Inference sweep started ===")
    logger.info("  checkpoint_dir : %s", ckpt_dir)
    logger.info("  save_dir       : %s", base_save_dir)
    logger.info("  run_name       : %s", run_name)
    logger.info("  dataset        : %s  eval_type: %s", dataset, eval_type)

    device = _device_from_config(config)
    logger.info("  device         : %s", device)

    # ------------------------------------------------------------------
    # Initialise the model once (loads the CLIP backbone weights once).
    # ------------------------------------------------------------------
    logger.info("Initialising CLIP-HBA model...")
    model = initialize_cliphba_model(
        backbone_name=config["backbone"],
        classnames=classnames66,
        vision_layers=config["vision_layers"],
        transformer_layers=config["transformer_layers"],
        rank=config["rank"],
        dora_dropout=0.1,
        logger=logger,
    )
    model = model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # Pre-load the reference RDM(s) once — they don't change per epoch.
    # ------------------------------------------------------------------
    logger.info("Loading reference RDM(s)...")
    reference_rdms = prepare_reference_rdms(config)
    reference_rdm_distance_metric = config["reference_rdm_distance_metric"]
    rsa_similarity_metric = config["rsa_similarity_metric"]
    for rdm_name, rdm_vec in reference_rdms.items():
        logger.info(
            "  [%s] upper-tri length: %d  min=%.4f  max=%.4f  mean=%.4f",
            rdm_name, len(rdm_vec), rdm_vec.min(), rdm_vec.max(), rdm_vec.mean(),
        )

    # ------------------------------------------------------------------
    # Load and cache all images once via a dry forward pass on epoch 0,
    # but we can't cache embeddings since they change per checkpoint.
    # We DO construct the DataLoader once so images are only read from
    # disk on the first epoch (OS page cache handles subsequent epochs).
    # ------------------------------------------------------------------
    logger.info("--- FILES ---")
    logger.info("  img_dir:          %s", config["img_dir"])
    logger.info("  annotations_file: %s", config["annotations_file"])

    checkpoints = _sorted_checkpoints(ckpt_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No epoch*_dora_params.pth files found under {ckpt_dir}")
    logger.info("Found %d checkpoints to evaluate.", len(checkpoints))

    # ------------------------------------------------------------------
    # Sweep over checkpoints.
    # ------------------------------------------------------------------
    agg = {
        "dataset": dataset,
        "evaluation_type": eval_type,
        "run_name": run_name,
        "description": "Behavioral alignment RSA results across checkpoints.",
        "results": [],
    }
    if agg_json.exists():
        with open(agg_json) as f:
            agg = json.load(f)

    for ckpt_path in checkpoints:
        stem = ckpt_path.stem  # epoch42_dora_params
        epoch = stem.replace("epoch", "").replace("_dora_params", "")

        logger.info("\n=== Epoch %s  (%s) ===", epoch, ckpt_path.name)

        # Swap in this checkpoint's DoRA state dict (no model re-init).
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        # Forward pass — extract embeddings for this checkpoint's weights.
        embedding_outputs = extract_embeddings(
            model=model,
            dataset_name=dataset,
            img_dir=config["img_dir"],
            annotations_file=config["annotations_file"],
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            max_images_per_category=config["max_images_per_category"],
            device=device,
            logger=logger,
        )

        embeddings = embedding_outputs["embeddings"]
        categories = embedding_outputs["categories"]
        logger.info(
            "  Embeddings shape: %s  unique categories: %d",
            tuple(embeddings.shape),
            len(set(categories)) if categories else 0,
        )

        # Build model RDM.
        model_rdm = compute_model_rdm(
            embeddings,
            dataset_name=dataset,
            annotations_file=config["annotations_file"],
            categories=categories,
            distance_metric=config["model_rdm_distance_metric"],
        )
        logger.info(
            "  Model RDM shape: %s  min=%.4f  max=%.4f  mean=%.4f",
            model_rdm.shape, model_rdm.min(), model_rdm.max(), model_rdm.mean(),
        )

        tri_idx = np.triu_indices_from(model_rdm, k=1)
        model_rdm_vec = model_rdm[tri_idx]

        # RSA against each reference RDM.
        rsa_results = {}
        for ref_name, ref_vec in reference_rdms.items():
            if ref_vec.shape != model_rdm_vec.shape:
                raise ValueError(
                    f"Shape mismatch for '{ref_name}': "
                    f"reference {ref_vec.shape} vs model {model_rdm_vec.shape}"
                )
            rho, p_value = compute_rdm_similarity(
                model_rdm_vec, ref_vec, similarity_metric=rsa_similarity_metric,
            )
            rsa_results[ref_name] = {
                "epoch": epoch,
                "evaluation_type": eval_type,
                "dataset": dataset,
                "reference_rdm_name": ref_name,
                "score": float(rho),
                "p_value": float(p_value),
                "rsa_similarity_metric": rsa_similarity_metric,
                "model_rdm_distance_metric": config["model_rdm_distance_metric"],
                "reference_rdm_distance_metric": reference_rdm_distance_metric,
            }
            logger.info("  RSA [%s] rho=%.4f  p=%.4g", ref_name, rho, p_value)

        # Append to aggregate JSON (update in place if epoch already present).
        record = {
            "epoch": epoch,
            "run_name": run_name,
            "dataset": dataset,
            "evaluation_type": eval_type,
            "rsa_results": rsa_results,
        }
        agg.setdefault("results", [])
        agg["results"] = [
            r for r in agg["results"]
            if not (r.get("epoch") == epoch and r.get("run_name") == run_name)
        ]
        agg["results"].append(record)
        agg["results"].sort(key=lambda r: (
            int(r.get("epoch", 0)) if str(r.get("epoch", 0)).isdigit() else 0,
            str(r.get("run_name", "")),
        ))
        with open(agg_json, "w") as f:
            json.dump(agg, f, indent=2)

    logger.info("\n=== Sweep complete. Aggregate results: %s ===", agg_json)


if __name__ == "__main__":
    main()
