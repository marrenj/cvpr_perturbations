"""
RSA sweep over all perceptual-init yfcc3m ViT-B/32 Lightning checkpoints.

Loads each .ckpt checkpoint once (swapping only the visual-encoder weights
between epochs), extracts 512-d image embeddings for the 48 THINGS images,
computes a model RDM, and reports the Spearman RSA against the behavioral
reference RDM.

Usage
-----
    python scripts/run_perceptual_init_inference.py \\
        --config  configs/perceptual_init_vitb32_inference.yaml \\
        --checkpoint_dir /home/wallacelab/teba/multimodal_brain_inspired/PI_dk/pct_checkpoints/perceptual_init_yfcc3m_vitb32 \\
        --save_dir       /home/wallacelab/teba/multimodal_brain_inspired/PI_dk/inference/perceptual_init_yfcc3m_vitb32 \\
        --run_name       perceptual_init_yfcc3m_vitb32
"""

from pathlib import Path
import argparse
import json
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.utils.load_yaml_config import load_yaml_config
from src.utils.logging import setup_logger
from src.models.perceptual_init.model import PerceptualInitCLIPWrapper
from src.models.perceptual_init.perceptual_init_utils import (
    create_vitb32_visual_encoder,
    load_visual_state_dict_from_lightning_checkpoint,
    parse_epoch_from_lightning_filename,
    sorted_lightning_checkpoints,
)
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
    p = argparse.ArgumentParser(
        description="RSA sweep over perceptual-init yfcc3m ViT-B/32 checkpoints."
    )
    p.add_argument("--config", required=True, type=Path,
                   help="Base inference config YAML.")
    p.add_argument("--checkpoint_dir", type=Path, default=None,
                   help="Directory containing Lightning .ckpt files. "
                        "Overrides model_weights_path in config.")
    p.add_argument("--save_dir", type=Path, default=None,
                   help="Root output directory. Overrides inference_save_dir in config.")
    p.add_argument("--run_name", type=str, default=None,
                   help="Human-readable label for this run (used in output JSON).")
    return p.parse_args()


def _device_from_config(config):
    cuda = config.get("cuda", 0)
    if cuda == -1:
        return torch.device("cuda")
    return torch.device(f"cuda:{cuda}")


def main():
    args = _parse_args()
    cwd = Path.cwd()

    config = load_yaml_config(args.config, numeric_keys=INFERENCE_INT_KEYS)

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

    ckpt_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else _resolve(config.get("model_weights_path"))
    )
    if ckpt_dir.is_file():
        ckpt_dir = ckpt_dir.parent

    base_save_dir = (
        Path(args.save_dir)
        if args.save_dir
        else _resolve(config.get("inference_save_dir"))
    )
    run_name = args.run_name or ckpt_dir.name

    dataset = config.get("dataset", "things")
    eval_type = config["evaluation_type"]
    agg_json = base_save_dir / f"test_{dataset}_{eval_type}_rsa_summary.json"

    os.makedirs(base_save_dir, exist_ok=True)
    log_file = base_save_dir / f"sweep_log_{dataset}_{eval_type}.txt"
    logger = setup_logger(str(log_file))
    logger.info("=== Perceptual-init ViT-B/32 inference sweep started ===")
    logger.info("  checkpoint_dir : %s", ckpt_dir)
    logger.info("  save_dir       : %s", base_save_dir)
    logger.info("  run_name       : %s", run_name)
    logger.info("  dataset        : %s  eval_type: %s", dataset, eval_type)

    device = _device_from_config(config)
    logger.info("  device         : %s", device)

    # Initialise the visual encoder once; weights are swapped per checkpoint.
    logger.info("Initialising ViT-B/32 visual encoder (bi_clip architecture)...")
    visual_encoder = create_vitb32_visual_encoder()
    model = PerceptualInitCLIPWrapper(visual_encoder)
    model = model.to(device)
    model.eval()

    # Pre-load reference RDM(s) — unchanged across checkpoints.
    logger.info("Loading reference RDM(s)...")
    reference_rdms = prepare_reference_rdms(config)
    rsa_similarity_metric = config["rsa_similarity_metric"]
    reference_rdm_distance_metric = config["reference_rdm_distance_metric"]
    for rdm_name, rdm_vec in reference_rdms.items():
        logger.info(
            "  [%s] upper-tri length: %d  min=%.4f  max=%.4f  mean=%.4f",
            rdm_name, len(rdm_vec), rdm_vec.min(), rdm_vec.max(), rdm_vec.mean(),
        )

    logger.info("  img_dir:          %s", config["img_dir"])
    logger.info("  annotations_file: %s", config["annotations_file"])

    checkpoints = sorted_lightning_checkpoints(ckpt_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No .ckpt files found under {ckpt_dir}")
    logger.info("Found %d checkpoints to evaluate.", len(checkpoints))

    agg = {
        "dataset": dataset,
        "evaluation_type": eval_type,
        "run_name": run_name,
        "description": "Behavioral RSA results for perceptual-init yfcc3m ViT-B/32.",
        "results": [],
    }
    if agg_json.exists():
        with open(agg_json) as f:
            agg = json.load(f)

    for ckpt_path in checkpoints:
        epoch = parse_epoch_from_lightning_filename(ckpt_path.name)
        logger.info("\n=== Epoch %s  (%s) ===", epoch, ckpt_path.name)

        # Swap in this checkpoint's visual weights without re-initialising the model.
        visual_sd = load_visual_state_dict_from_lightning_checkpoint(
            str(ckpt_path), map_location=str(device)
        )
        model.visual.load_state_dict(visual_sd, strict=True)
        model.eval()

        # Extract 512-d L2-normalised embeddings for the 48 THINGS images.
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

        # Build model RDM from the 512-d features.
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

        def _sort_key(r):
            e = r.get("epoch", 0)
            try:
                return (int(e), str(r.get("run_name", "")))
            except (ValueError, TypeError):
                return (10 ** 9, str(r.get("run_name", "")))

        agg["results"].sort(key=_sort_key)
        with open(agg_json, "w") as f:
            json.dump(agg, f, indent=2)

    logger.info("\n=== Sweep complete. Aggregate results: %s ===", agg_json)


if __name__ == "__main__":
    main()
