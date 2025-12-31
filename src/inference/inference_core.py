from src.models.clip_hba.clip_hba_utils import load_dora_checkpoint, initialize_cliphba_model
from src.data.spose_dimensions import classnames66
import torch
import os
import numpy as np
import scipy.io
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.utils.logging import setup_logger

from src.evaluation.rsa import compute_rdm, compute_rdm_similarity
from src.inference.evaluate_nights import evaluate_nights
from src.inference.extract_embeddings import extract_embeddings


def _validate_square_rdm(rdm: np.ndarray, source_label: str) -> np.ndarray:
    """
    Validate that an RDM is square and return it as a NumPy array.
    """
    rdm = np.asarray(rdm)
    if rdm.ndim != 2 or rdm.shape[0] != rdm.shape[1]:
        raise ValueError(f"Reference RDM from {source_label} must be square, got shape {rdm.shape}")
    return rdm


def prepare_reference_rdm(reference_rdm_path: Path) -> np.ndarray:
    """
    Load a reference RDM from disk. Supports .npy, .npz, .mat, .csv, .tsv, and .txt files.
    """
    if not reference_rdm_path.exists():
        raise FileNotFoundError(f"Reference RDM not found: {reference_rdm_path}")

    suffix = reference_rdm_path.suffix.lower()

    if suffix == ".npy":
        rdm = np.load(reference_rdm_path)
    elif suffix == ".npz":
        with np.load(reference_rdm_path) as data:
            if len(data.files) == 1:
                rdm = data[data.files[0]]
            else:
                raise ValueError(
                    f"Multiple arrays found in {reference_rdm_path}. "
                    "Please provide a single array in the .npz file."
                )
    elif suffix == ".mat":
        mat = scipy.io.loadmat(reference_rdm_path)
        candidate_keys = [k for k in mat.keys() if not k.startswith("__")]
        if len(candidate_keys) == 1:
            rdm = mat[candidate_keys[0]]
        else:
            raise ValueError(
                f"Multiple variables found in {reference_rdm_path}. "
                "Please provide a single array in the .mat file."
            )
    elif suffix in (".csv", ".tsv", ".txt"):
        delimiter = "\t" if suffix == ".tsv" else ","
        rdm = np.loadtxt(reference_rdm_path, delimiter=delimiter)
    else:
        raise ValueError(f"Unsupported reference RDM format: {reference_rdm_path.suffix}")

    return _validate_square_rdm(rdm, str(reference_rdm_path))


def prepare_reference_rdms(config: dict) -> dict:
    """
    Load one or more reference RDMs based on the provided config.

    Supports:
        - config['reference_rdm_paths']: mapping of ROI -> path (one or more paths to reference RDM(s))
    """
    reference_rdms = {}
    mapping = config.get("reference_rdm_paths")

    if mapping:
        for roi_name, reference_rdm_path_str in mapping.items():

            reference_rdm_path = Path(reference_rdm_path_str)
            
            suffix = reference_rdm_path.suffix.lower()

            if suffix == ".npy":
                rdm = np.load(reference_rdm_path)
            elif suffix == ".npz":
                with np.load(reference_rdm_path) as data:
                    if len(data.files) == 1:
                        rdm = data[data.files[0]]
                    else:
                        raise ValueError(
                            f"Multiple arrays found in {reference_rdm_path}. "
                            "Please provide a single array in the .npz file."
                        )
            elif suffix == ".mat":
                mat = scipy.io.loadmat(reference_rdm_path)
                candidate_keys = [k for k in mat.keys() if not k.startswith("__")]
                if len(candidate_keys) == 1:
                    rdm = mat[candidate_keys[0]]
                else:
                    raise ValueError(
                         f"Multiple variables found in {reference_rdm_path}. "
                         "Please provide a single array in the .mat file."
                        )
            elif suffix in (".csv", ".tsv", ".txt"):
                delimiter = "\t" if suffix == ".tsv" else ","
                rdm = np.loadtxt(reference_rdm_path, delimiter=delimiter)
            else:
                raise ValueError(f"Unsupported reference RDM format: {reference_rdm_path.suffix}")

            reference_rdms[roi_name] = rdm

    elif mapping is None:
        raise ValueError(
            "reference_rdm_paths must be provided for RSA-based evaluation."
        )
    else:
        raise ValueError("reference_rdm_paths is empty; provide at least one reference RDM path.")

    return reference_rdms


def compute_rsa_alignment(
    embedding_outputs: dict,
    reference_rdm: np.ndarray,
    rdm_distance_metric: str = "pearson",
    rsa_similarity_metric: str = "spearman",
    reference_rdm_distance_metric: Optional[str] = None,
    logger=None,
) -> dict:
    """
    Compute RSA alignment score between model embeddings and a reference RDM.
    """

    model_rdm = compute_rdm(
        embedding_outputs["embeddings"],
        distance_metric=rdm_distance_metric,
    )

    rho, p_value = compute_rdm_similarity(
        model_rdm,
        reference_rdm,
        similarity_metric=rsa_similarity_metric,
    )

    if logger:
        logger.info(
            "RSA (%s) score: %.4f (p=%.4g) using distance metric '%s'",
            rsa_similarity_metric,
            rho,
            p_value,
            rdm_distance_metric,
        )

    return {
        "score": float(rho),
        "p_value": float(p_value),
        "model_rdm": model_rdm,
        "rdm_distance_metric": rdm_distance_metric,
        "reference_rdm_distance_metric": reference_rdm_distance_metric or rdm_distance_metric,
        "rsa_similarity_metric": rsa_similarity_metric,
        "image_names": embedding_outputs.get("image_names"),
    }


def run_inference(config):

    ## INITIALIZE LOGGER
    os.makedirs(config['inference_save_dir'], exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(config['inference_save_dir'], f'inference_log_{timestamp}.txt')
    logger = setup_logger(log_file)
    
    ## INITIALIZE CLIPHBA MODEL
    model = initialize_cliphba_model(
        backbone_name=config['backbone'],
        classnames=classnames66,
        vision_layers=config['vision_layers'],
        transformer_layers=config['transformer_layers'],
        rank=config['rank'],
        dora_dropout=0.1,
        logger=logger
    )

    model.eval() # inference mode
    
    ## SET DEVICE
    if config['cuda'] == -1:
        device = torch.device("cuda")
    elif config['cuda'] == 0:
        device = torch.device("cuda:0")
    elif config['cuda'] == 1:
        device = torch.device("cuda:1")
    else:
        device = torch.device("cpu")

    ## LOAD MODEL WEIGHTS FROM CHECKPOINT
    weights_path = Path(config['model_weights_path'])
    if weights_path.is_file():
        checkpoint_root = (
            weights_path.parent.parent if weights_path.parent.name == "dora_params" else weights_path.parent
        )
        epoch = weights_path.stem.replace('epoch', '').replace('_dora_params', '')
    else:
        dora_dir = weights_path if weights_path.name == "dora_params" else weights_path / "dora_params"
        checkpoint_root = dora_dir.parent
        checkpoint_files = sorted(dora_dir.glob("epoch*_dora_params.pth"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No DoRA checkpoints found under {dora_dir}")

        requested_epoch = config.get("epoch")
        if requested_epoch is not None:
            epoch = str(requested_epoch)
            candidate = dora_dir / f"epoch{epoch}_dora_params.pth"
            if not candidate.exists():
                raise FileNotFoundError(f"Requested epoch checkpoint not found: {candidate}")
        else:
            latest_ckpt = checkpoint_files[-1]
            epoch = latest_ckpt.stem.replace('epoch', '').replace('_dora_params', '')

    ## SET EVALUATION TYPE
    evaluation_type = config.get("evaluation_type")
    if evaluation_type is None:
        logger.error("Evaluation type is not set. Please set evaluation_type in the config file.")
        raise ValueError("evaluation_type must be set in the config file.")

    logger.info("\n=== Processing epoch %s ===", epoch)

    load_dora_checkpoint(model, checkpoint_root=checkpoint_root, epoch=epoch, map_location=device)

    model = model.to(device)

    ## DISPATCH TO DATASET-SPECIFIC LOGIC
    results = None
    score = None

    if evaluation_type in ("behavioral", "neural"):
        embedding_outputs = extract_embeddings(
            model=model,
            dataset_name=config['dataset'],
            config=config,
            device=device,
            logger=logger,
        )

        model_rdm = compute_rdm(
            embedding_outputs["embeddings"],
            distance_metric=config.get("model_rdm_distance_metric", "pearson"),
        )

        reference_rdms = prepare_reference_rdms(config)
        rsa_similarity_metric = config.get("rsa_similarity_metric", "spearman")
        ref_rdm_distance_metric = config.get(
            "reference_rdm_distance_metric",
            config.get("model_rdm_distance_metric", "pearson"),
        )

        roi_results = {}
        roi_scores = []
        for roi_name, reference_rdm in reference_rdms.items():
            if reference_rdm.shape != model_rdm.shape:
                raise ValueError(
                    f"Reference RDM for ROI '{roi_name}' has shape {reference_rdm.shape} "
                    f"but model RDM has shape {model_rdm.shape}"
                )

            rho, p_value = compute_rdm_similarity(
                model_rdm,
                reference_rdm,
                similarity_metric=rsa_similarity_metric,
            )

            roi_results[roi_name] = {
                "score": float(rho),
                "p_value": float(p_value),
                "rdm_distance_metric": config.get("model_rdm_distance_metric", "pearson"),
                "reference_rdm_distance_metric": ref_rdm_distance_metric,
                "rsa_similarity_metric": rsa_similarity_metric,
            }
            roi_scores.append(rho)

            if logger:
                logger.info(
                    "RSA (%s) score for ROI '%s': %.4f (p=%.4g) using distance metric '%s'",
                    rsa_similarity_metric,
                    roi_name,
                    rho,
                    p_value,
                    config.get("model_rdm_distance_metric", "pearson"),
                )

        mean_score = float(np.mean(roi_scores)) if roi_scores else float("nan")
        if not roi_results:
            raise ValueError("No reference RDMs were loaded; check reference_rdm_path(s) or keys.")

        if logger:
            logger.info(
                "Mean RSA score across %d ROI(s): %.4f",
                len(roi_results),
                mean_score,
            )

        # Prepare results as a dictionary 
        results = {}
        results["alignment_target"] = evaluation_type
        results["epoch"] = epoch
        results["score"] = mean_score
        if len(roi_results) == 1:
            single_roi = next(iter(roi_results.values()))
            results["p_value"] = single_roi["p_value"]
        results["model_rdm_distance_metric"] = config.get("model_rdm_distance_metric")
        results["reference_rdm_distance_metric"] = ref_rdm_distance_metric
        results["rsa_similarity_metric"] = rsa_similarity_metric
        results["roi_results"] = roi_results
        score = mean_score

        # Write results to a JSON file (in addition to .pt)
        results_json_path = Path(config['inference_save_dir']) / f"inference_results_{config['dataset']}_{evaluation_type}_epoch{epoch}.json"
        import json
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        logger.info("Saved inference results (JSON) to %s", results_json_path)

    elif evaluation_type == "triplet" and config['dataset'] == 'nights':
        results, _ = evaluate_nights(
            model=model,
            nights_dir=config['img_dir'],
            split='test',
            batch_size=config.get('batch_size', 32),
            device=device,
            use_image_features=False,
            cached_batches=None,
        )
        score = float(results.get("accuracy", 0.0))

    else:
        raise ValueError(
            f"Unsupported combination of dataset '{config['dataset']}' "
            f"and evaluation_type '{evaluation_type}'."
        )

    logger.info("Score for epoch %s (%s): %.4f", epoch, evaluation_type, score)

    # Persist results for downstream analysis
    results_path = Path(config['inference_save_dir']) / f"inference_results_{config['dataset']}_{evaluation_type}_epoch{epoch}.pt"
    torch.save(
        {
            "dataset": config["dataset"],
            "evaluation_type": evaluation_type,
            "epoch": epoch,
            "results": results,
            "score": score,
        },
        results_path,
    )
    logger.info("Saved inference results to %s", results_path)

    return {
        "epoch": epoch,
        "score": score,
        "results": results,
    }