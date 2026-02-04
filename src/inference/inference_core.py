from src.models.clip_hba.clip_hba_utils import load_dora_checkpoint, initialize_cliphba_model
from src.data.spose_dimensions import classnames66
import torch
import os
import numpy as np
import scipy.io
import pandas as pd
import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr, pearsonr

from src.utils.logging import setup_logger

from src.inference.evaluate_nights import evaluate_nights
from src.inference.extract_embeddings import extract_embeddings


def compute_model_rdm(embeddings, dataset_name, annotations_file, distance_metric, categories=None):
    """
    Compute a representational dissimilarity matrix from embeddings.

    Args:
        embeddings (np.ndarray | torch.Tensor): Row-wise embedding matrix.
        distance_metric (str): Dissimilarity to apply. Supports ``'pearson'`` and
            ``'cosine'``.

    Returns:
        np.ndarray: Square RDM whose entries represent pairwise dissimilarities.

    Raises:
        ValueError: If ``distance_metric`` is not a supported option.
    """

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    else:
        embeddings = np.asarray(embeddings)

    if (
        dataset_name == "things"
        and "concept" in pd.read_csv(annotations_file, nrows=1).columns
        and categories is not None
    ):
        # average the embeddings by concept using provided category labels
        df = pd.DataFrame(embeddings)
        df["concept"] = categories
        embeddings = df.groupby("concept").mean().to_numpy()

    if distance_metric is None:
        raise ValueError("model_rdm_distance_metric must be provided in the config.")

    if distance_metric == 'pearson':
        rdm = 1 - np.corrcoef(embeddings)
        np.fill_diagonal(rdm, 0)
        return rdm
    elif distance_metric == 'cosine':
        rdm = 1 - cosine_similarity(embeddings)
        np.fill_diagonal(rdm, 0)
        return rdm
    else:
        raise ValueError(f"Invalid distance metric: {distance_metric}")


def prepare_reference_rdms(config: dict) -> dict:
    """
    Load one or more reference RDMs based on the provided config.

    Supports:
        - config['reference_rdm_paths']: mapping of ROI -> path (one or more paths to reference RDM(s))
    """
    reference_rdms = {}
    if "reference_rdm_paths" not in config:
        raise ValueError("reference_rdm_paths must be provided for RSA-based evaluation.")
    mapping = config["reference_rdm_paths"]

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

            rdm = np.asarray(rdm)
            # If square, flatten upper triangle (k=1 to drop diagonal)
            if rdm.ndim == 2 and rdm.shape[0] == rdm.shape[1]:
                tri_idx = np.triu_indices_from(rdm, k=1)
                rdm = rdm[tri_idx]
            reference_rdms[roi_name] = rdm

    elif mapping is None:
        raise ValueError(
            "reference_rdm_paths must be provided for RSA-based evaluation."
        )
    else:
        raise ValueError("reference_rdm_paths is empty; provide at least one reference RDM path.")

    return reference_rdms


def compute_rdm_similarity(model_rdm_upper_tri_vector, reference_rdm_upper_tri_vector, similarity_metric):
    """
    Compare two representational dissimilarity matrices using a rank or linear
    correlation on their upper-triangular entries.

    Args:
        model_rdm (np.ndarray): RDM derived from model embeddings.
        reference_rdm (np.ndarray): Target RDM to correlate against.
        similarity_metric (str): Correlation statistic to apply. Supports
            ``'spearman'`` (rank) or ``'pearson'`` (linear).

    Returns:
        tuple: ``(rho, p_value)`` as returned by the selected SciPy correlation.

    Raises:
        ValueError: If ``similarity_metric`` is not recognized.
    """

    if similarity_metric is None:
        raise ValueError("rsa_similarity_metric must be provided in the config.")

    if similarity_metric == 'spearman':
        rho, p_value = spearmanr(model_rdm_upper_tri_vector, reference_rdm_upper_tri_vector)
        return rho, p_value
    elif similarity_metric == 'pearson':
        rho, p_value = pearsonr(model_rdm_upper_tri_vector, reference_rdm_upper_tri_vector)
        return rho, p_value
    else:
        raise ValueError(f"Invalid similarity metric: {similarity_metric}")


def run_inference(config):

    ## INITIALIZE LOGGER
    os.makedirs(config['inference_save_dir'], exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(config['inference_save_dir'], f'inference_log_{config["dataset"]}_{config["evaluation_type"]}.txt')
    logger = setup_logger(log_file)
    logger.info("Run timestamp: %s", timestamp)

    # Persist the config snapshot alongside outputs
    snapshot_path = Path(config['inference_save_dir']) / "inference_config_snapshot.yaml"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    logger.info("Saved config snapshot to %s", snapshot_path)
    
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
        raise ValueError("Unsupported cuda setting. Expected -1, 0, or 1.")

    ## LOAD MODEL WEIGHTS FROM CHECKPOINT
    weights_path = Path(config['model_weights_path'])
    loaded_direct = False
    if weights_path.is_file():
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        epoch = weights_path.stem.replace('epoch', '').replace('_dora_params', '')
        loaded_direct = True
    else:
        dora_dir = weights_path if weights_path.name == "dora_params" else weights_path / "dora_params"
        checkpoint_root = dora_dir.parent
        checkpoint_files = sorted(dora_dir.glob("epoch*_dora_params.pth"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No DoRA checkpoints found under {dora_dir}")

        requested_epoch = config["epoch"]
        if requested_epoch is not None:
            epoch = str(requested_epoch)
            candidate = dora_dir / f"epoch{epoch}_dora_params.pth"
            if not candidate.exists():
                raise FileNotFoundError(f"Requested epoch checkpoint not found: {candidate}")
        else:
            latest_ckpt = checkpoint_files[-1]
            epoch = latest_ckpt.stem.replace('epoch', '').replace('_dora_params', '')

    ## SET EVALUATION TYPE
    if "evaluation_type" not in config:
        logger.error("Evaluation type is not set. Please set evaluation_type in the config file.")
        raise ValueError("evaluation_type must be set in the config file.")
    evaluation_type = config["evaluation_type"]

    logger.info("\n=== Processing epoch %s ===", epoch)

    if not loaded_direct:
        load_dora_checkpoint(model, checkpoint_root=checkpoint_root, epoch=epoch, map_location=device)

    model = model.to(device)

    ## DISPATCH TO DATASET-SPECIFIC LOGIC
    results = None
    score = None

    if evaluation_type in ("neural", "behavioral"):
        # extract embeddings from the model
        embedding_outputs = extract_embeddings(
            model=model,
            dataset_name=config['dataset'],
            img_dir=config['img_dir'],
            annotations_file=config['annotations_file'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            max_images_per_category=config['max_images_per_category'],
            device=device,
            logger=logger,
        )

        # create an RDM from the model's embeddings
        model_rdm = compute_model_rdm(
            embedding_outputs["embeddings"],
            dataset_name=config['dataset'],
            annotations_file=config['annotations_file'],
            categories=embedding_outputs["categories"],
            distance_metric=config["model_rdm_distance_metric"]
        )

        # get the upper triangular vector of the model RDM
        model_rdm_upper_tri_indices = np.triu_indices_from(model_rdm, k=1)
        model_rdm_upper_tri_vector = model_rdm[model_rdm_upper_tri_indices]

        # load reference RDM(s) - there may be multiple reference RDMs because we have
        reference_rdms_upper_tri_vectors = prepare_reference_rdms(config)
        reference_rdm_distance_metric = config["reference_rdm_distance_metric"]
        
        # compute RSA scores for each reference RDM
        rsa_similarity_metric = config["rsa_similarity_metric"]

        rsa_results = {}
        for reference_rdm_name, reference_rdm_upper_tri_vector in reference_rdms_upper_tri_vectors.items():
            if reference_rdm_upper_tri_vector.shape != model_rdm_upper_tri_vector.shape:
                raise ValueError(
                    f"Reference RDM for ROI '{reference_rdm_name}' has shape {reference_rdm_upper_tri_vector.shape} "
                    f"but model RDM has shape {model_rdm_upper_tri_vector.shape}"
                )

            rho, p_value = compute_rdm_similarity(
                model_rdm_upper_tri_vector,
                reference_rdm_upper_tri_vector,
                similarity_metric=rsa_similarity_metric,
            )

            rsa_results[reference_rdm_name] = {
                "epoch": epoch,
                "evaluation_type": evaluation_type,
                "dataset": config['dataset'],
                "reference_rdm_name": reference_rdm_name,
                "score": float(rho),
                "p_value": float(p_value),
                "rsa_similarity_metric": rsa_similarity_metric,
                "model_rdm_distance_metric": config["model_rdm_distance_metric"],
                "reference_rdm_distance_metric": reference_rdm_distance_metric,
            }

            if logger:
                logger.info(
                    "RSA (%s) score for target RDM '%s': %.4f (p=%.4g) using distance metric '%s'",
                    rsa_similarity_metric,
                    reference_rdm_name,
                    rho,
                    p_value,
                    config["model_rdm_distance_metric"],
                )

        # Persist RSA results 
        results = rsa_results
        score = None

        # Write results to a JSON file 
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
            batch_size=config['batch_size'],
            device=device,
            use_image_features=False,
            cached_batches=None,
        )
        score = float(results.get("accuracy", 0.0))

        triplet_evaluation_results = {
            "evaluation_type": evaluation_type,
            "dataset": config['dataset'],
            "score": score,
        }

        if logger:
            logger.info(
            "Triplet score for '%s' dataset: %.4f",
            config['dataset'],
            score,
            )

    else:
        raise ValueError(
            f"Unsupported combination of dataset '{config['dataset']}' "
            f"and evaluation_type '{evaluation_type}'."
        )

    if score is None:
        logger.info("Triplet score for epoch %s (%s): None", epoch, evaluation_type)
    else:
        logger.info("Triplet score for epoch %s (%s): %.4f", epoch, evaluation_type, score)

    return {
        "epoch": epoch,
        "score": score,
        "results": results,
    }