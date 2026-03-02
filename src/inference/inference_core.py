from src.models.clip_hba.clip_hba_utils import load_dora_checkpoint, initialize_cliphba_model
from src.models.factory import build_model
from src.training.trainer import load_model_checkpoint
from src.data.spose_dimensions import classnames66
import torch
import torch.nn.functional as F
import os
import numpy as np
import scipy.io
import pandas as pd
import yaml
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.utils.logging import setup_logger

from src.inference.evaluate_nights import evaluate_nights
from src.inference.extract_embeddings import extract_embeddings


def center_gram(gram, unbiased=False):
    """Double-center a symmetric Gram matrix for CKA computation.
    Equivalent to centering the (possibly infinite-dimensional) features
    induced by the kernel before computing the Gram matrix.
    Args:
        gram (np.ndarray): A num_examples x num_examples symmetric matrix.
        unbiased (bool): Whether to apply the unbiased HSIC correction
            (recommended when N is small, e.g. N=48).
    Returns:
        np.ndarray: Centered Gram matrix of the same shape.
    """
    if not np.allclose(gram, gram.T):
        raise ValueError("Input must be a symmetric matrix.")
    gram = gram.copy()
    if unbiased:
        n = gram.shape[0]
        np.fill_diagonal(gram, 0)
        means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
        means -= np.sum(means) / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        np.fill_diagonal(gram, 0)
    else:
        means = np.mean(gram, 0, dtype=np.float64)
        means -= np.mean(means) / 2
        gram -= means[:, None]
        gram -= means[None, :]
    return gram



def compute_cka(X, Y, unbiased=False):
    """Compute linear Centered Kernel Alignment (CKA) between two activation matrices.
    Uses the linear kernel (K = X X^T) and the HSIC-based formulation from
    Kornblith et al., "Similarity of Neural Network Representations Revisited"
    (ICML 2019, https://arxiv.org/abs/1905.00414).
    CKA = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))
    where HSIC is estimated via the Frobenius inner product of the centered
    Gram matrices:  HSIC(K, L) = <K_c, L_c>_F
    Args:
        X (np.ndarray | torch.Tensor): [N, D1] activation matrix (e.g. model embeddings).
        Y (np.ndarray | torch.Tensor): [N, D2] activation matrix (e.g. SPoSE targets).
        unbiased (bool): Use the unbiased HSIC estimator. Recommended for
            small N (e.g. N=48). CKA may still carry a small bias.
    Returns:
        float: CKA score in [0, 1], where 1 means the two representations
            are identical up to orthogonal transformation and isotropic scaling.
    """
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.detach().cpu().numpy()

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X and Y must have the same number of examples, "
            f"got {X.shape[0]} and {Y.shape[0]}."
        )

    gram_x = X @ X.T  # [N, N] linear kernel
    gram_y = Y @ Y.T  # [N, N] linear kernel

    gram_x = center_gram(gram_x, unbiased=unbiased)
    gram_y = center_gram(gram_y, unbiased=unbiased)

    scaled_hsic = gram_x.ravel().dot(gram_y.ravel())
    normalization_x = np.linalg.norm(gram_x)
    normalization_y = np.linalg.norm(gram_y)
    return float(scaled_hsic / (normalization_x * normalization_y))


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
        and categories is not None
        and annotations_file is not None
        and "concept" in pd.read_csv(annotations_file, nrows=1).columns
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
    log_file = os.path.join(
        config['inference_save_dir'],
        f'inference_log_{config.get("dataset", "unknown")}_{config["evaluation_type"]}.txt',
    )
    logger = setup_logger(log_file)
    logger.info("Run timestamp: %s", timestamp)

    # Persist the config snapshot alongside outputs
    snapshot_path = Path(config['inference_save_dir']) / "inference_config_snapshot.yaml"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    logger.info("Saved config snapshot to %s", snapshot_path)
    
    ## SET DEVICE
    if config['cuda'] == -1:
        device = torch.device("cuda")
    elif config['cuda'] == 0:
        device = torch.device("cuda:0")
    elif config['cuda'] == 1:
        device = torch.device("cuda:1")
    else:
        raise ValueError("Unsupported cuda setting. Expected -1, 0, or 1.")

    ## INITIALIZE MODEL & LOAD WEIGHTS
    if training_mode == 'scratch':
        # Build the same timm ViT/ResNet used during training
        model = build_model(
            architecture=config.get('architecture'),
            pretrained=False,
            clip_hba_backbone=None,
            vision_layers=None,
            transformer_layers=None,
            rank=None,
            cuda=cuda_cfg,
            device=device,
            wandb_watch_model=False,
            wandb_log_freq=None,
            num_classes=config.get('num_classes', 1000),
        )

        # Resolve checkpoint path and derive epoch label
        weights_path = Path(config['model_weights_path'])
        if weights_path.is_file():
            epoch = weights_path.stem.replace('epoch', '').replace('_model', '')
            load_model_checkpoint(model, str(weights_path), logger)


        else:
            # Directory mode: look inside model_checkpoints/ for epoch*_model.pth files
            ckpt_dir = (
                weights_path
                if weights_path.name == 'model_checkpoints'
                else weights_path / 'model_checkpoints'
            )
            ckpt_files = sorted(ckpt_dir.glob('epoch*_model.pth'))
            if not ckpt_files:
                raise FileNotFoundError(f"No model checkpoints found under {ckpt_dir}")
            requested_epoch = config.get('epoch')
            if requested_epoch is not None:
                epoch = str(requested_epoch)
                candidate = ckpt_dir / f'epoch{epoch}_model.pth'
                if not candidate.exists():
                    raise FileNotFoundError(f"Requested epoch checkpoint not found: {candidate}")
                load_model_checkpoint(model, str(candidate), logger)

            else:
                latest = ckpt_files[-1]
                epoch = latest.stem.replace('epoch', '').replace('_model', '')
                load_model_checkpoint(model, str(latest), logger)

        model.to(device)

    else:
        # Fine-tuning: CLIP-HBA with DoRA adapters
        model = initialize_cliphba_model(
            backbone_name=config['backbone'],
            classnames=classnames66,
            vision_layers=config['vision_layers'],
            transformer_layers=config['transformer_layers'],
            rank=config['rank'],
            dora_dropout=0.1,
            logger=logger,
        )

        weights_path = Path(config['model_weights_path'])
        loaded_direct = False
        if weights_path.is_file():
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            epoch = weights_path.stem.replace('epoch', '').replace('_dora_params', '')
            loaded_direct = True

        else:
            dora_dir = (
                weights_path
                if weights_path.name == "dora_params"
                else weights_path / "dora_params"
            )
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

        model = model.to(device)
        if not loaded_direct:
            load_dora_checkpoint(model, checkpoint_root=checkpoint_root, epoch=epoch, map_location=device)

    model.eval()

    ## SET EVALUATION TYPE
    if "evaluation_type" not in config:
        logger.error("evaluation_type is not set in the config file.")
        raise ValueError("evaluation_type must be set in the config file.")
    evaluation_type = config["evaluation_type"]

    logger.info("\n=== Processing epoch %s ===", epoch)

    ## DISPATCH TO EVALUATION LOGIC
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
            distance_metric=config["model_rdm_distance_metric"],
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
                    f"Reference RDM for ROI '{reference_rdm_name}' has shape "
                    f"{reference_rdm_upper_tri_vector.shape} but model RDM has shape "
                    f"{model_rdm_upper_tri_vector.shape}"
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
            logger.info(
                "RSA (%s) score for target RDM '%s': %.4f (p=%.4g) using distance metric '%s'",
                rsa_similarity_metric, reference_rdm_name, rho, p_value,
                config["model_rdm_distance_metric"],
            )

        results = rsa_results
        results_json_path = (
            Path(config['inference_save_dir'])
            / f"inference_results_{config['dataset']}_{evaluation_type}_epoch{epoch}.json"
        )
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        logger.info("Saved inference results (JSON) to %s", results_json_path)

    elif evaluation_type == "imagenet_val":
        # ImageNet top-1 / top-5 classification accuracy (scratch models only)
        if training_mode != 'scratch':
            raise ValueError(
                "evaluation_type='imagenet_val' requires training_mode='scratch'. "
                "Use 'behavioral' or 'neural' for finetune models."
            )
        from src.data.imagenet_dataset import ImagenetDataset

        val_dataset = ImagenetDataset(img_dir=config['img_dir'], split='val')
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            pin_memory=(device.type == 'cuda'),
        )
        logger.info("Running ImageNet val evaluation on %d images ...", len(val_dataset))

        top1_correct = 0
        top5_correct = 0
        total = 0
        with torch.no_grad():
            for _image_names, images, labels in tqdm(val_loader, desc="ImageNet val"):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                top1_correct += (logits.argmax(dim=1) == labels).sum().item()
                top5_preds = logits.topk(5, dim=1).indices
                top5_correct += (top5_preds == labels.unsqueeze(1)).any(dim=1).sum().item()
                total += labels.size(0)

        top1_acc = top1_correct / total
        top5_acc = top5_correct / total
        score = top1_acc
        results = {
            "epoch": epoch,
            "evaluation_type": "imagenet_val",
            "top1_accuracy": top1_acc,
            "top5_accuracy": top5_acc,
            "total_samples": total,
        }
        logger.info(
            "ImageNet val — Epoch %s: Top-1 %.4f | Top-5 %.4f (%d samples)",
            epoch, top1_acc, top5_acc, total,
        )
        results_json_path = (
            Path(config['inference_save_dir'])
            / f"inference_results_imagenet_val_epoch{epoch}.json"
        )
    
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        logger.info("Saved inference results (JSON) to %s", results_json_path)

    elif evaluation_type == "triplet" and config.get('dataset') == 'nights':
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
        logger.info("Triplet score for '%s' dataset: %.4f", config['dataset'], score)
        
    elif evaluation_type == "cka":
        # Extract CLIP-HBA last-layer activations for the 48 inference images
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
        model_activations = embedding_outputs["embeddings"]  # [N, 66] tensor

        # Load the SPoSE target embeddings from the same annotation CSV.
        # The CSV has index_col=0, column 0 = image_name, columns 1-66 = embedding dims,
        # in the same row order as the dataset, so alignment with model_activations is exact.
        annotations_df = pd.read_csv(config['inference_annotations_file'], index_col=0)
        spose_embeddings = annotations_df.iloc[:, 1:].values.astype(np.float64)  # [N, 66]

        n_images = model_activations.shape[0]
        if n_images != spose_embeddings.shape[0]:
            raise ValueError(
                f"Mismatch between number of model activations ({n_images}) "
                f"and SPoSE embeddings ({spose_embeddings.shape[0]})."
            )

        unbiased = config.get('cka_unbiased', False)
        cka_score = compute_cka(model_activations, spose_embeddings, unbiased=unbiased)
        score = cka_score

        results = {
            "epoch": epoch,
            "evaluation_type": "cka",
            "dataset": config['dataset'],
            "cka_score": cka_score,
            "unbiased": unbiased,
            "n_images": n_images,
        }
        logger.info(
            "CKA score (epoch %s, n=%d, unbiased=%s): %.4f",
            epoch, n_images, unbiased, cka_score,
        )
        results_json_path = (
            Path(config['inference_save_dir'])
            / f"inference_results_{config['dataset']}_cka_epoch{epoch}.json"
        )
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        logger.info("Saved CKA results (JSON) to %s", results_json_path)

    else:
        raise ValueError(
            f"Unsupported combination of dataset '{config.get('dataset', 'unknown')}' "
            f"and evaluation_type '{evaluation_type}'."
        )

    if score is None:
        logger.info("Score for epoch %s (%s): None", epoch, evaluation_type)
    else:
        logger.info("Score for epoch %s (%s): %.4f", epoch, evaluation_type, score)

    return {
        "epoch": epoch,
        "score": score,
        "results": results,
    }
