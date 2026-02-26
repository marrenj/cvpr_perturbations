"""
CLIP-HBA Training Script with DoRA Adaptation

Trains CLIP-based HBA models using DoRA (Weight-Decomposed
Low-Rank Adaptation) on behavioral similarity data. Supports perturbation strategies
and checkpoint resumption for reproducible experiments.
"""

import os
import random
import csv
import yaml
import wandb
from tqdm import tqdm
from numpy.random import set_state as np_set_state
import torch 
import torch.nn as nn
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, random_split, Subset
from torch.nn import DataParallel
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)

from src.training.test_loop import evaluate_model
from src.utils.save_random_states import save_random_states
from src.models.clip_hba.clip_hba_utils import save_dora_parameters
from src.inference.inference_core import (
    compute_model_rdm,
    compute_rdm_similarity,
)
from src.models.factory import build_model
from src.utils.seed import seed_everything
from src.utils.logging import setup_logger
from src.utils.count_parameters import count_trainable_parameters
from src.models.clip_hba.clip_hba_utils import load_dora_checkpoint
from src.data.things_dataset import ThingsBehavioralDataset
from src.data.imagenet_dataset import ImagenetDataset
from src.perturbations.perturbation_utils import choose_perturbation_strategy


# =============================================================================
# Helpers
# =============================================================================

def build_optimizer(model, config):
    """
    Build an optimizer from configuration.

    Supported optimizers (``opt`` key):
    * ``'sgd'``   – SGD with momentum.
    * ``'adamw'`` – AdamW.

    Args:
        model:  Model whose parameters to optimise.
        config: Configuration dictionary.

    Returns:
        torch.optim.Optimizer
    """
    opt_name    = str(config.get('opt')).lower()
    lr          = float(config['lr'])
    weight_decay = float(config.get('weight_decay', 0.0))

    if opt_name == 'sgd':
        momentum = float(config.get('momentum', 0.9))
        return SGD(
            model.parameters(), 
            lr=lr, 
            momentum=momentum, 
            weight_decay=weight_decay,
        )
    elif opt_name == 'adamw':
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    else:
        raise ValueError(
            f"Optimizer '{opt_name}' not supported. Use 'sgd' or 'adamw'."
        )


def build_scheduler(optimizer, config, epochs: int):
    """
    Build a learning-rate scheduler from configuration.

    Supported schedulers (``lr_scheduler`` key):
    * ``'cosineannealinglrwithwarmup'`` – Linear warmup followed by cosine
      annealing. Warmup length is controlled by ``lr_warmup_duration``
      (e.g. ``'5ep'``).
    * ``'cosineannealinglr'``           – Cosine annealing with no warmup.
    * ``'none'`` / absent               – No scheduler (constant LR).

    Args:
        optimizer: Configured optimizer.
        config:    Configuration dictionary.
        epochs:    Total number of training epochs.

    Returns:
        torch.optim.lr_scheduler._LRScheduler | None
    """
    name = str(config.get('lr_scheduler', 'none')).lower().strip()
    if not name or name == 'none':
        return None

    warmup_epochs = int(config.get('lr_warmup_duration'))


    if name == 'cosineannealinglrwithwarmup':
        cosine_epochs = max(1, epochs - warmup_epochs)
        if warmup_epochs > 0:
            warmup_sched = LinearLR(
                optimizer,
                start_factor=1e-6,   # near-zero start
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine_sched = CosineAnnealingLR(
                optimizer, T_max=cosine_epochs, eta_min=0
            )
            return SequentialLR(
                optimizer,
                schedulers=[warmup_sched, cosine_sched],
                milestones=[warmup_epochs],
            )
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

    if name == 'cosineannealinglr':
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

    raise ValueError(
        f"LR scheduler '{name}' not supported. "
        "Use 'cosineannealinglrwithwarmup', 'cosineannealinglr', or 'none'."
    )



# =============================================================================
# Path Setup
# =============================================================================


def get_wandb_tags(config):
    """
    Generate tags for wandb run based on configuration.

    Works for both training modes (scratch and finetune).
    
    Args:
        config: Configuration dictionary
        
    Returns:
        list: List of tags
    """
    training_mode = config.get('training_mode')

    # Use clip_hba_backbone for finetune, architecture for scratch
    backbone_tag = (
        config.get('clip_hba_backbone')
        if training_mode == 'finetune'
        else config.get('architecture')
    )

    tags = [
        f"training_mode_{training_mode}",
        f"backbone_{backbone_tag}",
        f"perturb_type_{config['perturb_type']}",
        f"dataset_type_{config['dataset_type']}",
        f"init_seed_{config['random_seed']}",
    ]

    if training_mode == 'finetune':
        tags.append(f"rank_{config.get('rank')}")

    if config.get('behavioral_rsa'):
        tags.append("behavioral_rsa")
    
    perturb_type = str(config.get('perturb_type')).lower()
    if perturb_type != 'none':
        tags.append(f"perturb_epoch_{config.get('perturb_epoch')}")
        tags.append(f"perturb_length_{config.get('perturb_length')}")
        tags.append(f"perturb_seed_{config.get('perturb_seed')}")
    
    if config.get('wandb_tags'):
        tags.extend(config.get('wandb_tags'))
    
    return tags


def get_run_name(config):
    """
    Build a deterministic run name shared by wandb and local save paths.'

    Works for both training modes (scratch and finetune).
    """
    def _sanitize(name: str) -> str:
        # Make a filesystem-safe, lowercase token (replace slashes/spaces)
        return name.replace("-", "_").replace("/", "_").replace(" ", "_").lower()

    if config.get('wandb_run_name'):
        return config['wandb_run_name']
    
    training_mode = config.get('training_mode')

    if training_mode == 'finetune':
        backbone_token = _sanitize(config.get('clip_hba_backbone'))
        rank_token = f"rank{config.get('rank')}"
    else:
        backbone_token = _sanitize(config.get('architecture'))
        rank_token = "scratch"

    perturb_type = str(config.get('perturb_type', 'none'))
    normalized_ptype = perturb_type.lower()

    if normalized_ptype == 'none':
        return (
            f"{backbone_token}_"
            f"{rank_token}_"
            f"perturb-type-none_"
            f"init-seed{config.get('random_seed')}"
            f"behavioral-rsa-{config.get('behavioral_rsa')}"
        )

    return (
        f"{backbone_token}_"
        f"{rank_token}_"
        f"perturb-type-{perturb_type}_"
        f"epoch{config.get('perturb_epoch')}_"
        f"length{config.get('perturb_length')}_"
        f"perturb-seed{config.get('perturb_seed')}_"
        f"init-seed{config.get('random_seed')}"
        f"behavioral-rsa-{config.get('behavioral_rsa')}"
    )


def init_wandb(config, resume_epoch=0):
    """
    Initialize Weights & Biases run with proper configuration.
    
    Args:
        config: Configuration dictionary
        resume_epoch: Epoch to resume from
        
    Returns:
        wandb.Run: Weights & Biases run
    """
    resume_mode = "allow" if resume_epoch > 0 else None
    run_id = config.get('wandb_run_id', None)

    # If project/entity are missing or blank, run wandb in offline mode
    project = config.get('wandb_project')
    entity = config.get('wandb_entity')
    offline_mode = (not project) or (entity is None or entity == "")
    if offline_mode:
        os.environ["WANDB_MODE"] = "offline"
        project = project or "offline-run"

    # Create descriptive run name (shared with filesystem path)
    run_name = get_run_name(config)
    
    run = wandb.init(
        project=project,
        entity=entity if not offline_mode else None,
        name=run_name,
        id=run_id,
        resume=resume_mode,
        config=config,
        tags=get_wandb_tags(config),
        notes=config.get('wandb_notes', ''),
        save_code=True,
    )
    
    config['wandb_run_id'] = run.id
    config['wandb_run_name'] = run.name or run_name
    return run


def log_model_architecture(model, logger):
    """
    Log model architecture details to wandb.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    wandb.config.update({
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'trainable_percentage': 100 * trainable_params / total_params if total_params > 0 else 0
    })
    
    trainable_layers = [name for name, param in model.named_parameters() if param.requires_grad]
    wandb.config.update({'trainable_layers': trainable_layers})
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")


def setup_paths(config):
    """
    Create and return all necessary directory paths for training outputs.
    
    Args:
        save_path: Root directory for saving all training artifacts
        
    Returns:
        tuple: (save_path, training_results_save_path, random_state_save_path)
    """
    os.makedirs(config['save_path'], exist_ok=True)
    training_results_save_path = os.path.join(config['save_path'], 'training_res.csv')
    random_state_save_path = os.path.join(config['save_path'], 'random_states')
    checkpoints_save_path = os.path.join(config['save_path'], 'model_checkpoints')

    # Check if files/directories already exist and ask for overwrite permission
    for path, description in [
        (training_results_save_path, 'training results file'),
        (random_state_save_path, 'random state save directory'),
        (checkpoints_save_path, 'model checkpoints save directory'),
    ]:
        if os.path.exists(path):
            while True:
                response = input(f"{description.capitalize()} '{path}' already exists, are you sure you want to rewrite it? (Yes/No): ")
                if response.lower() in ["yes", "y"]:
                    break
                elif response.lower() in ["no", "n"]:
                    raise FileExistsError(f"Aborted because {description} '{path}' already exists.")
                else:
                    print("Please answer 'Yes' or 'No'.")
    return config['save_path'], training_results_save_path, random_state_save_path


# =============================================================================
# Dataset Setup
# =============================================================================

def setup_dataset(config, logger):
    """
    Initialize and split dataset according to configuration.

    For ``training_mode='scratch'`` with ImageNet the dataset already ships
    with canonical train/val directories, so no random split is performed.
    For ``training_mode='finetune'`` the existing random-split logic is used.
    
    Args:
        config: Configuration dictionary containing dataset parameters
        logger: Logger instance for tracking dataset operations
        
    Returns:
        tuple: (train_dataset, val_dataset, split_info, global_target_stats, inference_dataset)
    """
    training_mode = config.get('training_mode')

    if training_mode == 'scratch':
        if config.get('dataset_type') != 'imagenet':
            raise ValueError(
                "training_mode='scratch' currently only supports dataset_type='imagenet'"
            )

        if config.get('perturb_type') == 'random_target':
            raise ValueError(
                "perturb_type='random_target' requires continuous embedding targets; "
                "and is only compatible with training_mode='finetune'."
            )

        train_dataset = ImagenetDataset(img_dir=config.get('img_dir'), split='train')
        val_dataset = ImagenetDataset(img_dir=config.get('img_dir'), split='val')
        split_info = {'split_type': 'imagenet_predefined'}

        logger.info(
            f"ImageNet predefined splits: "
            f"{len(train_dataset):,} train / {len(val_dataset):,} val images"
        )

        wandb.config.update({
            'dataset_size': len(train_dataset) + len(val_dataset),
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'train_portion': None,
        })

        # Build a THINGS inference dataset for behavioral RSA if paths are provided
        rsa_annotations_file = config.get('rsa_annotations_file')
        rsa_img_dir = config.get('rsa_things_img_dir')
        if config.get('behavioral_rsa') and rsa_annotations_file and rsa_img_dir:
            inference_dataset = ThingsBehavioralDataset(
                img_annotations_file=rsa_annotations_file,
                img_dir=rsa_img_dir,
            )
            logger.info(f"Inference dataset (THINGS): {len(inference_dataset):,} images")
        else:
            inference_dataset = None

        return train_dataset, val_dataset, split_info, (None, None), inference_dataset

    if config.get('dataset_type') == 'things':
        dataset = ThingsBehavioralDataset(
            img_annotations_file=config.get('img_annotations_file'),
            img_dir=config.get('img_dir'),
        )

    elif config.get('dataset_type') == 'imagenet':
        raise ValueError(
            "training_mode='finetune' currently only supports dataset_type='things'"
        )

    else:
        raise ValueError(f"Dataset type {config.get('dataset_type')} not supported")

    if config.get('behavioral_rsa'):
        inference_dataset = ThingsBehavioralDataset(
            img_annotations_file=config.get('rsa_annotations_file'),
            img_dir=config.get('rsa_things_img_dir'),
        )
        
    global_target_mean = None
    global_target_std = None
    
    if config.get('perturb_type') == 'random_target':
        embeddings = dataset.annotations.iloc[:, 1:].values.astype('float32')
        global_target_mean = torch.tensor(np.mean(embeddings), dtype=torch.float32)
        global_target_std = torch.tensor(np.std(embeddings), dtype=torch.float32)
    
    # Determine if we should reuse an existing split
    baseline_checkpoint_path = config.get('baseline_checkpoint_path')
    split_indices_path = None
    if baseline_checkpoint_path:
        split_candidate = os.path.join(
            baseline_checkpoint_path, 'random_states', 'dataset_split_indices.pth'
        )
        if os.path.isfile(split_candidate):
            split_indices_path = split_candidate
    
    # Load or create dataset split
    if split_indices_path:
        if not os.path.isfile(split_indices_path):
            raise FileNotFoundError(f"dataset_split_indices_path not found: {split_indices_path}")
        split_info = torch.load(split_indices_path)
        train_dataset = Subset(dataset, split_info['train_indices'])
        val_dataset  = Subset(dataset, split_info['val_indices'])
        logger.info(f"Loaded dataset split indices from {split_indices_path}")
    else:
        train_size = int(config['train_portion'] * len(dataset))
        val_size  = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        split_info = {
            'train_indices': list(train_dataset.indices),
            'val_indices':  list(val_dataset.indices),
            'random_seed':   config['random_seed'],
            'train_portion': config['train_portion']
        }

    wandb.config.update({
        'dataset_size':  len(dataset),
        'train_size':    len(train_dataset),
        'val_size':      len(val_dataset),
        'train_portion': config['train_portion']
    })
    
    return train_dataset, val_dataset, split_info, (global_target_mean, global_target_std), inference_dataset


def create_dataloaders(train_dataset, val_dataset, inference_dataset, config):
    """
    Create DataLoader instances for training and validation.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Configuration dictionary with batch_size and random_seed
        
    Returns:
        tuple: (train_loader, val_loader, inference_loader, dataloader_generator)
    """
    cuda_available = torch.cuda.is_available()

    # num_workers: parallel image loading workers per DataLoader.
    # 4 per GPU is a good starting point; override via config if needed.
    num_gpus = torch.cuda.device_count() if cuda_available else 1
    num_workers = config.get('num_workers', 4 * max(num_gpus, 1))

    # pin_memory: pages host tensors into non-pageable memory so the
    # CUDA DMA engine can transfer them without a CPU copy.
    pin_memory = cuda_available

    # persistent_workers: keeps worker processes alive between epochs,
    # avoiding the fork/init overhead at the start of every epoch.
    persistent_workers = num_workers > 0

    # prefetch_factor: how many batches each worker pre-loads ahead of
    # the training loop consuming them.
    prefetch_factor = config.get('prefetch_factor', 2) if num_workers > 0 else None

    # Create generator for reproducible DataLoader shuffling
    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(config['random_seed'])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        generator=dataloader_generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    if inference_dataset is not None:
        inference_loader = DataLoader(
            inference_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=min(num_workers, 4),
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
    else:
        inference_loader = None

    return train_loader, val_loader, inference_loader, dataloader_generator


# =============================================================================
# Model Setup
# =============================================================================


def get_device(cuda_config, logger):
    """
    Get the appropriate device based on CUDA configuration.
    
    Args:
        cuda_config: CUDA device configuration (-1 for all, 0/1 for specific GPU)
        logger: Logger instance for logging messages
        
    Returns:
        torch.device: Configured device
    """
    if torch.cuda.is_available():
        if cuda_config == -1:
            device = torch.device("cuda")
            logger.info(f"Using all {torch.cuda.device_count()} GPUs")
        elif cuda_config == 0:
            device = torch.device("cuda:0")
            logger.info(f"Using GPU 0")
        elif cuda_config == 1:
            device = torch.device("cuda:1")
            logger.info(f"Using GPU 1")
        else:
            device = torch.device("cuda")
            logger.info(f"Using all {torch.cuda.device_count()} GPUs")
    else:
        device = torch.device("cpu")
        logger.info(f"Using CPU")
    logger.info(f"Using device: {device}")
    return device


def compute_behavioral_rsa(
    model,
    data_loader,
    device,
    annotations_file,
    distance_metric,
    similarity_metric,
    logger=None,
    epoch_tag=None,
):
    """
    Compute behavioral RSA between model embeddings and target embeddings
    on the provided data_loader.

    For classification models (e.g. ResNet50 trained from scratch), we extract
    penultimate-layer features via ``forward_features`` when available (all
    timm models support this), falling back to the full forward pass otherwise.
    For DataParallel-wrapped models the underlying module is used.
    """
    log = logger.info if logger else print

    # Unwrap DataParallel to access timm's forward_features
    base_model = model.module if hasattr(model, 'module') else model
    use_features = hasattr(base_model, 'forward_features')
    if use_features:
        log("Behavioral RSA: using penultimate-layer features (forward_features)")

    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for _, (_, images, target) in enumerate(data_loader):
            images = images.to(device, non_blocking=True)
            if use_features:
                feats = base_model.forward_features(images)
                # Collapse any spatial / token dims down to a 1-D vector per image:
                #   [B, C, H, W]  → global avg pool → [B, C]   (CNN)
                #   [B, tokens, D] → mean over tokens → [B, D]  (ViT)
                if feats.dim() == 4:
                    feats = feats.mean(dim=(2, 3))
                elif feats.dim() == 3:
                    feats = feats.mean(dim=1)
                preds.append(feats.cpu())
            else:
                preds.append(model(images).cpu())
            targets.append(target.cpu())

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)

    model_rdm = compute_model_rdm(
        preds, dataset_name="things", annotations_file=annotations_file,
        categories=None, distance_metric=distance_metric
    )
    target_rdm = compute_model_rdm(
        targets, dataset_name="targets", annotations_file=None,
        categories=None, distance_metric=distance_metric
    )

    model_ut_idx = np.triu_indices_from(model_rdm, k=1)
    target_ut_idx = np.triu_indices_from(target_rdm, k=1)
    model_ut = model_rdm[model_ut_idx]
    target_ut = target_rdm[target_ut_idx]

    corr, pval = compute_rdm_similarity(model_ut, target_ut, similarity_metric)
    log(f"Behavioral RSA: corr={corr:.4f}, p={pval:.4g}")
    wandb.log({
        'rsa_behavioral_corr': corr,
        'rsa_behavioral_p': pval,
        'epoch': epoch_tag,
    })
    return corr, pval


# =============================================================================
# Checkpoint & State Management
# =============================================================================


def save_model_checkpoint(model, checkpoint_dir: str, epoch: int, log_fn=None):
    """
    Save a full model state_dict for from-scratch training.
 
    Handles ``DataParallel`` wrappers transparently by saving the underlying
    ``module`` state_dict so checkpoints are portable across GPU configs.
 
    Args:
        model:          Model (possibly DataParallel-wrapped) to checkpoint.
        checkpoint_dir: Directory to write the file into.
        epoch:          Current epoch index (used in the filename).
        log_fn:         Callable for logging; defaults to ``print``.
    """
    log = log_fn or print
    os.makedirs(checkpoint_dir, exist_ok=True)
    state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    checkpoint_file = os.path.join(checkpoint_dir, f"epoch{epoch}_model.pth")
    torch.save(state_dict, checkpoint_file)
    log(f"Model checkpoint saved: {checkpoint_file}")


def save_model_checkpoing(model, checkpoint_dir: str, epoch: int, log_fn=None):
    """
    Save a full model state_dict for from-scratch training.
 
    Handles ``DataParallel`` wrappers transparently by saving the underlying
    ``module`` state_dict so checkpoints are portable across GPU configs.
 
    Args:
        model:          Model (possibly DataParallel-wrapped) to checkpoint.
        checkpoint_dir: Directory to write the file into.
        epoch:          Current epoch index (used in the filename).
        log_fn:         Callable for logging; defaults to ``print``.
    """
    log = log_fn or print
    os.makedirs(checkpoint_dir, exist_ok=True)
    state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    checkpoint_file = os.path.join(checkpoint_dir, f"epoch{epoch}_model.pth")
    torch.save(state_dict, checkpoint_file)
    log(f"Model checkpoint saved: {checkpoint_file}")


def load_checkpoint_and_states(
    architecture, model, optimizer, dataloader_generator, checkpoint_path,
    random_state_path, logger=None, scheduler=None,
    ):
    """
    Load model weights and restore all random states for reproducibility.
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to restore state
        dataloader_generator: Generator for DataLoader shuffling
        checkpoint_path: Path to checkpoint directory
        random_state_path: Path to random state file
        logger: Logger instance
        training_mode: Training mode ('scratch' or 'finetune')
        scheduler: Scheduler to restore state
    Returns:
        int: Next epoch to resume training from
    """
    epoch_num = int(random_state_path.split('epoch')[-1].split('_')[0])

    if architecture == 'CLIP-HBA':
        # Load DoRA weights
        loaded_dora_path = load_dora_checkpoint(
            model,
            checkpoint_root=checkpoint_path,
            epoch=epoch_num,
            strict=False,
        )
        logger.info(f"Loaded DoRA parameters from: {loaded_dora_path}")

    else:
        model_file = os.path.join(
            checkpoint_path, 'model_checkpoints', f'epoch{epoch_num}_model.pth'
        )
        load_model_checkpoint(model, model_file, logger)
        logger.info(f"Loaded model weights from: {model_file}")
    
    # Restore all RNG and optimizer states 
    if not os.path.isfile(random_state_path):
        raise FileNotFoundError(f"Missing random state file: {random_state_path}")
    
    state_payload = torch.load(random_state_path, map_location='cpu', weights_only=False)
    
    # Restore RNG states
    torch.set_rng_state(state_payload['torch_rng_state'])
    np_set_state(state_payload['numpy_rng_state'])
    random.setstate(state_payload['python_rng_state'])
    
    # Restore CUDA RNG states if available
    if torch.cuda.is_available():
        if 'cuda_rng_state' in state_payload:
            torch.cuda.set_rng_state(state_payload['cuda_rng_state'])
        if 'cuda_rng_state_all' in state_payload:
            torch.cuda.set_rng_state_all(state_payload['cuda_rng_state_all'])
    
    # Restore DataLoader generator state
    if dataloader_generator is not None and 'dataloader_generator_state' in state_payload:
        dataloader_generator.set_state(state_payload['dataloader_generator_state'])
    
    # Restore optimizer state
    if 'optimizer_state_dict' in state_payload:
        optimizer.load_state_dict(state_payload['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in state_payload:
        scheduler.load_state_dict(state_payload['scheduler_state_dict'])

    logger.info(f"Restored optimizer and RNG states from {random_state_path}")
    
    return epoch_num + 1


def handle_checkpoint_resumption(config, model, optimizer, dataloader_generator, 
logger, scheduler=None):
    """
    Handle checkpoint loading for either explicit resumption or baseline loading.
    
    Args:
        config: Configuration dictionary
        model: Model instance
        optimizer: Optimizer instance
        dataloader_generator: DataLoader generator
        logger: Logger instance
        scheduler: Optional LR scheduler to restore.

    Returns:
        int: Epoch to start/resume from
    """
    resume_epoch = 0
    architecture = config.get("architecture")
    resume_checkpoint_path = config.get("resume_checkpoint_path")
    resume_from_epoch = config.get("resume_from_epoch")
    
    # Case 1: Explicit resumption from a specific checkpoint
    if resume_checkpoint_path and resume_from_epoch is not None:
        random_state_file = os.path.join(
            resume_checkpoint_path,
            'random_states',
            f'epoch{resume_from_epoch}_random_states.pth',
        )
        resume_epoch = load_checkpoint_and_states(
            architecture, model, optimizer, dataloader_generator,
            resume_checkpoint_path, random_state_file, logger,
            scheduler=scheduler,
        )
    
    # Case 2: Load baseline checkpoint before applying perturbation
    elif config.get('perturb_epoch') > 0:
        baseline_epoch = config['perturb_epoch'] - 1
        baseline_checkpoint_path = config.get('baseline_checkpoint_path')
        
        if not baseline_checkpoint_path:
            raise ValueError(
                "baseline_checkpoint_path must be provided when perturb_epoch > 0."
            )
        
        random_state_file = os.path.join(
            baseline_checkpoint_path,
            'random_states',
            f'epoch{baseline_epoch}_random_states.pth',
        )
        resume_epoch = load_checkpoint_and_states(
            architecture, model, optimizer, dataloader_generator, baseline_checkpoint_path,
            random_state_file, logger, scheduler=scheduler,
        )
        logger.info(f"Loaded baseline checkpoint from epoch {baseline_epoch}")
    
    return resume_epoch


# =============================================================================
# Training Loop
# =============================================================================


def train_one_epoch(model, train_loader, device, optimizer, criterion, 
                   perturb_strategy, epoch_idx, logger, log_interval=10, debug_logging=False):
    """
    Train model for a single epoch.
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        device: Device to run training on
        optimizer: Optimizer instance
        criterion: Loss function
        perturb_strategy: Perturbation strategy to apply
        epoch_idx: Current epoch index
        logger: Logger instance
        log_interval: Interval to log metrics with wandb
        debug_logging: Whether to log debug information to the console
    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    batch_losses = []
    
    # Log if perturbation is active this epoch
    if perturb_strategy.is_active_epoch(epoch_idx):
        logger.info(f"Applying {perturb_strategy.__class__.__name__} perturbation during epoch {epoch_idx}")
        wandb.log({'perturbation_active': 1, 'epoch': epoch_idx})
    else:
        wandb.log({'perturbation_active': 0, 'epoch': epoch_idx})
    
    progress_bar = tqdm(
        enumerate(train_loader), 
        total=len(train_loader), 
        desc=f"Epoch {epoch_idx}"
    )
    
    for batch_idx, (image_names, images, targets) in progress_bar:
        # non_blocking=True overlaps H→D transfer with GPU compute when
        # pin_memory=True is set on the DataLoader.
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if debug_logging is True and batch_idx == 0:
            print("\n=== DEBUG: BEFORE PERTURBATION ===")
            print("targets (first 10 values in first 10 rows):", targets[:10, :10])
            print("targets shape:", targets.shape)
            print("images (first 10):", images[:10])
            print("images shape:", images.shape)
            print("image_names (first 10):", image_names[:10])
            print("image_names shape:", len(image_names))

        images_before = images.clone()
        targets_before = targets.clone()
        
        # Apply perturbation if active
        images, targets = perturb_strategy.apply_to_batch(
            images, targets, device, epoch_idx, batch_idx
        )

        if debug_logging is True and batch_idx == 0:
            print("\n=== DEBUG: AFTER PERTURBATION ===")
            print("targets (first 10 values in first 10 rows):", targets[:10, :10])
            print("targets shape:", targets.shape)
            print("images (first 10):", images[:10])
            print("images shape:", images.shape)
            print("image_names (first 10):", image_names[:10])
            print("image_names shape:", len(image_names))

            # Numeric check: how much changed?
            delta_targets = (targets - targets_before).abs().mean()
            print("mean |target change|:", delta_targets.item())
            delta_images = (images - images_before).abs().mean()
            print("mean |image change|:", delta_images.item())
            same_rows = (targets == targets_before).all(dim=1).sum().item()
            print("num target vectors unchanged:", same_rows, "/", targets.shape[0])

        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        batch_loss = loss.item()
        total_loss += loss.item() * images.size(0)
        batch_losses.append(batch_loss)
        progress_bar.set_postfix({'loss': batch_loss})

        if batch_idx % log_interval == 0:
            wandb.log({
                'batch_loss': batch_loss,
                'batch': epoch_idx * len(train_loader) + batch_idx,
                'epoch': epoch_idx,
            })
    
    avg_train_loss = total_loss / len(train_loader.dataset)

    wandb.log({
        'train_loss': avg_train_loss,
        'train_loss_std': np.std(batch_losses),
        'epoch': epoch_idx,
    })

    return avg_train_loss


def save_epoch_results(epoch, train_loss, val_loss, training_results_save_path, rsa_corr=None, rsa_p=None):
    """
    Save training metrics to CSV file.
    
    Args:
        epoch: Current epoch number
        train_loss: Training loss
        val_loss: Validation loss
        training_results_save_path: Path to CSV file
        rsa_corr: Optional behavioral RSA correlation
        rsa_p: Optional p-value for behavioral RSA correlation
    """
    data_row = [epoch, train_loss, val_loss]
    if rsa_corr is not None and rsa_p is not None:
        data_row.extend([rsa_corr, rsa_p])
    
    with open(training_results_save_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data_row)


def train_model(
    model,
    train_loader,
    val_loader,
    inference_loader,
    device,
    optimizer,
    criterion,
    epochs,
    training_results_save_path,
    save_path,
    random_state_save_path,
    logger=None,
    early_stopping_patience=5,
    dataloader_generator=None,
    vision_layers=1,
    transformer_layers=1,
    perturb_strategy=None,
    start_epoch=0,
    behavioral_rsa=False,
    rsa_annotations_file=None,
    model_rdm_distance_metric="pearson",
    rsa_similarity_metric="spearman",
    debug_logging=False,
    scheduler=None,
    training_mode='finetune',
):
    """
    Main training loop with logging, checkpointing, and early stopping.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        inference_loader: Inference data loader
        device: Device for training
        optimizer: Optimizer instance
        criterion: Loss function
        epochs: Maximum number of epochs
        training_results_save_path: Path to save training metrics CSV
        save_path: Directory for checkpoints
        random_state_save_path: Directory for random state snapshots
        logger: Logger instance (uses print if None)
        early_stopping_patience: Epochs without improvement before stopping
        dataloader_generator: Generator for reproducible shuffling
        vision_layers: Number of vision transformer layers to save
        transformer_layers: Number of text transformer layers to save
        perturb_strategy: Perturbation strategy to apply
        start_epoch: Starting epoch (for resumption)
        behavioral_rsa: Whether to compute behavioral RSA
        rsa_annotations_file: Path to annotations file for behavioral RSA
        model_rdm_distance_metric: Distance metric for model RDM
        rsa_similarity_metric: Similarity metric for behavioral RSA
        debug_logging: Whether to log debug information to the console
    """
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # Use logger if provided, otherwise use print
    log = logger.info if logger else print
    
    # Initial evaluation
    log("*********************************")
    log("Evaluating initial model")
    best_val_loss = evaluate_model(model, val_loader, device, criterion)
    log(f"Initial Validation Loss: {best_val_loss:.4f}")
    # Optional behavioral RSA at initial evaluation
    if behavioral_rsa:
        try:
            compute_behavioral_rsa(
                model=model,
                data_loader=inference_loader,
                device=device,
                annotations_file=rsa_annotations_file,
                distance_metric=model_rdm_distance_metric,
                similarity_metric=rsa_similarity_metric,
                logger=logger,
                epoch_tag=-1,
            )
        except Exception as rsa_err:
            if logger:
                logger.warning(f"Initial behavioral RSA failed: {rsa_err}")
            else:
                print(f"Initial behavioral RSA failed: {rsa_err}")
    log("*********************************\n")

    wandb.log({
        'initial_val_loss': best_val_loss,
        'epoch': -1,
    })
    
    # Create directories for outputs
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.dirname(training_results_save_path), exist_ok=True)
    
    # Initialize CSV file (only if starting from scratch)
    if not (start_epoch > 0 and os.path.exists(training_results_save_path)):
        with open(training_results_save_path, 'w', newline='') as file:
            writer = csv.writer(file)
            header = ['epoch', 'train_loss', 'val_loss']
            if behavioral_rsa:
                header.extend(['rsa_behavioral_corr', 'rsa_behavioral_p'])
            writer.writerow(header)

    # Main training loop
    for epoch in range(start_epoch, epochs):
        # Train one epoch
        avg_train_loss = train_one_epoch(
            model, train_loader, device, optimizer, criterion,
            perturb_strategy, epoch, logger, debug_logging=debug_logging,
        )
        
        # Evaluate on validation set
        avg_val_loss = evaluate_model(model, val_loader, device, criterion)
        
        rsa_corr = None
        rsa_p = None

        # Log results
        print(f"Epoch {epoch}: Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        log(f"Epoch {epoch}: Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        
        if perturb_strategy.is_active_epoch(epoch):
            logger.info(f"*** Perturbation '{perturb_strategy.__class__.__name__}' was applied during epoch {epoch} ***")

        if scheduler is not None:
            sscheduler.step()

        wandb.log({
            'val_loss': avg_val_loss,
            'epoch': epoch,
            'learning_rate': optimizer.param_groups[0]['lr'],
        })

        # Optional behavioral RSA at end of epoch
        if behavioral_rsa:
            try:
                rsa_corr, rsa_p = compute_behavioral_rsa(
                    model=model,
                    data_loader=inference_loader,
                    device=device,
                    annotations_file=rsa_annotations_file,
                    distance_metric=model_rdm_distance_metric,
                    similarity_metric=rsa_similarity_metric,
                    logger=logger,
                    epoch_tag=epoch,
                )
            except Exception as rsa_err:
                if logger:
                    logger.warning(f"Behavioral RSA failed at epoch {epoch}: {rsa_err}")
                else:
                    print(f"Behavioral RSA failed at epoch {epoch}: {rsa_err}")
        
        # Save metrics to CSV
        save_epoch_results(epoch, avg_train_loss, avg_val_loss, training_results_save_path, rsa_corr, rsa_p)
        
        # Save random states for reproducibility
        save_random_states(
            optimizer, epoch, random_state_save_path, 
            dataloader_generator, logger=logger, scheduler=scheduler,
        )
        
        # Save model checkpoint
        if training_mode == 'scratch':
            checkpoint_dir = os.path.join(save_path, "model_checkpoints")
            save_model_checkpoint(model, checkpoint_dir, epoch, log_fn=log)
        else:
            checkpoint_dir = os.path.join(save_path, "dora_params")
            save_dora_parameters(
                model, checkpoint_dir, epoch, 
                vision_layers, transformer_layers, log_fn=log
            )
        log(f"Checkpoint saved for epoch {epoch}")

        save_to_wandb = (epoch < 20) or (epoch % 5 == 0)
        if save_to_wandb:  # Save every epoch for first 10, then every 5 epochs
            artifact = wandb.Artifact(
                name=f"model-checkpoint-epoch-{epoch}",
                type="model",
                description=f"Model checkpoint at epoch {epoch}",
                metadata={
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                }
            )
            artifact.add_dir(checkpoint_dir)
            artifact.add_file(training_results_save_path)
            wandb.log_artifact(artifact)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            wandb.run.summary["best_val_loss"] = best_val_loss
            wandb.run.summary["best_epoch"] = epoch
        else:
            epochs_no_improve += 1

        wandb.log({
            'epochs_no_improve': epochs_no_improve,
            'best_val_loss': best_val_loss,
            'epoch': epoch,
        })
        
        if epochs_no_improve == early_stopping_patience:
            log("\n\n*********************************")
            log(f"Early stopping triggered at epoch {epoch}")
            log("*********************************\n\n")
            wandb.run.summary["stopped_early"] = True
            wandb.run.summary["final_epoch"] = epoch
            break

    if epochs_no_improve < early_stopping_patience:
        wandb.run.summary["stopped_early"] = False
        wandb.run.summary["final_epoch"] = epochs

    # Log full training metrics CSV once at completion
    try:
        metrics_artifact = wandb.Artifact(
            name="training-results-final",
            type="metrics",
            description="Full training results CSV",
            metadata={
                'best_val_loss': best_val_loss,
                'final_epoch': wandb.run.summary.get("final_epoch", epochs),
            }
        )
        metrics_artifact.add_file(training_results_save_path)
        wandb.log_artifact(metrics_artifact)
    except Exception as art_err:
        if logger:
            logger.warning(f"Failed to log training results artifact: {art_err}")
        else:
            print(f"Failed to log training results artifact: {art_err}")


# =============================================================================
# Main Training Orchestration
# =============================================================================


def run_training_experiment(config):
    """
    Main entry point for running a training experiment.
    
    Supports two training modes controlled by ``config['training_mode']``:
 
    * ``'scratch'``  – Train a timm ViT or ResNet from scratch on ImageNet
      (CrossEntropyLoss, full model checkpoints, no DoRA).
    * ``'finetune'`` – Fine-tune a CLIP-HBA model with DoRA adaptation
      (MSELoss, DoRA checkpoints, optional behavioral RSA).
 
    Both modes support all perturbation types from
    ``perturbation_utils.py`` (except ``random_target`` which requires
    continuous embedding targets and is therefore only valid for
    ``training_mode='finetune'``).
    
    Args:
        config: Configuration dictionary containing all training parameters.
    """
    training_mode = config.get("training_mode")

    # Set random seed for reproducibility
    seed_everything(config['random_seed'])
    
    # Build run name once so wandb and filesystem stay in sync
    run_name = get_run_name(config)
    config['wandb_run_name'] = run_name
    
    # Align save_path with the run name (append unless already present)
    base_save_path = config.get('save_path', '')
    if base_save_path:
        normalized_base = os.path.normpath(base_save_path)
        if os.path.basename(normalized_base) != run_name:
            save_path = os.path.join(base_save_path, run_name)
        else:
            save_path = base_save_path
    else:
        save_path = run_name
    config['save_path'] = save_path
    
    # Setup logging
    os.makedirs(config['save_path'], exist_ok=True)
    perturb_type = config.get('perturb_type')
    if perturb_type:
        log_file = os.path.join(config['save_path'], f'training_log_{perturb_type}.txt')
    else:
        log_file = os.path.join(config['save_path'], 'training_log.txt')
    logger = setup_logger(log_file)
    
    logger.info("=" * 80)
    logger.info("Starting Training Run")
    logger.info(f"Training mode : {training_mode}")
    logger.info(f"Architecture  : {config.get('architecture', 'unknown')}")
    logger.info(f"Log file      : {log_file}")
    logger.info("=" * 80)
    
    # Persist config snapshots inside the run folder
    resolved_cfg_path = os.path.join(save_path, "resolved_config.yaml")
    input_cfg_path    = os.path.join(save_path, "training_config_snapshot.yaml")
    with open(resolved_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    # Input snapshot is the same available config dict; retained for provenance
    with open(input_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    
    # Setup paths
    save_path, training_results_save_path, random_state_save_path = setup_paths(config)

    # Initialize wandb before any wandb.config updates in dataset setup
    wandb_run = init_wandb(config, resume_epoch=0)
    
    # Setup dataset
    train_dataset, val_dataset, split_info, target_stats, inference_dataset = setup_dataset(config, logger)
    global_target_mean, global_target_std = target_stats
    
    # Save dataset split information
    split_file = os.path.join(random_state_save_path, 'dataset_split_indices.pth')
    os.makedirs(random_state_save_path, exist_ok=True)
    torch.save(split_info, split_file)
    logger.info(f"Dataset split info saved: {split_file}")
    
    # Create data loaders
    train_loader, val_loader, inference_loader, dataloader_generator = create_dataloaders(
        train_dataset, val_dataset, inference_dataset, config
    )
    
    # Setup device
    device = get_device(config['cuda'], logger)
    
    # Setup model
    model = build_model(
        architecture=config.get('architecture'),
        pretrained=config.get('pretrained'),
        clip_hba_backbone=config.get('clip_hba_backbone'),
        vision_layers=config.get('vision_layers'),
        transformer_layers=config.get('transformer_layers'),
        rank=config.get('rank'),
        cuda=config.get('cuda'),
        device=device,
        wandb_watch_model=config.get('wandb_watch_model'),
        wandb_log_freq=config.get('wandb_log_freq'),
        num_classes=config.get('num_classes', 1000),
    )
    model.to(device)
    
    # Resolve total epoch count (max_duration overrides epochs if present)
    epochs = config.get('max_duration')
    config['epochs'] = epochs  # normalise so downstream code sees an int

    # Build optimizer and LR scheduler
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, epochs)
    if scheduler is not None:
        logger.info(f"LR scheduler: {scheduler.__class__.__name__}")
    
    # Handle checkpoint resumption if needed
    resume_epoch = handle_checkpoint_resumption(
        config, model, optimizer, dataloader_generator, logger, scheduler=scheduler,
    )

    log_model_architecture(model, logger)
    
    # Initialize loss criterion
    criterion_name = config.get('criterion')
    if criterion_name == 'MSELoss':
        criterion = nn.MSELoss()
    elif criterion_name == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Criterion {criterion_name} not supported")
    
    # Choose perturbation strategy
    perturb_strategy = choose_perturbation_strategy(
        config['perturb_type'],
        config['perturb_epoch'],
        config['perturb_length'],
        config['perturb_seed'],
        target_mean=global_target_mean,
        target_std=global_target_std,
    )
    
    # Log training configuration
    logger.info("\nModel Configuration:")
    logger.info("-------------------")
    for key, value in config.items():
        logger.info(f"{key}: {value}")
    
    logger.info("\nTraining layers:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(name)
    
    logger.info(f"\nNumber of trainable parameters: {count_trainable_parameters(model)}\n")
    
    # Run training
    try:
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            inference_loader=inference_loader,
            device=device,
            optimizer=optimizer,
            criterion=criterion,
            epochs=epochs,
            training_results_save_path=training_results_save_path,
            logger=logger,
            early_stopping_patience=config['early_stopping_patience'],
            save_path=save_path,
            random_state_save_path=random_state_save_path,
            dataloader_generator=dataloader_generator,
            vision_layers=config.get('vision_layers'),
            transformer_layers=config.get('transformer_layers'),
            perturb_strategy=perturb_strategy,
            start_epoch=resume_epoch,
            behavioral_rsa=config.get('behavioral_rsa', False),
            rsa_annotations_file=config.get('rsa_annotations_file'),
            model_rdm_distance_metric=config.get('model_rdm_distance_metric', 'pearson'),
            rsa_similarity_metric=config.get('rsa_similarity_metric', 'spearman'),
            debug_logging=config.get('debug_logging', False),
            scheduler=scheduler,
            training_mode=training_mode,
        )
    finally:
        wandb.finish()
        if hasattr(logger, 'restore_streams'):
            logger.restore_streams()