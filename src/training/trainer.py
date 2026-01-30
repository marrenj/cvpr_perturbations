"""
CLIP-HBA Training Script with DoRA Adaptation

Trains CLIP-based HBA models using DoRA (Weight-Decomposed
Low-Rank Adaptation) on behavioral similarity data. Supports perturbation strategies
and checkpoint resumption for reproducible experiments.
"""

import os
import random
import csv
from tqdm import tqdm
from numpy.random import set_state as np_set_state
import torch 
import torch.nn as nn
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, random_split, Subset
from torch.nn import DataParallel
from torch.optim import AdamW

from src.training.test_loop import evaluate_model
from src.utils.save_random_states import save_random_states
from src.models.clip_hba.clip_hba_utils import save_dora_parameters

from src.utils.seed import seed_everything
from src.utils.logging import setup_logger
from src.utils.count_parameters import count_trainable_parameters
from src.models.clip_hba.clip_hba_utils import (
    CLIPHBA,
    apply_dora_to_ViT,
    switch_dora_layers,
    load_dora_checkpoint,
)
from src.data.things_dataset import ThingsBehavioralDataset
from src.data.spose_dimensions import classnames66
from src.perturbations.perturbation_utils import choose_perturbation_strategy


# =============================================================================
# Path Setup
# =============================================================================

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
                return config['save_path'], training_results_save_path, random_state_save_path, checkpoints_save_path


# =============================================================================
# Dataset Setup
# =============================================================================

def setup_dataset(config, logger):
    """
    Initialize and split dataset according to configuration.
    
    Args:
        config: Configuration dictionary containing dataset parameters
        logger: Logger instance for tracking dataset operations
        
    Returns:
        tuple: (train_dataset, test_dataset, split_info, global_target_stats)
    """
    # Initialize dataset
    if config['dataset_type'] == 'things':
        dataset = ThingsBehavioralDataset(
            img_annotations_file=config['img_annotations_file'],
            img_dir=config['img_dir'],
        )
    else:
        raise ValueError(f"Dataset type {config['dataset_type']} not supported")
    
    # Compute global dataset statistics if needed for random target perturbations
    global_target_mean = None
    global_target_std = None
    
    if config['perturb_type'] == 'random_target':
        embeddings = dataset.annotations.iloc[:, 1:].values.astype('float32')
        global_target_mean = torch.tensor(np.mean(embeddings), dtype=torch.float32)
        global_target_std = torch.tensor(np.std(embeddings), dtype=torch.float32)
    
    # Determine if we should reuse an existing split
    baseline_checkpoint_path = config.get('baseline_checkpoint_path')
    split_indices_path = None
    if baseline_checkpoint_path:
        split_candidate = os.path.join(baseline_checkpoint_path, 'random_states', 'dataset_split_indices.pth')
        if os.path.isfile(split_candidate):
            split_indices_path = split_candidate
    
    # Load or create dataset split
    if split_indices_path:
        if not os.path.isfile(split_indices_path):
            raise FileNotFoundError(f"dataset_split_indices_path not found: {split_indices_path}")
        split_info = torch.load(split_indices_path)
        train_dataset = Subset(dataset, split_info['train_indices'])
        test_dataset = Subset(dataset, split_info['test_indices'])
        logger.info(f"Loaded dataset split indices from {split_indices_path}")
    else:
        train_size = int(config['train_portion'] * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        split_info = {
            'train_indices': list(train_dataset.indices),
            'test_indices': list(test_dataset.indices),
            'random_seed': config['random_seed'],
            'train_portion': config['train_portion']
        }
    
    return train_dataset, test_dataset, split_info, (global_target_mean, global_target_std)


def create_dataloaders(train_dataset, test_dataset, config):
    """
    Create DataLoader instances for training and testing.
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test/validation dataset
        config: Configuration dictionary with batch_size and random_seed
        
    Returns:
        tuple: (train_loader, test_loader, dataloader_generator)
    """
    # Create generator for reproducible DataLoader shuffling
    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(config['random_seed'])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        generator=dataloader_generator
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    return train_loader, test_loader, dataloader_generator


# =============================================================================
# Model Setup
# =============================================================================

def setup_model(config, device):
    """
    Initialize model and move to device.
    
    Args:
        config: Configuration dictionary containing model parameters
        device: Target device for model
        
    Returns:
        torch.nn.Module: Configured model ready for training
    """
    # Determine positional embedding based on backbone
    pos_embedding = False if config['backbone'] == 'RN50' else True
    print(f"pos_embedding is {pos_embedding}")
    
    # Initialize base model
    model = CLIPHBA(
        classnames=classnames66, 
        backbone_name=config['backbone'], 
        pos_embedding=pos_embedding
    )
    
    # Apply DoRA adapters
    apply_dora_to_ViT(
        model, 
        n_vision_layers=config['vision_layers'],
        n_transformer_layers=config['transformer_layers'],
        r=config['rank'],
        dora_dropout=0.1
    )
    switch_dora_layers(model, freeze_all=True, dora_state=True)
    
    # Use DataParallel if using all GPUs
    if config['cuda'] == -1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)
    
    model.to(device)
    
    return model


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


# =============================================================================
# Checkpoint & State Management
# =============================================================================


def load_checkpoint_and_states(model, optimizer, dataloader_generator, checkpoint_path, 
                               random_state_path, logger):
    """
    Load model checkpoint and restore all random states for reproducibility.
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to restore state
        dataloader_generator: Generator for DataLoader shuffling
        checkpoint_path: Path to checkpoint directory
        random_state_path: Path to random state file
        logger: Logger instance
        
    Returns:
        int: Next epoch to resume from
    """
    # Load DoRA weights
    loaded_dora_path = load_dora_checkpoint(
        model,
        checkpoint_root=checkpoint_path,
        epoch=random_state_path.split('epoch')[-1].split('_')[0],
        strict=False,
    )
    logger.info(f"Loaded DoRA parameters from: {loaded_dora_path}")
    
    # Load and restore random states
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
        logger.info(f"Restored optimizer and RNG states from {random_state_path}")
    
    return int(random_state_path.split('epoch')[-1].split('_')[0]) + 1


def handle_checkpoint_resumption(config, model, optimizer, dataloader_generator, logger):
    """
    Handle checkpoint loading for either explicit resumption or baseline loading.
    
    Args:
        config: Configuration dictionary
        model: Model instance
        optimizer: Optimizer instance
        dataloader_generator: DataLoader generator
        logger: Logger instance
        
    Returns:
        int: Epoch to start/resume from
    """
    resume_epoch = 0
    resume_checkpoint_path = config.get("resume_checkpoint_path")
    resume_from_epoch = config.get("resume_from_epoch")
    
    # Case 1: Explicit resumption from a specific checkpoint
    if resume_checkpoint_path and resume_from_epoch is not None:
        random_state_file = os.path.join(
            resume_checkpoint_path,
            'random_states',
            f'epoch{resume_from_epoch}_random_states.pth'
        )
        resume_epoch = load_checkpoint_and_states(
            model, optimizer, dataloader_generator,
            resume_checkpoint_path, random_state_file, logger
        )
    
    # Case 2: Load baseline checkpoint before applying perturbation
    elif config['perturb_epoch'] > 0:
        baseline_epoch = config['perturb_epoch'] - 1
        baseline_checkpoint_path = config.get('baseline_checkpoint_path')
        
        if not baseline_checkpoint_path:
            raise ValueError("baseline_checkpoint_path must be provided when perturb_epoch > 0")
        
        random_state_file = os.path.join(
            baseline_checkpoint_path,
            'random_states',
            f'epoch{baseline_epoch}_random_states.pth'
        )
        resume_epoch = load_checkpoint_and_states(
            model, optimizer, dataloader_generator,
            baseline_checkpoint_path, random_state_file, logger
        )
        logger.info(f"Loaded baseline checkpoint from epoch {baseline_epoch}")
    
    return resume_epoch


# =============================================================================
# Training Loop
# =============================================================================


def train_one_epoch(model, train_loader, device, optimizer, criterion, 
                   perturb_strategy, epoch_idx, logger):
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
        
    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    # Log if perturbation is active this epoch
    if perturb_strategy.is_active_epoch(epoch_idx):
        logger.info(f"Applying {perturb_strategy.__class__.__name__} perturbation during epoch {epoch_idx}")
    
    progress_bar = tqdm(
        enumerate(train_loader), 
        total=len(train_loader), 
        desc=f"Epoch {epoch_idx}"
    )
    
    for batch_idx, (_, images, targets) in progress_bar:
        # Move data to device
        images = images.to(device)
        targets = targets.to(device)
        
        # Apply perturbation if active
        images, targets = perturb_strategy.apply_to_batch(
            images, targets, device, epoch_idx, batch_idx
        )
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item() * images.size(0)
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_train_loss = total_loss / len(train_loader.dataset)
    return avg_train_loss


def save_epoch_results(epoch, train_loss, test_loss, training_results_save_path):
    """
    Save training metrics to CSV file.
    
    Args:
        epoch: Current epoch number
        train_loss: Training loss
        test_loss: Validation loss
        training_results_save_path: Path to CSV file
    """
    data_row = [epoch, train_loss, test_loss]
    
    with open(training_results_save_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data_row)


def train_model(
    model,
    train_loader,
    test_loader,
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
):
    """
    Main training loop with logging, checkpointing, and early stopping.

    Args:
        model: Model to train
        train_loader: Training data loader
        test_loader: Validation data loader
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
    """
    best_test_loss = float('inf')
    epochs_no_improve = 0
    
    # Use logger if provided, otherwise use print
    log = logger.info if logger else print
    
    # Initial evaluation
    log("*********************************")
    log("Evaluating initial model")
    best_test_loss = evaluate_model(model, test_loader, device, criterion)
    log(f"Initial Validation Loss: {best_test_loss:.4f}")
    log("*********************************\n")
    
    # Create directories for outputs
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.dirname(training_results_save_path), exist_ok=True)
    
    # Initialize CSV file (only if starting from scratch)
    if not (start_epoch > 0 and os.path.exists(training_results_save_path)):
        with open(training_results_save_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['epoch', 'train_loss', 'test_loss'])
    # Main training loop
    for epoch in range(start_epoch, epochs):
        # Train one epoch
        avg_train_loss = train_one_epoch(
            model, train_loader, device, optimizer, criterion,
            perturb_strategy, epoch, logger
        )
        
        # Evaluate on validation set
        avg_test_loss = evaluate_model(model, test_loader, device, criterion)
        
        # Log results
        print(f"Epoch {epoch}: Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_test_loss:.4f}")
        log(f"Epoch {epoch}: Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_test_loss:.4f}")
        
        if perturb_strategy.is_active_epoch(epoch):
            logger.info(f"*** Perturbation '{perturb_strategy.__class__.__name__}' was applied during epoch {epoch} ***")
        
        # Save metrics to CSV
        save_epoch_results(epoch, avg_train_loss, avg_test_loss, training_results_save_path)
        
        # Save random states for reproducibility
        save_random_states(
            optimizer, epoch, random_state_save_path, 
            dataloader_generator, logger=logger
        )
        
        # Save model checkpoint
        dora_params_dir = os.path.join(save_path, "dora_params")
        save_dora_parameters(
            model, dora_params_dir, epoch,
            vision_layers, transformer_layers,
            log_fn=log,
        )
        log(f"Checkpoint saved for epoch {epoch}")
        
        # Early stopping check
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve == early_stopping_patience:
            log("\n\n*********************************")
            log(f"Early stopping triggered at epoch {epoch}")
            log("*********************************\n\n")
            break


# =============================================================================
# Main Training Orchestration
# =============================================================================


def run_training_experiment(config):
    """
    Main entry point for running a training experiment.
    
    Orchestrates the entire training pipeline including:
    - Environment setup and seeding
    - Dataset initialization and splitting
    - Model creation and configuration
    - Training loop execution
    - Checkpoint and state management
    
    Args:
        config: Configuration dictionary containing all training parameters
    """
    # Set random seed for reproducibility
    seed_everything(config['random_seed'])
    
    # Setup logging
    os.makedirs(config['save_path'], exist_ok=True)
    log_file = os.path.join(config['save_path'], f'training_log_{config["perturb_type"]}.txt')
    logger = setup_logger(log_file)
    
    logger.info("="*80)
    logger.info("Starting Training Run")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)
    
    # Setup paths
    save_path, training_results_save_path, random_state_save_path = setup_paths(config['save_path'])
    
    # Setup dataset
    train_dataset, test_dataset, split_info, target_stats = setup_dataset(config, logger)
    global_target_mean, global_target_std = target_stats
    
    # Save dataset split information
    split_file = os.path.join(random_state_save_path, 'dataset_split_indices.pth')
    os.makedirs(random_state_save_path, exist_ok=True)
    torch.save(split_info, split_file)
    logger.info(f"Dataset split indices saved: {split_file}")
    
    # Create data loaders
    train_loader, test_loader, dataloader_generator = create_dataloaders(
        train_dataset, test_dataset, config
    )
    
    # Setup device
    device = get_device(config['cuda'])
    
    # Setup model
    model = setup_model(config, device)
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=config['lr'])
    
    # Handle checkpoint resumption if needed
    resume_epoch = handle_checkpoint_resumption(
        config, model, optimizer, dataloader_generator, logger
    )
    
    # Initialize loss criterion
    if config['criterion'] == 'MSELoss':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Criterion {config['criterion']} not supported")
    
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
    
    logger.info("\nUpdating layers:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(name)
    
    logger.info(f"\nNumber of trainable parameters: {count_trainable_parameters(model)}\n")
    
    # Run training
    train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        optimizer=optimizer,
        criterion=criterion,
        epochs=config['epochs'],
        training_results_save_path=training_results_save_path,
        logger=logger,
        early_stopping_patience=config['early_stopping_patience'],
        save_path=save_path,
        random_state_save_path=random_state_save_path,
        dataloader_generator=dataloader_generator,
        vision_layers=config['vision_layers'],
        transformer_layers=config['transformer_layers'],
        perturb_strategy=perturb_strategy,
        start_epoch=resume_epoch,
    )