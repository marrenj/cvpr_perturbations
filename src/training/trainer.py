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
    End-to-end training loop that logs metrics, checkpoints DoRA weights, and
    optionally performs early stopping.

    Parameters
    ----------
    model : torch.nn.Module | torch.nn.DataParallel
        Model to optimize; assumed compatible with ``train_loader`` batches.
    train_loader : DataLoader
        Iterable producing (metadata, images, targets) for training.
    test_loader : DataLoader
        Iterable used for computing validation loss each epoch.
    device : torch.device
        Device to which tensors are moved before forward/backward passes.
    optimizer : torch.optim.Optimizer
        Optimizer used to update model parameters.
    criterion : Callable
        Loss function applied to ``(predictions, targets)``.
    epochs : int
        Maximum number of training epochs.
    training_results_save_path : str | Path
        CSV filepath where per-epoch losses are recorded.
    save_path : str | Path
        Directory where DoRA checkpoints are written.
    random_state_save_path : str | Path
        Directory that stores RNG and optimizer state snapshots.
    logger : logging.Logger, optional
        Logger for structured output; defaults to ``print`` when omitted.
    early_stopping_patience : int, default 5
        Number of consecutive epochs without validation improvement before
        stopping early.
    dataloader_generator : torch.Generator, optional
        Generator whose state is persisted for deterministic shuffling.
    vision_layers : int, default 1
        Number of final visual transformer blocks with DoRA adapters to save.
    transformer_layers : int, default 1
        Number of final text transformer blocks with DoRA adapters to save.
    start_epoch : int, default 0
        Epoch index to begin (useful when resuming from checkpoints).
    """
    best_test_loss = float('inf')
    epochs_no_improve = 0

    # Use logger if provided, otherwise use print
    log = logger.info if logger else print

    # initial evaluation
    log("*********************************")
    log("Evaluating initial model")
    best_test_loss = evaluate_model(model, test_loader, device, criterion)
    log(f"Initial Validation Loss: {best_test_loss:.4f}")
    log("*********************************\n")

    # Create folder to store checkpoints
    os.makedirs(save_path, exist_ok=True)

    # Create directory for training results CSV if it doesn't exist
    os.makedirs(os.path.dirname(training_results_save_path), exist_ok=True)

    headers = ['epoch', 'train_loss', 'test_loss']

    # When resuming and the CSV already exists, keep existing rows; otherwise initialize.
    if not (start_epoch > 0 and os.path.exists(training_results_save_path)):
        with open(training_results_save_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        epoch_idx = epoch

        if perturb_strategy.is_active_epoch(epoch_idx):
            logger.info(f"Applying {perturb_strategy.__class__.__name__} perturbation during epoch {epoch}")

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{epochs}")
        for batch_idx, (_, images, targets) in progress_bar:

            images = images.to(device)
            targets = targets.to(device)

            images, targets = perturb_strategy.apply_to_batch(images, targets, device, epoch_idx, batch_idx)

            optimizer.zero_grad()
            predictions = model(images)
            
            loss = criterion(predictions, targets)
            progress_bar.set_postfix({'loss': loss.item()})
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
        avg_train_loss = total_loss / len(train_loader.dataset)

        # Evaluate after every epoch
        avg_test_loss = evaluate_model(model, test_loader, device, criterion)
        print(f"Epoch {epoch}: Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_test_loss:.4f}")
        log(f"Epoch {epoch}: Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_test_loss:.4f}")

        if perturb_strategy.is_active_epoch(epoch_idx):
            logger.info(f"*** Perturbation '{perturb_strategy.__class__.__name__}' was applied during epoch {epoch} ***")

        # Prepare the data row with the epoch number and loss values
        data_row = [epoch, avg_train_loss, avg_test_loss]

        # Append the data row to the CSV file
        with open(training_results_save_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data_row)

        # Save random states and optimizer after every epoch for full reproducibility
        save_random_states(optimizer, epoch, random_state_save_path, dataloader_generator, logger=logger)

        # Save the DoRA parameters (i.e., the checkpoint weights)
        dora_params_dir = os.path.join(save_path, "dora_params")
        save_dora_parameters(
            model,
            dora_params_dir,
            epoch,
            vision_layers,
            transformer_layers,
            log_fn=log,
        )
        log(f"Checkpoint saved for epoch {epoch}")

        # Check for early stopping and saving checkpoint
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


def setup_paths(save_path):
    """
    Set up paths for the training run.
    """
    os.makedirs(save_path, exist_ok=True)
    training_results_save_path = os.path.join(save_path, 'training_res.csv')
    random_state_save_path = os.path.join(save_path, 'random_states')
    return save_path, training_results_save_path, random_state_save_path


def run_training_experiment(config):
    """
    Run behavioral training with the given configuration.
    
    Args:
        config (dict): Configuration dictionary containing training parameters
    """
    seed_everything(config['random_seed'])

    # Set up logger
    os.makedirs(config['save_path'], exist_ok=True)
    log_file = os.path.join(config['save_path'], f'training_log_{config["perturb_type"]}.txt')
    logger = setup_logger(log_file)
    
    logger.info("="*80)
    logger.info("Starting Training Run")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)

    # Set up paths
    save_path, training_results_save_path, random_state_save_path = setup_paths(config['save_path'])
    
    # Initialize dataset
    if config['dataset_type'] == 'things':
        dataset = ThingsBehavioralDataset(
            img_annotations_file=config['img_annotations_file'],
            img_dir=config['img_dir'],
        )
    else:
        raise ValueError(f"Dataset type {config['dataset_type']} not supported")
    
    global_target_mean = None
    global_target_std = None

    if config['perturb_type'] == 'random_target':
        target_array = torch.tensor(
            dataset.annotations.iloc[:, 1:].values.astype('float32'),
            dtype=torch.float32,
        )

        embeddings = dataset.annotations.iloc[:, 1:].values.astype('float32')
        
        global_target_mean = torch.tensor(np.mean(embeddings), dtype=torch.float32)
        global_target_std = torch.tensor(np.std(embeddings), dtype=torch.float32)

    baseline_checkpoint_path = config.get('baseline_checkpoint_path')
    split_indices_path = None
    if baseline_checkpoint_path:
        split_candidate = os.path.join(baseline_checkpoint_path, 'random_states', 'dataset_split_indices.pth')
        if os.path.isfile(split_candidate):
            split_indices_path = split_candidate

    # Split dataset (optionally reuse a precomputed split)
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
    split_file = os.path.join(random_state_save_path, 'dataset_split_indices.pth')
    os.makedirs(random_state_save_path, exist_ok=True)
    torch.save(split_info, split_file)
    logger.info(f"Dataset split indices saved: {split_file}")

    # Initialize inference dataset
    # inference_dataset = ThingsInferenceDataset(inference_csv_file=config['inference_csv_file'], img_dir=config['img_dir'], RDM48_triplet_dir=config['RDM48_triplet_dir'])

    # Create a generator for reproducible DataLoader shuffling
    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(config['random_seed'])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, generator=dataloader_generator)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    # inference_loader = DataLoader(inference_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Determine pos_embedding based on backbone
    pos_embedding = False if config['backbone'] == 'RN50' else True
    print(f"pos_embedding is {pos_embedding}")
    
    # Initialize model
    model = CLIPHBA(classnames=classnames66, backbone_name=config['backbone'], 
                    pos_embedding=pos_embedding)

    # Set device
    if config['cuda'] == -1:
        device = torch.device("cuda")
    elif config['cuda'] == 0:
        device = torch.device("cuda:0")
    elif config['cuda'] == 1:
        device = torch.device("cuda:1")
    else:
        device = torch.device("cpu")
    
    # Apply DoRA
    apply_dora_to_ViT(model, 
                      n_vision_layers=config['vision_layers'],
                      n_transformer_layers=config['transformer_layers'],
                      r=config['rank'],
                      dora_dropout=0.1)
    switch_dora_layers(model, freeze_all=True, dora_state=True)
    
    # Use DataParallel if using all GPUs
    if config['cuda'] == -1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)
    
    model.to(device)
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=config['lr'])

    # Optionally resume from baseline state right before perturbation
    resume_epoch = 0
    resume_checkpoint_path = config.get("resume_checkpoint_path")
    resume_from_epoch = config.get("resume_from_epoch")

    if resume_checkpoint_path and resume_from_epoch is not None:
        loaded_dora_path = load_dora_checkpoint(
            model,
            checkpoint_root=resume_checkpoint_path,
            epoch=resume_from_epoch,
            strict=False,
        )
        logger.info(f"Loaded DoRA parameters from resume checkpoint: {loaded_dora_path}")

        random_state_file = os.path.join(
            resume_checkpoint_path,
            'random_states',
            f'epoch{resume_from_epoch}_random_states.pth'
        )
        if not os.path.isfile(random_state_file):
            raise FileNotFoundError(f"Missing resume random state file: {random_state_file}")

        state_payload = torch.load(random_state_file, map_location='cpu', weights_only=False)
        torch.set_rng_state(state_payload['torch_rng_state'])
        np_set_state(state_payload['numpy_rng_state'])
        random.setstate(state_payload['python_rng_state'])

        if torch.cuda.is_available():
            if 'cuda_rng_state' in state_payload:
                torch.cuda.set_rng_state(state_payload['cuda_rng_state'])
            if 'cuda_rng_state_all' in state_payload:
                torch.cuda.set_rng_state_all(state_payload['cuda_rng_state_all'])

        if dataloader_generator is not None and 'dataloader_generator_state' in state_payload:
            dataloader_generator.set_state(state_payload['dataloader_generator_state'])

        if 'optimizer_state_dict' in state_payload:
            optimizer.load_state_dict(state_payload['optimizer_state_dict'])
            logger.info(f"Restored optimizer and RNG states from {random_state_file}")

        resume_epoch = int(resume_from_epoch) + 1

    elif config['perturb_epoch'] > 0:
        baseline_epoch = config['perturb_epoch'] - 1
        if not baseline_checkpoint_path:
            raise ValueError("baseline_checkpoint_path must be provided when perturb_epoch > 0")

        loaded_dora_path = load_dora_checkpoint(
            model,
            checkpoint_root=baseline_checkpoint_path,
            epoch=baseline_epoch,
            strict=False,
        )

        logger.info(f"Loaded baseline DoRA parameters from {loaded_dora_path}")

        random_state_file = os.path.join(
            baseline_checkpoint_path,
            'random_states',
            f'epoch{baseline_epoch}_random_states.pth'
        )
        if not os.path.isfile(random_state_file):
            raise FileNotFoundError(f"Missing baseline random state file: {random_state_file}")

        state_payload = torch.load(random_state_file, map_location='cpu', weights_only=False)
        torch.set_rng_state(state_payload['torch_rng_state'])
        np_set_state(state_payload['numpy_rng_state'])
        random.setstate(state_payload['python_rng_state'])

        if torch.cuda.is_available():
            if 'cuda_rng_state' in state_payload:
                torch.cuda.set_rng_state(state_payload['cuda_rng_state'])
            if 'cuda_rng_state_all' in state_payload:
                torch.cuda.set_rng_state_all(state_payload['cuda_rng_state_all'])

        if dataloader_generator is not None and 'dataloader_generator_state' in state_payload:
            dataloader_generator.set_state(state_payload['dataloader_generator_state'])

        optimizer.load_state_dict(state_payload['optimizer_state_dict'])
        logger.info(f"Restored baseline optimizer and RNG states from {random_state_file}")
        resume_epoch = baseline_epoch + 1

    # Initialize criterion
    if config['criterion'] == 'MSELoss':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Criterion {config['criterion']} not supported")

    # Choose the appropriate perturbation strategy
    perturb_strategy = choose_perturbation_strategy(
        config['perturb_type'],
        config['perturb_epoch'],
        config['perturb_length'],
        config['perturb_seed'],
        target_mean=global_target_mean,
        target_std=global_target_std,
    )

    # Print training information
    logger.info("\nModel Configuration:")
    logger.info("-------------------")
    for key, value in config.items():
        logger.info(f"{key}: {value}")
    logger.info("\nUpdating layers:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(name)
    logger.info(f"\nNumber of trainable parameters: {count_trainable_parameters(model)}\n")

    # Train model
    train_model(
        model,
        train_loader,
        test_loader,
        device,
        optimizer,
        criterion,
        config['epochs'],
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
