import os
import random
from numpy.random import set_state as np_set_state
import torch 
import torch.nn as nn

from datetime import datetime
from torch.utils.data import DataLoader, random_split, Subset
from torch.nn import DataParallel
from torch.optim import AdamW

from src.utils.seed import seed_everything
from src.utils.logging import setup_logger
from src.utils.count_parameters import count_trainable_parameters
from src.training.train_loop import train_model
from src.models.clip_hba.clip_hba_utils import (
    CLIPHBA,
    apply_dora_to_ViT,
    switch_dora_layers,
    load_dora_checkpoint,
)
from src.data.things_dataset import ThingsDataset
from src.data.spose_dimensions import classnames66
from src.perturbations.perturbation_utils import choose_perturbation_strategy
from src.utils.path_setup import setup_paths

def run_training_experiment(config):
    """
    Run behavioral training with the given configuration.
    
    Args:
        config (dict): Configuration dictionary containing training parameters
    """
    seed_everything(config['random_seed'])

    # Set up logger
    os.makedirs(config['checkpoint_path'], exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(config['checkpoint_path'], f'training_log_{timestamp}.txt')
    logger = setup_logger(log_file)
    
    logger.info("="*80)
    logger.info("Starting Training Run")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)

    # Set up paths
    checkpoint_path, training_res_path, random_state_path = setup_paths(config['checkpoint_path'])
    
    # Initialize dataset
    if config['dataset_type'] == 'things':
        dataset = ThingsDataset(img_annotations_file=config['img_annotations_file'], img_dir=config['img_dir'])
    else:
        raise ValueError(f"Dataset type {config['dataset_type']} not supported")
    
    global_target_mean = None
    global_target_std = None

    if config['perturb_type'] == 'random_target':
        target_array = torch.tensor(
            dataset.annotations.iloc[:, 1:].values.astype('float32'),
            dtype=torch.float32,
        )
        flat_targets = target_array.flatten()
        global_target_mean = flat_targets.mean()
        global_target_std = flat_targets.std(unbiased=False)

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
    split_file = os.path.join(random_state_path, 'dataset_split_indices.pth')
    os.makedirs(random_state_path, exist_ok=True)
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
    if config['perturb_epoch'] > 0:
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

        state_payload = torch.load(random_state_file, map_location='cpu')
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
        training_res_path=training_res_path,
        logger=logger,
        early_stopping_patience=config['early_stopping_patience'],
        checkpoint_path=checkpoint_path,
        random_state_path=random_state_path,
        dataloader_generator=dataloader_generator,
        vision_layers=config['vision_layers'],
        transformer_layers=config['transformer_layers'],
        perturb_strategy=perturb_strategy,
        start_epoch=resume_epoch,
    )