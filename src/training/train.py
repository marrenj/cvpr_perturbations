import os
import torch 
import torch.nn as nn

from datetime import datetime
from torch.utils.data import DataLoader, random_split
from torch.nn import DataParallel
from torch.optim import AdamW

from src.utils.seed import seed_everything
from src.utils.logging import setup_logger
from src.utils.count_parameters import count_trainable_parameters
from src.training.train_loop import train_model
from src.models.clip_hba.clip_hba_utils import CLIPHBA, apply_dora_to_ViT, switch_dora_layers
from src.data.things_dataset import ThingsDataset
from src.data.spose_dimensions import classnames66


def run_training(config):
    """
    Run behavioral training with the given configuration.
    
    Args:
        config (dict): Configuration dictionary containing training parameters
    """
    seed_everything(config['random_seed'])

    # Set up logger
    log_dir = os.path.dirname(config['checkpoint_path'])
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_log_{timestamp}.txt')
    logger = setup_logger(log_file)
    
    logger.info("="*80)
    logger.info("Starting Training Run")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)
    
    # Initialize dataset
    if config['dataset_type'] == 'things':
        dataset = ThingsDataset(img_annotations_file=config['img_annotations_file'], img_dir=config['img_dir'])
    else:
        raise ValueError(f"Dataset type {config['dataset_type']} not supported")
    
    # Split dataset
    train_size = int(config['train_portion'] * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    split_info = {
        'train_indices': train_dataset.indices.copy() if hasattr(train_dataset, 'indices') else list(train_dataset.indices),
        'test_indices': test_dataset.indices.copy() if hasattr(test_dataset, 'indices') else list(test_dataset.indices),
        'random_seed': config['random_seed'],
        'train_portion': config['train_portion']
    }
    split_file = os.path.join(config['random_state_path'], 'dataset_split_indices.pth')
    os.makedirs(config['random_state_path'], exist_ok=True)
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

    # Initialize criterion
    if config['criterion'] == 'MSELoss':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Criterion {config['criterion']} not supported")
    
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
        # inference_loader,
        device,
        optimizer,
        criterion,
        config['epochs'],
        config['training_res_path'],
        logger=logger,
        early_stopping_patience=config['early_stopping_patience'],
        checkpoint_path=config['checkpoint_path'],
        random_state_path=config['random_state_path'],
        dataloader_generator=dataloader_generator,
        vision_layers=config['vision_layers'],
        transformer_layers=config['transformer_layers'],
    )