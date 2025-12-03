from src.models.clip_hba.clip_hba_utils import CLIPHBA, apply_dora_to_ViT, load_dora_checkpoint
from src.data.spose_dimensions import classnames66
import torch
import os
import pandas as pd
from datetime import datetime
from pathlib import Path

from src.utils.logger import setup_logger

from src.inference.evaluate_nights import evaluate_nights
from src.data.things_dataset import ThingsDataset
from src.data.nod_dataset import NODDataset
from src.inference.inference_task_adapter import adapt_inference_task


## INITIALIZE LOGGER

## INITIALIZE CLIPHBA MODEL 

## LOAD MODEL WEIGHTS FROM A CHECKPOINT

## DISPATCH TO DATASET-SPECIFIC LOGIC


def run_inference(config):

    ## INITIALIZE LOGGER
    os.makedirs(config['inference_save_dir'], exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(config['inference_save_dir'], f'inference_log_{timestamp}.txt')
    logger = setup_logger(log_file)
    
    ## INITIALIZE CLIPHBA MODEL
    # Determine pos_embedding based on backbone
    pos_embedding = False if config['backbone'] == 'RN50' else True
    logger.info(f"pos_embedding is {pos_embedding}")

    # Initialize model
    model = CLIPHBA(classnames=classnames66, backbone_name=config['backbone'], 
                pos_embedding=pos_embedding)
    model.eval() # inference mode

    # Apply DoRA
    apply_dora_to_ViT(model, 
                      n_vision_layers=config['vision_layers'],
                      n_transformer_layers=config['transformer_layers'],
                      r=config['rank'],
                      dora_dropout=0.1)
    
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
    epoch = weights_path.name.split('_')[0].replace('epoch', '')
    logger.info(f"\n=== Processing epoch {epoch} ===")
    load_dora_checkpoint(model, checkpoint_root=weights_path, epoch=epoch, map_location=device)


    ## DISPATCH TO DATASET-SPECIFIC LOGIC
    if config['dataset'] == 'things':
        evaluate_things
    elif config['dataset'] == 'nod':
        evaluate_nod
    elif config['dataset'] == 'nights':
        results, cached_batches = evaluate_nights(model=model, nights_dir=config['img_dir'], split='test', batch_size=32, 
                    device=device, use_image_features=False, cached_batches=None)
    else:
        raise ValueError(f"Dataset type {config['dataset']} not supported")

    return results, cached_batches