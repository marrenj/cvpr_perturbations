from src.models.clip_hba.clip_hba_utils import CLIPHBA, apply_dora_to_ViT, load_dora_checkpoint
from src.data.spose_dimensions import classnames66
import torch
import os
from datetime import datetime
from pathlib import Path

from src.utils.logging import setup_logger

from src.inference.evaluate_nights import evaluate_nights
from src.inference.extract_embeddings import extract_embeddings


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


    ## LOAD MODEL WEIGHTS FROM CHECKPOINT(S)
    weights_path = Path(config['model_weights_path'])

    # Resolve checkpoint root and the set of epoch identifiers to run
    if weights_path.is_file():
        checkpoint_root = (
            weights_path.parent.parent if weights_path.parent.name == "dora_params" else weights_path.parent
        )
        epochs = [weights_path.stem.replace('epoch', '').replace('_dora_params', '')]
    else:
        dora_dir = weights_path if weights_path.name == "dora_params" else weights_path / "dora_params"
        checkpoint_root = dora_dir.parent
        checkpoint_files = sorted(dora_dir.glob("epoch*_dora_params.pth"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No DoRA checkpoints found under {dora_dir}")
        epochs = [
            ckpt.stem.replace('epoch', '').replace('_dora_params', '')
            for ckpt in checkpoint_files
        ]

    cached_batches = None  # allow reuse for nights

    for epoch in epochs:
        logger.info(f"\n=== Processing epoch {epoch} ===")
        load_dora_checkpoint(model, checkpoint_root=checkpoint_root, epoch=epoch, map_location=device)

        # Move model to target device for inference
        model = model.to(device)

        ## DISPATCH TO DATASET-SPECIFIC LOGIC
        results = None

        if config['dataset'] in ('things', 'nod'):
            results = extract_embeddings(
                model=model,
                dataset_name=config['dataset'],
                config=config,
                device=device,
                logger=logger,
            )

        elif config['dataset'] == 'nights':
            results, cached_batches = evaluate_nights(
                model=model,
                nights_dir=config['img_dir'],
                split='test',
                batch_size=config.get('batch_size', 32),
                device=device,
                use_image_features=False,
                cached_batches=cached_batches,
            )
        else:
            raise ValueError(f"Dataset type {config['dataset']} not supported")

        # Persist results for downstream analysis (one file per epoch)
        results_path = Path(config['inference_save_dir']) / f"inference_results_{config['dataset']}_epoch{epoch}.pt"
        torch.save(
            {
                "dataset": config["dataset"],
                "epoch": epoch,
                "results": results,
            },
            results_path,
        )
        logger.info(f"Saved inference results to {results_path}")

    return results, cached_batches