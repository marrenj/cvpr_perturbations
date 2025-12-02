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


def run_inference(config):

    # Set up logger
    os.makedirs(config['inference_save_dir'], exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(config['inference_save_dir'], f'inference_log_{timestamp}.txt')
    logger = setup_logger(log_file)

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
    
    # Set device
    if config['cuda'] == -1:
        device = torch.device("cuda")
    elif config['cuda'] == 0:
        device = torch.device("cuda:0")
    elif config['cuda'] == 1:
        device = torch.device("cuda:1")
    else:
        device = torch.device("cpu")

    
    # Initialize dataset
    if config['dataset'] == 'things':
        dataset = ThingsDataset(img_annotations_file=config['img_annotations_file'], img_dir=config['img_dir'])
    elif config['dataset'] == 'nod':
        dataset = NodDataset(img_annotations_file=config['img_annotations_file'], img_dir=config['img_dir'])
    elif config['dataset'] == 'nights':
        dataset = NightsTripletDataset(nights_dir=config['nights_dir'], split=config['nights_split'])
    else:
        raise ValueError(f"Dataset type {config['dataset_type']} not supported")


    # Adapt inference task
    task = adapt_inference_task(config['dataset'])


    # LOAD MODEL WEIGHTS FROM CHECKPOINT
    weights_path = Path(config['model']['model_weights'])

    dora_files = []

    if weights_path.is_dir():
        dora_files = sorted([f for f in weights_path.iterdir() if f.suffix == '.pth']),
        key=lambda f: int(f.name.split('_')[0].replace('epoch', '')))
        # Filter epochs by min test loss in training results CSV   
        if config.get('training_res_path'):
            res_df = pd.read_csv(config['training_res_path'])
            min_idx = res_df['test_loss'].idxmin()
            min_epoch = int(res_df.loc[min_idx, 'epoch'])
            dora_files = [f for f in dora_files 
                          if int(f.name.split('_')[0].replace('epoch', '')) <= min_epoch]
            print(f"Filtered DoRA files to epoch <= {min_epoch} (min test loss)")

    model.to(device)

    if dora_files:
        for file in dora_files:
            epoch = file.name.split('_')[0].replace('epoch', '')
            print(f"\n=== Processing epoch {epoch} ===")
            load_dora_checkpoint(model, checkpoint_root=weights_path, epoch=epoch, map_location=device)
    else:
        # Single model checkpoint
        epoch = dora_files.name.split('_')[0].replace('epoch', '')
        print(f"Loading model weights from {weights_path}")
        load_dora_checkpoint(model, checkpoint_root=weights_path, epoch=epoch, map_location=device)


    # INFERENCE ON NIGHTS DATASET
    nights_dir = config['nights_dir']
    splits = config.get('split', config.get('splits', 'test'))
    if isinstance(splits, str):
        splits = [splits]
    results = {}
    cached_batches = {}  # to reuse cached data for multiple splits or epochs

    # Cache dataset once if multiple model states or multiple splits
    if weights_path.is_dir() or len(splits) > 1:
        for split in splits:
            cached = task(model, nights_dir, split=split, batch_size=config['batch_size'],
                                 device=device, use_image_features=config.get('use_image_features', False))[1]
            cached_batches[split] = cached

    for file in (dora_files if dora_files else [None]):
        if file:
            epoch = file.name.split('_')[0].replace('epoch','')
            model.load_state_dict(torch.load(file, map_location='cpu'), strict=False)
            model.to(device)
        for split in splits:
            res, cached_out = evaluate_nights(model, nights_dir, split=split, batch_size=config['batch_size'],
                                          device=device, use_image_features=config.get('use_image_features', False),
                                          cached_batches=cached_batches.get(split))
            results[(file.name if file else 'model', split)] = res
            print(f"[NIGHTS] {split} accuracy = {res['accuracy']:.2f}%")
            # Optionally, save detailed predictions if needed
            if config.get('save_predictions'):
                df = pd.DataFrame({
                    'ref_index': res['ground_truths'].index,  # example content
                    'human_choice': res['ground_truths'],
                    'model_choice': res['predictions'],
                    'dist_img0': [d[0] for d in res['distances']],
                    'dist_img1': [d[1] for d in res['distances']]
                })
            df.to_csv(Path(config['inference_save_dir'])/f"nights_{split}_predictions_{epoch or 'final'}.csv", index=False)