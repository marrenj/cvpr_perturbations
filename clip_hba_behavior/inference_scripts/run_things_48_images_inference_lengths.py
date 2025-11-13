"""
Run 48 Things images inference on perturbation length experiments.

This script:
1. Finds all training runs in the specified directory
2. Groups runs by starting epoch (e.g., all e2_* experiments)
3. For each group, processes runs in increasing perturbation length order
4. When processing a run, if other runs with the same starting epoch have been processed,
   it copies their existing embeddings to avoid redundant computation
5. Only processes epochs that haven't been computed yet

This optimization avoids recomputing identical embeddings across experiments with the same
starting epoch but different perturbation lengths.
"""

from functions.things_48_images_inference_pipeline import run_behavior_inference
from pathlib import Path
from functions.spose_dimensions import *
import torch


def validate_paths(config):
    """Validate that required paths exist."""
    results_dir = Path(config['results_dir'])
    img_dir = Path(config['img_dir'])
    inference_csv_file = Path(config['inference_csv_file'])
    rdm48_triplet_dir = Path(config['RDM48_triplet_dir'])
    
    errors = []
    if not results_dir.exists():
        errors.append(f"Results directory not found: {results_dir}")
    if not img_dir.exists():
        errors.append(f"Image directory not found: {img_dir}")
    if not inference_csv_file.exists():
        errors.append(f"Inference CSV file not found: {inference_csv_file}")
    if not rdm48_triplet_dir.exists():
        errors.append(f"RDM48 triplet file not found: {rdm48_triplet_dir}")
    
    if errors:
        raise FileNotFoundError("\n".join(errors))
    
    return results_dir, img_dir


def main():
    # Define configuration directly
    config = {
        'results_dir': '/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior_loops/perturb_length_experiments_baselineseed1_perturbseed0',
        'img_dir': '../Data/Things1854',
        'inference_csv_file': '../Data/spose_embedding66d_rescaled_48val_reordered.csv',
        'RDM48_triplet_dir': '../Data/RDM48_triplet.mat',
        'batch_size': 48,
        'cuda': 'cuda:0',
        'load_hba': True,
        'backbone': 'ViT-L/14'
    }
    
    # Validate paths exist
    results_dir, img_dir = validate_paths(config)
    
    # List all training run directories in the results_dir
    run_dirs = sorted([d for d in results_dir.iterdir() 
                      if d.is_dir()
                      and d.name.startswith('random_target')])
    
    # # Uncomment to filter runs that end with _l50
    # run_dirs = sorted([d for d in results_dir.iterdir() 
    #                   if d.is_dir()
    #                   and d.name.startswith('random_target')
    #                   and d.name.endswith('_l50')])
    
    print(f"Found {len(run_dirs)} training run directories:")
    for run_dir in run_dirs:
        # the epoch number is the second to last part of the directory name
        epoch_part = run_dir.name.split('_')[-2]
        epoch_number = int(epoch_part.lstrip('e'))
        # the perturbation length is the last part of the directory name
        length_part = run_dir.name.split('_')[-1]  
        perturbation_length = int(length_part.lstrip('l'))
        print(f"  - {run_dir.name} (start epoch: {epoch_number}, length: {perturbation_length})")
    
    # Sort the run_dirs by the epoch number, then by the perturbation length
    run_dirs = sorted(run_dirs, key=lambda x: (int(x.name.split('_')[-2].lstrip('e')), int(x.name.split('_')[-1].lstrip('l'))))
    
    # Prepare inference config dictionary
    inference_config = {
        'img_dir': str(img_dir),  # input images directory
        'inference_csv_file': str(config['inference_csv_file']),
        'RDM48_triplet_dir': str(config['RDM48_triplet_dir']),
        'load_hba': config['load_hba'],  # False will load the original CLIP-ViT weights
        'backbone': config['backbone'],  # CLIP backbone model
        'batch_size': config['batch_size'],  # batch size
        'cuda': config['cuda'],  # 'cuda:0' for GPU 0, 'cuda:1' for GPU 1, '-1' for all GPUs
    }

    # Loop through the run_dirs and run inference
    for run_dir in run_dirs:
        # Extract epoch number and perturbation length from directory name
        epoch_part = run_dir.name.split('_')[-2]
        epoch_number = int(epoch_part.lstrip('e'))
        length_part = run_dir.name.split('_')[-1]  
        perturbation_length = int(length_part.lstrip('l'))  
        
        print(f"\n{'='*80}")
        print(f"Processing: {run_dir.name}")
        print(f"Starting Epoch: {epoch_number}")
        print(f"Perturbation length: {perturbation_length}")
        print(f"{'='*80}\n")

        # Construct paths specific to this training run
        dora_params_path = run_dir / f'dora_params_{epoch_number}'
        save_folder = run_dir / 'things_48_inference_results'
        #  if 'metrics.csv' exists, use it, otherwise use 'training_res.csv'
        # Find any file in run_dir that starts with "metrics" or "training_res"
        metric_files = [f for f in run_dir.iterdir() if f.is_file() and (f.name.startswith('metrics') or f.name.startswith('training_res'))]
        # Use the first one found, or None if not found
        training_res_path = metric_files[0] if metric_files else None

        print(f"Dora params path: {dora_params_path}")
        print(f"Training results path: {training_res_path}")
        print(f"Save folder: {save_folder}\n")

        # Create config for this specific run
        inference_config['dora_params_path'] = str(dora_params_path)
        inference_config['save_folder'] = str(save_folder)
        inference_config['training_res_path'] = str(training_res_path) if training_res_path and training_res_path.exists() else None

        # Run inference with configuration
        try:
            run_behavior_inference(inference_config)
            print(f"✓ Completed inference for {run_dir.name}\n")
        except Exception as e:
            print(f"✗ Error during inference for {run_dir.name}: {e}\n")
            continue

        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()

