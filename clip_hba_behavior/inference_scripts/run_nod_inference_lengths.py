"""
Run NOD inference on perturbation length experiments.

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

from functions.nod_inference_pipeline import run_behavior_inference
from pathlib import Path
from functions.spose_dimensions import *
import torch
import json
import sys

# Get the script's directory for resolving relative paths
SCRIPT_DIR = Path(__file__).parent.resolve()


def load_config(config_path=None):
    """
    Load configuration from JSON file or use defaults.
    
    Args:
        config_path: Path to JSON config file. If None, looks for 'config.json' 
                     in the script directory, or uses defaults.
    
    Returns:
        dict: Configuration dictionary with paths and settings.
    """
    # Default configuration
    default_config = {
        'results_dir': '/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior_loops/perturb_length_experiments_baselineseed1_perturbseed0',
        'img_dir': '/home/wallacelab/teba/multimodal_brain_inspired/NOD/imagenet',
        #'model_path': '/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior/models/cliphba_behavior_20250919_212822.pth',
        'batch_size': 64,
        'cuda': 'cuda:0',
        'load_hba': True,
        'backbone': 'ViT-L/14'
    }
    
    # Determine config file path
    if config_path is None:
        config_path = SCRIPT_DIR / 'config.json'
    else:
        config_path = Path(config_path)
    
    # Load config from file if it exists
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
            # Merge with defaults (file config takes precedence)
            config = {**default_config, **file_config}
            print(f"Loaded configuration from: {config_path}")
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in config file {config_path}: {e}")
            print("Using default configuration.")
            config = default_config
    else:
        print(f"Config file not found at {config_path}. Using default configuration.")
        config = default_config
    
    return config


def validate_paths(config):
    """Validate that required paths exist."""
    results_dir = Path(config['results_dir'])
    img_dir = Path(config['img_dir'])
    #model_path = Path(config['model_path'])
    
    errors = []
    if not results_dir.exists():
        errors.append(f"Results directory not found: {results_dir}")
    if not img_dir.exists():
        errors.append(f"Image directory not found: {img_dir}")
    # if not model_path.exists():
    #     errors.append(f"Model file not found: {model_path}")
    
    if errors:
        raise FileNotFoundError("\n".join(errors))
    
    return results_dir, img_dir


def main():
    # Load configuration from JSON file or use defaults
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    file_config = load_config(config_path)
    
    # Validate paths exist
    results_dir, img_dir = validate_paths(file_config)
    
    # Get category index file (relative to script directory)
    category_index_file = SCRIPT_DIR.parent / 'analysis' / 'sorted_file_categories.csv'
    
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
        'category_index_file': str(category_index_file), 
        'load_hba': file_config['load_hba'],  # False will load the original CLIP-ViT weights
        'backbone': file_config['backbone'],  # CLIP backbone model
        #'model_path': str(model_path),  # path to the final trained model
        'batch_size': file_config['batch_size'],  # batch size
        'cuda': file_config['cuda'],  # 'cuda:0' for GPU 0, 'cuda:1' for GPU 1, '-1' for all GPUs
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
        save_folder = run_dir / 'nod_inference_results'
        #  if 'metrics.csv' exists, use it, otherwise use 'training_res.csv'
        training_res_path = run_dir / 'metrics.csv' if (run_dir / 'metrics.csv').exists() else run_dir / 'training_res.csv'

        print(f"Dora params path: {dora_params_path}")
        print(f"Training results path: {training_res_path}")
        print(f"Save folder: {save_folder}\n")

        # Create config for this specific run
        inference_config['dora_params_path'] = str(dora_params_path)
        inference_config['save_folder'] = str(save_folder)
        inference_config['training_res_path'] = str(training_res_path) if training_res_path.exists() else None

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