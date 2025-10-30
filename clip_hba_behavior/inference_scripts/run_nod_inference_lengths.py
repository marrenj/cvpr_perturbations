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
import shutil

results_dir = Path('/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior_loops/perturb_length_experiments')

# list all training run directories in the results_dir
run_dirs = sorted([d for d in results_dir.iterdir() 
                  if d.is_dir()
                  and d.name.startswith('random_target')])

print(f"Found {len(run_dirs)} training run directories:")
for run_dir in run_dirs:
    # the epoch number is the second to last part of the directory name
    epoch_part = run_dir.name.split('_')[-2]
    epoch_number = int(epoch_part.lstrip('e'))
    # the perturbation length is the last part of the directory name
    length_part = run_dir.name.split('_')[-1]  
    perturbation_length = int(length_part.lstrip('l'))
    print(f"  - {run_dir.name} (start epoch: {epoch_number}, length: {perturbation_length})")

# sort the run_dirs by the epoch number, then by the perturbation length
run_dirs = sorted(run_dirs, key=lambda x: (int(x.name.split('_')[-2].lstrip('e')), int(x.name.split('_')[-1].lstrip('l'))))


def main(): 
    config = {
        'img_dir': '/home/wallacelab/teba/multimodal_brain_inspired/NOD/imagenet',  # input images directory,
        'category_index_file': '../analysis/sorted_file_categories.csv', 
        'load_hba': True,  # False will load the original CLIP-ViT weights
        'backbone': 'ViT-L/14',  # CLIP backbone model
        'model_path': '/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior/models/cliphba_behavior_20250919_212822.pth',  # path to the final trained model
        'batch_size': 64,  # batch size (increased for better GPU utilization)
        'cuda': 'cuda:1',  # 'cuda:0' for GPU 0, 'cuda:1' for GPU 1, '-1' for all GPUs
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
        training_res_path = run_dir / 'metrics.csv'

        print(f"Dora params path: {dora_params_path}")
        print(f"Training results path: {training_res_path}")
        print(f"Save folder: {save_folder}\n")

        # Create config for this specific run
        config['dora_params_path'] = str(dora_params_path)
        config['save_folder'] = str(save_folder)
        config['training_res_path'] = str(training_res_path) if training_res_path.exists() else None

        # Run inference with configuration
        try:
            run_behavior_inference(config)
            print(f"✓ Completed inference for {run_dir.name}\n")
        except Exception as e:
            print(f"✗ Error during inference for {run_dir.name}: {e}\n")
            continue

        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()