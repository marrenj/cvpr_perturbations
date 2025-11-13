"""
Run 48 Things images inference across training runs using midpoint sweep ordering.
"""

from functions.things_48_images_inference_pipeline import run_behavior_inference
from pathlib import Path
from functions.spose_dimensions import *
import torch
import sys


def validate_paths(config):
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


def generate_midpoint_order(num_runs):
    """
    Generate a midpoint/binary search ordering for exploring training runs.
    Starts with boundaries [1, num_runs], then fills in midpoints.
    """
    visited = set()
    ordering = []
    queue = [(1, num_runs)]  # Start with the full range
    
    while queue:
        low, high = queue.pop(0)
        mid = (low + high) // 2
        
        # Process endpoints first if not already visited
        if low not in visited:
            ordering.append(low)
            visited.add(low)
        if high not in visited:
            ordering.append(high)
            visited.add(high)
        
        # Add midpoint if it exists and hasn't been visited
        if low < mid < high and mid not in visited:
            ordering.append(mid)
            visited.add(mid)
            
            # Add sub-ranges to queue for further midpoint processing
            if low < mid - 1:
                queue.append((low, mid))
            if mid < high - 1:
                queue.append((mid, high))
    
    return ordering

def main():
    # Define configuration directly
    config = {
        'results_dir': '/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/single_epoch_perturbation_sweeps/perturb_sweep_baselineseed3_perturbseed44',
        'img_dir': '../Data/Things1854',
        'inference_csv_file': '../Data/spose_embedding66d_rescaled_48val_reordered.csv',
        'RDM48_triplet_dir': '../Data/RDM48_triplet.mat',
        'batch_size': 48,
        'cuda': 'cuda:0',
        'load_hba': True,
        'backbone': 'ViT-L/14'
    }

    results_dir, img_dir = validate_paths(config)

    run_dirs = sorted(
        [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("training_run")],
        key=lambda x: int(x.name.split('run')[1])
    )

    print(f"Found {len(run_dirs)} training runs:")
    for run_dir in run_dirs:
        print(f"  - {run_dir.name}")

    midpoint_order = generate_midpoint_order(len(run_dirs))
    print(f"Midpoint ordering for {len(run_dirs)} runs: {midpoint_order[:30]}")
    print(f"Total runs to process: {len(midpoint_order)}")

    inference_config = {
        'img_dir': str(img_dir),
        'inference_csv_file': str(config['inference_csv_file']),
        'RDM48_triplet_dir': str(config['RDM48_triplet_dir']),
        'load_hba': config['load_hba'],
        'backbone': config['backbone'],
        'batch_size': config['batch_size'],
        'cuda': config['cuda'],
    }

    for iteration, run_index in enumerate(midpoint_order, 1):
        run_dir = run_dirs[run_index - 1]
        run_number = run_dir.name.replace('training_run', '')

        dora_params_path = run_dir / f'dora_params_run{run_number}'
        save_folder = run_dir / 'things_48_inference_results'
        training_res_path = run_dir / f'training_res_run{run_number}.csv'

        if not dora_params_path.exists():
            print(f"Skipping {run_dir.name}: dora_params directory not found")
            continue

        print(f"\n{'='*80}")
        print(f"Processing {run_dir.name} (Iteration {iteration}/{len(midpoint_order)})")
        print(f"Dora params path: {dora_params_path}")
        print(f"Training results path: {training_res_path}")
        print(f"Save folder: {save_folder}")
        print(f"{'='*80}\n")

        metric_files = [f for f in run_dir.iterdir() if f.is_file() and
                        (f.name.startswith('metrics') or f.name.startswith('training_res_run'))]
        training_res_path = metric_files[0] if metric_files else None

        inference_config['dora_params_path'] = str(dora_params_path)
        inference_config['save_folder'] = str(save_folder)
        inference_config['training_res_path'] = str(training_res_path) if training_res_path and training_res_path.exists() else None

        try:
            run_behavior_inference(inference_config)
            print(f"✓ Completed inference for {run_dir.name}\n")
        except Exception as e:
            print(f"✗ Error during inference for {run_dir.name}: {e}\n")
            continue

        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()

