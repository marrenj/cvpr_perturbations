"""
Run 720 Things concepts inference across training runs using midpoint sweep ordering.
"""

from functions.things_720_concepts_inference_pipeline import run_behavior_inference, ImageDataset, cache_dataloader_to_pinned_memory
from torch.utils.data import DataLoader
from pathlib import Path
from functions.spose_dimensions import *
import torch
import sys


def validate_paths(config):
    results_dir = Path(config['results_dir'])
    img_dir = Path(config['img_dir'])
    stimuli_file = Path(config['stimuli_file'])

    errors = []
    if not results_dir.exists():
        errors.append(f"Results directory not found: {results_dir}")
    if not img_dir.exists():
        errors.append(f"Image directory not found: {img_dir}")
    if not stimuli_file.exists():
        errors.append(f"Stimuli file not found: {stimuli_file}")

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
        'results_dir': '/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/marren_image_noise_single_epoch_perturbation_sweeps',
        'img_dir': '/home/wallacelab/teba/multimodal_brain_inspired/THINGS_images',
        'stimuli_file': '/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/things_720concepts_stimulus_metadata.csv',
        'batch_size': 64,
        'cuda': 'cuda:1',
        'load_hba': True,
        'backbone': 'ViT-L/14',
        'min_epoch_to_process': None,  # Set to an integer to filter epochs, or None to process all
        'max_files_to_process': 1  # Only process the first file in each folder
    }

    results_dir, img_dir = validate_paths(config)

    run_dirs = sorted(
        [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("training_run")],
        key=lambda x: int(x.name.split('run')[1])
    )

    print(f"Found {len(run_dirs)} training runs:")
    for run_dir in run_dirs:
        print(f"  - {run_dir.name}")

    # midpoint_order = generate_midpoint_order(len(run_dirs))
    # print(f"Midpoint ordering for {len(run_dirs)} runs: {midpoint_order[:30]}")
    # print(f"Total runs to process: {len(midpoint_order)}")

    order = list(range(1, len(run_dirs) + 1))
    print(f"Total runs to process: {len(order)}")

    # Cache images once at the beginning for all runs
    print(f"\n{'='*80}")
    print("Caching images to memory...")
    print(f"{'='*80}\n")
    
    dataset = ImageDataset(
        stimuli_file=str(config['stimuli_file']),
        img_dir=str(img_dir)
    )
    num_workers = 2
    data_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else 2
    )
    
    print(f"Using {num_workers} workers for image loading...")
    cached_batches = cache_dataloader_to_pinned_memory(data_loader)
    del data_loader
    print(f"✓ Cached {len(cached_batches)} batches ({len(cached_batches) * config['batch_size']} images) to memory\n")

    inference_config = {
        'img_dir': str(img_dir),
        'stimuli_file': str(config['stimuli_file']),
        'load_hba': config['load_hba'],
        'backbone': config['backbone'],
        'batch_size': config['batch_size'],
        'cuda': config['cuda'],
        'min_epoch_to_process': config.get('min_epoch_to_process', None),
        'max_files_to_process': config.get('max_files_to_process', None)
    }

    for iteration, run_index in enumerate(order, 1):
        run_dir = run_dirs[run_index - 1]
        run_number = run_dir.name.replace('training_run', '')

        dora_params_path = run_dir / f'dora_params_run{run_number}'
        save_folder = run_dir / 'things_720_concepts_inference_results'
        training_res_path = run_dir / f'training_res_run{run_number}.csv'

        if not dora_params_path.exists():
            print(f"Skipping {run_dir.name}: dora_params directory not found")
            continue

        print(f"\n{'='*80}")
        print(f"Processing {run_dir.name} (Iteration {iteration}/{len(order)})")
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
            run_behavior_inference(inference_config, cached_batches=cached_batches)
            print(f"✓ Completed inference for {run_dir.name}\n")
        except Exception as e:
            print(f"✗ Error during inference for {run_dir.name}: {e}\n")
            continue

        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()

