"""
Run NIGHTS inference across training runs using midpoint sweep ordering.
"""

from functions.nights_inference_pipeline import run_nights_inference
from pathlib import Path
from functions.spose_dimensions import *
import torch
import json
import sys
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent.resolve()


def load_config(config_path=None):
    default_config = {
        'results_dir': '/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior_loops/20251016_125025',
        'nights_dir': '/home/wallacelab/teba/multimodal_brain_inspired/dreamsim/dreamsim/nights',
        'batch_size': 256,
        'device': 'cuda:1',
        'load_hba': True,
        'backbone': 'ViT-L/14',
        'splits': ['test'],
        'use_image_features': False,
        'save_predictions': False
    }

    if config_path is None:
        config_path = SCRIPT_DIR / 'config_sweep.json'
    else:
        config_path = Path(config_path)

    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
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
    results_dir = Path(config['results_dir'])
    nights_dir = Path(config['nights_dir'])

    errors = []
    if not results_dir.exists():
        errors.append(f"Results directory not found: {results_dir}")
    if not nights_dir.exists():
        errors.append(f"NIGHTS directory not found: {nights_dir}")

    if errors:
        raise FileNotFoundError("\n".join(errors))

    return results_dir, nights_dir


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
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    file_config = load_config(config_path)

    results_dir, nights_dir = validate_paths(file_config)

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

    base_inference_config = {
        'nights_dir': str(nights_dir),
        'load_hba': file_config['load_hba'],
        'backbone': file_config['backbone'],
        'batch_size': file_config['batch_size'],
        'device': file_config['device'],
        'splits': file_config.get('splits', ['test']),
        'use_image_features': file_config.get('use_image_features', False),
        'save_predictions': file_config.get('save_predictions', False)
    }

    for iteration, run_index in enumerate(midpoint_order, 1):
        run_dir = run_dirs[run_index - 1]
        run_number = run_dir.name.replace('training_run', '')

        dora_params_path = run_dir / f'dora_params_run{run_number}'
        save_folder = run_dir / 'nights_inference_results'
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

        # Create save folder
        save_folder.mkdir(parents=True, exist_ok=True)

        # Create inference config - pass directory to let run_nights_inference loop through epochs
        inference_config = base_inference_config.copy()
        inference_config['dora_params_path'] = str(dora_params_path)
        inference_config['output_dir'] = str(save_folder)
        inference_config['training_res_path'] = str(training_res_path) if training_res_path and training_res_path.exists() else None

        try:
            # run_nights_inference will loop through all epochs in dora_params_path
            run_nights_inference(inference_config)
            print(f"✓ Completed all epochs for {run_dir.name}\n")
        except Exception as e:
            print(f"✗ Error during inference for {run_dir.name}: {e}\n")
            import traceback
            traceback.print_exc()

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

