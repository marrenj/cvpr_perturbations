from functions.nod_inference_pipeline import run_behavior_inference
from pathlib import Path
from functions.spose_dimensions import *
import torch

results_dir = Path('/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior_loops/20251016_125025')

# list all the run directories in the results_dir
run_dirs = sorted([d for d in results_dir.glob('training_run*') if d.is_dir()],
                  key=lambda x: int(x.name.split('run')[1]))

print(f"Found {len(run_dirs)} training runs:")
for run_dir in run_dirs:
    print(f"  - {run_dir.name}")


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

# Generate midpoint ordering for the runs
midpoint_order = generate_midpoint_order(len(run_dirs))
print(f"Midpoint ordering for {len(run_dirs)} runs: {midpoint_order[:30]}")  # Show first 30
print(f"Total runs to process: {len(midpoint_order)}")

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

    # Use midpoint ordering to process runs
    for iteration, run_index in enumerate(midpoint_order, 1):
        run_dir = run_dirs[run_index - 1]  # Convert to 0-based index
        run_number = run_dir.name.replace('training_run', '')

        # Construct paths specific to this training run
        dora_params_path = run_dir / f'dora_params_run{run_number}'
        save_folder = run_dir / 'nod_inference_results'
        training_res_path = run_dir / f'training_res_run{run_number}.csv'

        # Check if dora params exist
        if not dora_params_path.exists():
            print(f"Skipping {run_dir.name}: dora_params directory not found")
            continue

        print(f"\n{'='*80}")
        print(f"Processing {run_dir.name} (Iteration {iteration}/{len(midpoint_order)})")
        print(f"Dora params path: {dora_params_path}")
        print(f"Training results path: {training_res_path}")
        print(f"Save folder: {save_folder}")
        print(f"{'='*80}\n")

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