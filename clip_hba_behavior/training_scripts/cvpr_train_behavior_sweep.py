from functions.new_cvpr_train_behavior_things_pipeline import run_behavioral_training
import torch.nn as nn
from datetime import datetime
import os
import logging
import sys

def generate_midpoint_order(start=1, end=98):
    """
    Generate training order that fills in midpoints progressively.
    Starts with 1, 98, 49, then recursively fills in intervals.
    
    Returns a list of epochs in the order they should be trained.
    """

    if start > end:
        return []

    epochs = []
    # Use a queue to track intervals to process (breadth-first approach)
    from collections import deque

    # Start with the first and last epochs
    epochs.append(start)
    if start != end:
        epochs.append(end)

    # Calculate and add the middle
    mid = (start + end) // 2
    if mid != start and mid != end:
        epochs.append(mid)

    # Now process intervals recursively using a queue
    queue = deque()
    if mid > start + 1:
        queue.append((start, mid))
    if end > mid + 1:
        queue.append((mid, end))

    while queue:
        left, right = queue.popleft()
        new_mid = (left + right) // 2
        # Only add if it's not already one of the boundaries
        if new_mid != left and new_mid != right:
            epochs.append(new_mid)
            # Add new intervals to process
            if new_mid > left + 1:
                queue.append((left, new_mid))
            if right > new_mid + 1:
                queue.append((new_mid, right))

    return epochs


def generate_hybrid_training_order():
    """
    Generate training order: epochs 1-15 sequentially, then midpoint order starting with 16, 98.
    """
    # Sequential order for epochs 1-15
    sequential_order = list(range(1, 16))  # [1, 2, 3, ..., 15]
    
    # Midpoint order starting from 16 and 98
    midpoint_order = generate_midpoint_order(start=16, end=98)
    
    # Combine the orders
    hybrid_order = sequential_order + midpoint_order
    
    return hybrid_order


def generate_sweep_training_order():
    """
    Generate training order: epochs 1-40 sequentially, then every 3 epochs thereafter.
    """
    # Sequential order for epochs 1-40, then every 3 epochs thereafter
    sequential_order = list(range(1, 41)) + list(range(41, 103, 3))
    
    return sequential_order


def setup_main_logger(log_file_path):
    """
    Set up logger for the main training loop to track all 93 runs.
    """
    # Create logger
    logger = logging.getLogger('main_training_loop')
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                datefmt='%Y-%m-%d %H:%M:%S')
    
    # File handler
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler (so you can see progress in terminal too)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def main():

    # Generate unique timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define configuration
    config = {
        'csv_file': '../Data/spose_embedding66d_rescaled_1806train.csv', # csv data annotations of the training stimuli with the corresponding target embeddings
        'img_dir': '../Data/Things1854', # path to the image directory
        'inference_csv_file': '../Data/spose_embedding66d_rescaled_48val_reordered.csv', # csv data annotations of the inference stimuli with the corresponding target embeddings
        'RDM48_triplet_dir': '../Data/RDM48_triplet.mat', # location of the reference behavioral RDM
        'backbone': 'ViT-L/14', # CLIP backbone model, ViT-L/14 is the CLIP-HBA model default
        'epochs': 500, 
        'batch_size': 64,
        'train_portion': 0.8,
        'lr': 3e-4, # learning rate
        'logger': None,
        'early_stopping_patience': 20, # early stopping patience
        #'checkpoint_path': f'/home/wallacelab/teba/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior/models/cliphba_behavior_{timestamp}.pth', # path to save the trained model weights
        #'training_res_path': f'/home/wallacelab/teba/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior/training_results/training_res_{timestamp}.csv', # location to save the training results
        #'dora_parameters_path': f'/home/wallacelab/teba/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior/training_artifacts/dora_params/dora_params_{timestamp}', # location to save the DoRA parameters
        'random_seed': 1, # random seed, default CLIP-HBA is 1
        'vision_layers': 2, # Last n ViT layers of the model to be trained, default CLIP-HBA-Behavior is 2
        'transformer_layers': 1, # Last n transformer layers to be trained, default CLIP-HBA-Behavior is 1
        'rank': 32, # Rank of the feature reweighting matrix, default CLIP-HBA-Behavior is 32
        'criterion': nn.MSELoss(), # MSE Loss
        'cuda': 0,  # -1 for all GPUs, 0 for GPU 0, 1 for GPU 1, 2 for CPU
        'baseline_dora_directory': '/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/baseline_runs/clip_hba_behavior_seed1/training_artifacts/dora_params/dora_params_seed1', # location of the DoRA parameters for the baseline training run
        'baseline_random_state_path': '/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/baseline_runs/clip_hba_behavior_seed1/training_artifacts/random_states/random_states_seed1', # location of the random states for the baseline training run
        'baseline_split_indices_path': '/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/baseline_runs/clip_hba_behavior_seed1/training_artifacts/random_states/random_states_seed1/dataset_split_indices.pth', # location of the train/test split indices from baseline training
        'perturb_type': 'uniform_images', # either 'random_target', 'label_shuffle', 'uniform_images', 'image_noise'
        'perturb_length': 1, # length of the perturbation window in epochs
        'perturb_distribution': 'target', # draw from either the 'normal' or 'target' distribution when generating random targets (only used for random_target runs)
        'perturb_seed': 42, # seed for the random target generator
        'output_base_directory': f'/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/uniform_images_single_epoch_perturbation_sweeps/perturb_sweep_baselineseed1_perturbseed42_{timestamp}', # base directory for saving the training results and artifacts
    }

    # Set up main logger for the entire loop
    main_log_path = os.path.join(config['output_base_directory'], f'main_training_log_{timestamp}.txt')
    main_logger = setup_main_logger(main_log_path)

    main_logger.info("="*80)
    main_logger.info("STARTING MAIN TRAINING LOOP - MIDPOINT-BASED TRAINING ORDER")
    main_logger.info(f"Timestamp: {timestamp}")
    main_logger.info(f"Perturbation Type: {config['perturb_type']}")
    main_logger.info(f"Perturbation Distribution: {config['perturb_distribution']}")
    main_logger.info(f"Perturbation Seed: {config['perturb_seed']}")
    main_logger.info(f"Output Directory: {config['output_base_directory']}")
    main_logger.info(f"Baseline Checkpoints: {config['baseline_dora_directory']}")
    main_logger.info("="*80)
    main_logger.info("")

    # Track statistics
    successful_runs = 0
    failed_runs = 0
    failed_run_list = []

    # # Generate the midpoint-based training order
    # training_order = generate_midpoint_order(start=1, end=98)

    # # # Generate the sweep-based training order
    # training_order = generate_sweep_training_order()
    # training_order = [8, 39, 40, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80, 83, 86, 89, 92]
    # print(f"Sweep training order: {training_order}")

    # # Generate full order but only run from 98 onwards
    # full_order = generate_midpoint_order(start=1, end=98)
    # # Find where 98 appears and slice from there
    # start_idx = full_order.index(98)
    # training_order = full_order[start_idx:]

    # # Generate hybrid training order
    # training_order = generate_hybrid_training_order()

    training_order = [15, 25, 35, 70]
    
    main_logger.info(f"Training order (first 20 epochs): {training_order[:20]}")
    main_logger.info(f"Total epochs to train: {len(training_order)}")
    main_logger.info("")

    for idx, training_run in enumerate(training_order, 1):  # Loop through midpoint order
        main_logger.info("-"*80)
        main_logger.info(f"TRAINING RUN {idx}/{len(training_order)} (Epoch {training_run})")
        main_logger.info(f"  Perturbing epoch: {training_run}")
        main_logger.info(f"  Resume from epoch: {training_run - 1}")
        
        config['training_run'] = training_run
        
        # create a subfolder in the output_base_directory for this training run
        training_run_directory = os.path.join(config['output_base_directory'], f'training_run{training_run}')
        os.makedirs(training_run_directory, exist_ok=True)
        config['checkpoint_path'] = os.path.join(training_run_directory, f'model_checkpoint_run{training_run}.pth')
        config['training_res_path'] = os.path.join(training_run_directory, f'training_res_run{training_run}.csv')
        config['dora_parameters_path'] = os.path.join(training_run_directory, f'dora_params_run{training_run}')
        config['random_state_path'] = os.path.join(training_run_directory, f'random_states_run{training_run}')
        config['resume_from_epoch'] = training_run - 1

        try:
            main_logger.info(f"  Starting training run {training_run}...")
            run_behavioral_training(config)
            successful_runs += 1
            main_logger.info(f"  ✓ Training run {training_run} completed successfully")
            main_logger.info(f"  Progress: {successful_runs} successful, {failed_runs} failed")
        except Exception as e:
            failed_runs += 1
            failed_run_list.append(training_run)
            main_logger.error(f"  ✗ Training run {training_run} FAILED with error:")
            main_logger.error(f"  {str(e)}")
            main_logger.error(f"  Progress: {successful_runs} successful, {failed_runs} failed")
            # Optionally: decide whether to continue or stop
            # continue  # Continue to next run
            # raise     # Stop entire loop
        
        main_logger.info("-"*80)
        main_logger.info("")

    # Final summary
    main_logger.info("="*80)
    main_logger.info("MAIN TRAINING LOOP COMPLETED")
    main_logger.info(f"Total runs: {len(training_order)}")
    main_logger.info(f"Successful: {successful_runs}")
    main_logger.info(f"Failed: {failed_runs}")
    if failed_run_list:
        main_logger.info(f"Failed runs: {failed_run_list}")
    main_logger.info(f"Main log saved to: {main_log_path}")
    main_logger.info("="*80)

if __name__ == '__main__':
    main()