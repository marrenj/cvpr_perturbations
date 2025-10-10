from functions.cvpr_train_behavior_things_pipeline import run_behavioral_training
import torch.nn as nn
from datetime import datetime
import os
import logging
import sys

## THINGS THAT WILL BE THE SAME FOR ALL TRAINING RUNS:
# DoRA parameter checkpoint directory (also determines the number of training runs)
# Perturbation types: label_shuffle, random_target
# Perturb_length: 1 (one epoch per run)
# Perturb_distribution: normal (for random_target runs) or target (for label_shuffle runs)
# Seed: 42 for now
# Output folder base directory: results/run_$RUN_ID for saving all runs

## THINGS THAT WILL VARY BY TRAINING RUN:
# The specific naming convention of the file saved (this can be done within the training loop)


# |-- temporal_dynamics_of_human_alignment
# |   |-- clip_hba_behavior_loops
# |   |   |-- timestamp1
# |   |   |   |-- output log for entire loop
# |   |   |   |-- run1
# |   |   |   |   |-- output log for run
# |   |   |   |   |-- training_results
# |   |   |   |   |-- training_artifacts
# |   |   |   |   |    |-- dora_params
# |   |   |   |-- run2
# |   |   |   |   |-- output log for run
# |   |   |   |   |-- training_results
# |   |   |   |   |-- training_artifacts
# |   |   |   |   |    |-- dora_params
# |   |   |   |-- run93
# |   |   |   |   |-- output log for run
# |   |   |   |   |-- training_results
# |   |   |   |   |-- training_artifacts
# |   |   |   |   |    |-- dora_params
# |   |   |-- timestamp2
# |   |   |   |-- output log for entire loop
# |   |   |   |-- run1
# |   |   |   |   |-- output log for run
# |   |   |   |   |-- training_results
# |   |   |   |   |-- training_artifacts
# |   |   |   |   |    |-- dora_params
# |   |   |   |-- run2
# |   |   |   |   |-- output log for run
# |   |   |   |   |-- training_results
# |   |   |   |   |-- training_artifacts
# |   |   |   |   |    |-- dora_params
# |   |   |   |-- run93
# |   |   |   |   |-- output log for run
# |   |   |   |   |-- training_results
# |   |   |   |   |-- training_artifacts
# |   |   |   |   |    |-- dora_params

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
        'cuda': 1,  # -1 for all GPUs, 0 for GPU 0, 1 for GPU 1, 2 for CPU
        'baseline_dora_directory': '/home/wallacelab/teba/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior/training_artifacts/dora_params/dora_params_20251008_211424', # location of the DoRA parameters for the baseline training run
        'perturb_type': 'random_target', # either 'random_target' or 'label_shuffle'
        'perturb_distribution': 'normal', # draw from either the 'normal' or 'target' distribution when generating random targets (only used for random_target runs)
        'perturb_seed': 42, # seed for the random target generator
        'output_base_directory': f'/home/wallacelab/teba/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior_loops/{timestamp}', # base directory for saving the training results and artifacts
    }

    # Set up main logger for the entire loop
    main_log_path = os.path.join(config['output_base_directory'], f'main_training_log_{timestamp}.txt')
    main_logger = setup_main_logger(main_log_path)

    main_logger.info("="*80)
    main_logger.info("STARTING MAIN TRAINING LOOP - 93 TRAINING RUNS")
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

    for training_run in range(3, 94):  # set the training runs you'd like to loop through
        main_logger.info("-"*80)
        main_logger.info(f"TRAINING RUN {training_run}/93")
        main_logger.info(f"  Perturbing epoch: {training_run}")
        main_logger.info(f"  Resume from epoch: {training_run - 1}")
        
        config['training_run'] = training_run
        
        # create a subfolder in the output_base_directory for this training run
        training_run_directory = os.path.join(config['output_base_directory'], f'training_run{training_run}')
        os.makedirs(training_run_directory, exist_ok=True)
        config['checkpoint_path'] = os.path.join(training_run_directory, f'model_checkpoint_run{training_run}.pth')
        config['training_res_path'] = os.path.join(training_run_directory, f'training_res_run{training_run}.csv')
        config['dora_parameters_path'] = os.path.join(training_run_directory, f'dora_params_run{training_run}')
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
    main_logger.info(f"Total runs: 93")
    main_logger.info(f"Successful: {successful_runs}")
    main_logger.info(f"Failed: {failed_runs}")
    if failed_run_list:
        main_logger.info(f"Failed runs: {failed_run_list}")
    main_logger.info(f"Main log saved to: {main_log_path}")
    main_logger.info("="*80)

if __name__ == '__main__':
    main()