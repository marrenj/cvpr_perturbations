from functions.cvpr_train_behavior_things_pipeline import run_behavioral_training
import torch.nn as nn
from datetime import datetime
import os
import logging
import sys
import argparse
import csv


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


def parse_args():
    """
    Parse command line arguments for SLURM integration.
    """
    parser = argparse.ArgumentParser(description='CVPR Behavior Training - SLURM Integration')
    parser.add_argument('--model', type=str, default='clip_hba', help='Model type')
    parser.add_argument('--perturb_type', type=str, default='random_target', 
                       choices=['random_target', 'label_shuffle', 'baseline'],
                       help='Perturbation type')
    parser.add_argument('--perturb_epoch', type=int, required=True, 
                       help='Epoch to perturb (0 for baseline)')
    parser.add_argument('--perturb_length', type=int, required=True, 
                       help='Length of perturbation (0 for baseline)')
    parser.add_argument('--perturb_distribution', type=str, default='target',
                       choices=['normal', 'target'],
                       help='Perturbation distribution')
    parser.add_argument('--perturb_seed', type=int, default=0, 
                       help='Perturbation seed')
    parser.add_argument('--output_dir', type=str, required=True, 
                       help='Output directory (e.g., output/random_target_e2_l2)')
    parser.add_argument('--cuda', type=int, default=1, 
                       help='CUDA device (-1 for all GPUs, 0 for GPU 0, 1 for GPU 1, 2 for CPU)')
    parser.add_argument('--epochs', type=int, default=500, 
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, 
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, 
                       help='Learning rate')
    parser.add_argument('--early_stopping_patience', type=int, default=20, 
                       help='Early stopping patience')
    parser.add_argument('--random_seed', type=int, default=1, 
                       help='Random seed for reproducibility')
    parser.add_argument('--baseline_dora_directory', type=str, required=True, 
                       help='Baseline DoRA directory')
    parser.add_argument('--baseline_random_state_path', type=str, required=True, 
                       help='Baseline random state path')
    parser.add_argument('--baseline_split_indices_path', type=str, required=True, 
                       help='Baseline split indices path')
    parser.add_argument('--output_base_directory', type=str, required=True, 
                       help='Output base directory')
    
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    # Generate unique timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define configuration
    config = {
        'csv_file': '../Data/spose_embedding66d_rescaled_1806train.csv', # csv data annotations of the training stimuli with the corresponding target embeddings
        'img_dir': '../Data/Things1854', # path to the image directory
        'inference_csv_file': '../Data/spose_embedding66d_rescaled_48val_reordered.csv', # csv data annotations of the inference stimuli with the corresponding target embeddings
        'RDM48_triplet_dir': '../Data/RDM48_triplet.mat', # location of the reference behavioral RDM
        'backbone': 'ViT-L/14', # CLIP backbone model, ViT-L/14 is the CLIP-HBA model default
        'epochs': args.epochs, 
        'batch_size': args.batch_size,
        'train_portion': 0.8,
        'lr': args.lr, # learning rate
        'logger': None,
        'early_stopping_patience': args.early_stopping_patience, # early stopping patience
        #'checkpoint_path': f'/home/wallacelab/teba/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior/models/cliphba_behavior_{timestamp}.pth', # path to save the trained model weights
        #'training_res_path': f'/home/wallacelab/teba/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior/training_results/training_res_{timestamp}.csv', # location to save the training results
        #'dora_parameters_path': f'/home/wallacelab/teba/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior/training_artifacts/dora_params/dora_params_{timestamp}', # location to save the DoRA parameters
        'random_seed': args.random_seed, # random seed, default CLIP-HBA is 1
        'vision_layers': 2, # Last n ViT layers of the model to be trained, default CLIP-HBA-Behavior is 2
        'transformer_layers': 1, # Last n transformer layers to be trained, default CLIP-HBA-Behavior is 1
        'rank': 32, # Rank of the feature reweighting matrix, default CLIP-HBA-Behavior is 32
        'criterion': nn.MSELoss(), # MSE Loss
        'cuda': args.cuda,  # -1 for all GPUs, 0 for GPU 0, 1 for GPU 1, 2 for CPU
        'baseline_dora_directory': args.baseline_dora_directory, # location of the DoRA parameters for the baseline training run
        'baseline_random_state_path': args.baseline_random_state_path, # location of the random states for the baseline training run
        'baseline_split_indices_path': args.baseline_split_indices_path, # location of the train/test split indices from baseline training
        'perturb_type': args.perturb_type, # either 'random_target', 'label_shuffle', or 'baseline'
        'perturb_distribution': args.perturb_distribution, # draw from either the 'normal' or 'target' distribution when generating random targets (only used for random_target runs)
        'perturb_seed': args.perturb_seed, # seed for the random target generator
        'training_run': args.perturb_epoch, # the epoch to train from
        'resume_from_epoch': max(0, args.perturb_epoch - 1),  # Ensure non-negative
        'output_base_directory': args.output_base_directory, # base directory for saving the training results and artifacts
        'output_directory': args.output_dir, # output directory for saving the training results and artifacts
    }

    # Create output directory structure compatible with analysis script
    config['output_dir'] = os.path.join(config['output_base_directory'], config['output_directory'])

    # Set up paths to match expected structure for analysis script
    config['checkpoint_path'] = os.path.join(config['output_dir'], f'model_checkpoint_{args.perturb_epoch}.pth')
    config['training_res_path'] = os.path.join(config['output_dir'], 'training_res.csv')
    config['dora_parameters_path'] = os.path.join(config['output_dir'], f'dora_params_{args.perturb_epoch}')
    config['random_state_path'] = os.path.join(config['output_dir'], f'random_states_{args.perturb_epoch}')

    # Check if this run already exists and find the last completed epoch
    existing_csv = config['training_res_path']
    resume_from_existing = False
    last_completed_epoch = -1
    
    if os.path.exists(existing_csv):
        # Read the CSV to find the last completed epoch
        try:
            with open(existing_csv, 'r') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                for row in reader:
                    if row and len(row) > 0:
                        try:
                            epoch_val = int(row[0])  # CSV stores 1-indexed epochs
                            last_completed_epoch = max(last_completed_epoch, epoch_val - 1)  # Convert to 0-indexed
                        except (ValueError, IndexError):
                            continue
            if last_completed_epoch >= 0:
                resume_from_existing = True
                print(f"Detected existing training run. Last completed epoch: {last_completed_epoch + 1}")
                print(f"Resuming from epoch {last_completed_epoch + 1}")
        except Exception as e:
            print(f"Warning: Could not read existing CSV file: {e}")
            print("Starting fresh training run.")
    
    # Create output directory if it doesn't exist
    os.makedirs(config['output_dir'], exist_ok=True)

    # Set up logger
    log_file = os.path.join(config['output_dir'], f'training_log_{timestamp}.txt')
    logger = setup_main_logger(log_file)

    logger.info("="*80)
    logger.info("STARTING SINGLE TRAINING RUN - SLURM INTEGRATION")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Perturbation Type: {args.perturb_type}")
    logger.info(f"Perturbation Epoch: {args.perturb_epoch}")
    logger.info(f"Perturbation Length: {args.perturb_length}")
    logger.info(f"Perturbation Distribution: {args.perturb_distribution}")
    logger.info(f"Perturbation Seed: {args.perturb_seed}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"CUDA Device: {args.cuda}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Learning Rate: {args.lr}")
    logger.info(f"Random Seed: {args.random_seed}")
    logger.info(f"Baseline Checkpoints: {config['baseline_dora_directory']}")
    logger.info("="*80)
    logger.info("")

    # Helper: attempt to find a previous run dir with same starting epoch and smaller perturb length
    def find_previous_run_dir(base_dir, perturb_type, start_epoch, current_length):
        candidates = []
        if not os.path.isdir(base_dir):
            return None, None
        for name in os.listdir(base_dir):
            full_path = os.path.join(base_dir, name)
            if not os.path.isdir(full_path):
                continue
            # Require matching start epoch token 'e{start_epoch}_' and matching perturb_type prefix if present
            if f"e{start_epoch}_" not in name:
                continue
            if perturb_type in ['random_target', 'label_shuffle'] and not name.startswith(perturb_type):
                continue
            # Extract length from pattern '_l{num}'
            length_val = None
            try:
                parts = name.split('_')
                for p in parts:
                    if p.startswith('l') and p[1:].isdigit():
                        length_val = int(p[1:])
                        break
                if length_val is None:
                    continue
            except Exception:
                continue
            if length_val < current_length:
                candidates.append((length_val, full_path))
        if not candidates:
            return None, None
        best = max(candidates, key=lambda t: t[0])
        return best[1], best[0]

    # Handle baseline case
    if args.perturb_type == 'baseline':
        logger.info("Running baseline training (no perturbations)")
        config['perturb_type'] = 'baseline'
        config['perturb_epoch'] = 0
        config['perturb_length'] = 0
        if resume_from_existing:
            config['resume_from_epoch'] = last_completed_epoch + 1  # Resume from next epoch after last completed
            logger.info(f"Resuming existing baseline run from epoch {config['resume_from_epoch'] + 1}")
        else:
            config['resume_from_epoch'] = 0
    else:
        logger.info(f"Running perturbation training:")
        logger.info(f"  - Perturbing at epoch: {args.perturb_epoch}")
        logger.info(f"  - Perturbation length: {args.perturb_length}")
        config['perturb_length'] = args.perturb_length
        
        if resume_from_existing:
            # Resume from the existing run
            config['resume_from_epoch'] = last_completed_epoch + 1  # Resume from next epoch after last completed
            config['previous_training_res_path'] = config['training_res_path']  # Use same CSV file
            config['resume_random_state_path'] = config['random_state_path']
            config['resume_dora_parameters_path'] = config['dora_parameters_path']
            logger.info(f"Resuming existing run from epoch {config['resume_from_epoch'] + 1}")
        else:
            # Try to resume from the latest existing perturbation with same starting epoch but smaller length
            prev_dir, prev_length = find_previous_run_dir(config['output_base_directory'], args.perturb_type, args.perturb_epoch, args.perturb_length)
            if prev_dir and prev_length is not None:
                last_epoch = max(0, args.perturb_epoch - 1) + prev_length
                config['resume_from_epoch'] = last_epoch
                config['previous_training_res_path'] = os.path.join(prev_dir, 'training_res.csv')
                config['resume_random_state_path'] = os.path.join(prev_dir, f'random_states_{args.perturb_epoch}')
                config['resume_dora_parameters_path'] = os.path.join(prev_dir, f'dora_params_{args.perturb_epoch}')
                logger.info(f"Detected previous run at '{prev_dir}' with length {prev_length}; resuming from epoch {last_epoch + 1}")
            else:
                logger.info("No previous matching run found; starting from baseline epoch.")
                logger.info(f"  - Resuming from epoch: {config['resume_from_epoch'] + 1}")

    try:
        logger.info(f"Starting training run...")
        logger.info(f"Configuration summary:")
        for key, value in config.items():
            if key not in ['criterion']:  # Skip printing the criterion object
                logger.info(f"  {key}: {value}")
        logger.info("")
        
        run_behavioral_training(config)
        
        logger.info("="*80)
        logger.info("✓ TRAINING RUN COMPLETED SUCCESSFULLY")
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info(f"Log file: {log_file}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error("="*80)
        logger.error("✗ TRAINING RUN FAILED")
        logger.error(f"Error: {str(e)}")
        logger.error(f"Log file: {log_file}")
        logger.error("="*80)
        raise

if __name__ == '__main__':
    main()