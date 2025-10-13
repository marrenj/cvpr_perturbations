from functions.cvpr_train_behavior_things_pipeline import run_behavioral_training
import torch.nn as nn
from datetime import datetime
import os
import logging
import sys

def setup_test_logger(log_file_path):
    """
    Set up logger for the reproducibility test.
    """
    # Create logger
    logger = logging.getLogger('reproducibility_test')
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
    """
    Test reproducibility by resuming from a baseline checkpoint and running
    a few epochs without any perturbations. The results should exactly match
    the corresponding epochs from the baseline training run.
    
    Usage:
        python test_reproducibility.py
    
    Then compare the results in the output CSV with your baseline training CSV.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ========================================================================
    # CONFIGURATION - MODIFY THESE VALUES FOR YOUR TEST
    # ========================================================================
    
    # Which epoch to resume from (e.g., 5 means load epoch5 checkpoint and run epoch 6, 7, 8)
    RESUME_FROM_EPOCH = 1
    
    # How many epochs to run (should be > RESUME_FROM_EPOCH)
    # If RESUME_FROM_EPOCH=5 and TOTAL_EPOCHS=8, will run epochs 6, 7, 8
    TOTAL_EPOCHS = 5
    
    # Path to your baseline DoRA parameters directory
    BASELINE_DORA_DIR = '/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior/training_artifacts/dora_params/dora_params_20251013_143326'
    
    # Path to your baseline random states directory (NEW: for training_artifacts structure)
    # If your baseline used the old structure, set this to None and specify BASELINE_CHECKPOINT_DIR instead
    BASELINE_RANDOM_STATES_DIR = '/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior/training_artifacts/random_states/random_states_20251013_143326'
    
    # Path to your baseline checkpoint directory (for backward compatibility with old structure)
    # Only used if BASELINE_RANDOM_STATES_DIR is None
    BASELINE_CHECKPOINT_DIR = '/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior/models'
    
    # Output directory for test results
    OUTPUT_DIR = f'/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior_reproducibility_test/{timestamp}'
    
    # Path to your baseline training results CSV (for comparison instructions)
    BASELINE_RESULTS_CSV = '/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior/training_results/training_res_20251013_143326.csv'
    
    # ========================================================================
    
    # Validate configuration
    if TOTAL_EPOCHS <= RESUME_FROM_EPOCH:
        raise ValueError(f"TOTAL_EPOCHS ({TOTAL_EPOCHS}) must be > RESUME_FROM_EPOCH ({RESUME_FROM_EPOCH})")
    
    # Check if checkpoint files exist
    dora_checkpoint = os.path.join(BASELINE_DORA_DIR, f'epoch{RESUME_FROM_EPOCH}_dora_params.pth')
    
    # Check for random states in new structure first, then fall back to old structure
    if BASELINE_RANDOM_STATES_DIR is not None:
        random_checkpoint = os.path.join(BASELINE_RANDOM_STATES_DIR, f'epoch{RESUME_FROM_EPOCH}_random_states.pth')
    else:
        random_checkpoint = os.path.join(BASELINE_CHECKPOINT_DIR, f'epoch{RESUME_FROM_EPOCH}_random_states.pth')
    
    if not os.path.exists(dora_checkpoint):
        raise FileNotFoundError(f"DoRA checkpoint not found: {dora_checkpoint}")
    if not os.path.exists(random_checkpoint):
        raise FileNotFoundError(f"Random states checkpoint not found: {random_checkpoint}")
    
    # Create test configuration
    test_config = {
        'csv_file': '../Data/spose_embedding66d_rescaled_1806train.csv',
        'img_dir': '../Data/Things1854',
        'inference_csv_file': '../Data/spose_embedding66d_rescaled_48val_reordered.csv',
        'RDM48_triplet_dir': '../Data/RDM48_triplet.mat',
        'backbone': 'ViT-L/14',
        'epochs': TOTAL_EPOCHS,
        'batch_size': 64,
        'train_portion': 0.8,
        'lr': 3e-4,
        'logger': None,
        'early_stopping_patience': 999,  # Set high to prevent early stopping during test
        'random_seed': 1,
        'vision_layers': 2,
        'transformer_layers': 1,
        'rank': 32,
        'criterion': nn.MSELoss(),
        'cuda': 0,  # Change this if you want to use a different GPU
        'baseline_dora_directory': BASELINE_DORA_DIR,
        'baseline_random_states_directory': BASELINE_RANDOM_STATES_DIR,
        'baseline_checkpoint_directory': BASELINE_CHECKPOINT_DIR,
        'perturb_type': 'none',  # NO PERTURBATIONS - just resume normal training
        'perturb_distribution': 'normal',  # Not used when perturb_type='none'
        'perturb_seed': 42,  # Not used when perturb_type='none'
        'training_run': RESUME_FROM_EPOCH + 1,  # Needs to be > 0 for loading logic
        'resume_from_epoch': RESUME_FROM_EPOCH,
        'output_base_directory': OUTPUT_DIR,
        'checkpoint_path': os.path.join(OUTPUT_DIR, 'test_checkpoint.pth'),
        'training_res_path': os.path.join(OUTPUT_DIR, 'test_training_res.csv'),
        'dora_parameters_path': os.path.join(OUTPUT_DIR, 'test_dora_params'),
        'random_states_path': os.path.join(OUTPUT_DIR, 'test_random_states'),
    }
    
    # Set up logger
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    test_log_path = os.path.join(OUTPUT_DIR, f'reproducibility_test_log_{timestamp}.txt')
    test_logger = setup_test_logger(test_log_path)
    
    # Print test information
    test_logger.info("="*80)
    test_logger.info("REPRODUCIBILITY TEST - NO PERTURBATIONS")
    test_logger.info("="*80)
    test_logger.info("")
    test_logger.info("Test Configuration:")
    test_logger.info(f"  Resume from epoch: {RESUME_FROM_EPOCH}")
    test_logger.info(f"  Will run epochs: {RESUME_FROM_EPOCH + 1} through {TOTAL_EPOCHS}")
    test_logger.info(f"  Total epochs to run: {TOTAL_EPOCHS - RESUME_FROM_EPOCH}")
    test_logger.info(f"  Perturbation type: {test_config['perturb_type']} (no perturbations)")
    test_logger.info("")
    test_logger.info("Checkpoint Files:")
    test_logger.info(f"  DoRA params: {dora_checkpoint}")
    test_logger.info(f"  Random states: {random_checkpoint}")
    test_logger.info("")
    test_logger.info("Output Directory:")
    test_logger.info(f"  {OUTPUT_DIR}")
    test_logger.info("")
    test_logger.info("="*80)
    test_logger.info("")
    
    # Run the test
    try:
        test_logger.info("Starting reproducibility test...")
        test_logger.info("")
        run_behavioral_training(test_config)
        test_logger.info("")
        test_logger.info("="*80)
        test_logger.info("✓ REPRODUCIBILITY TEST COMPLETED SUCCESSFULLY!")
        test_logger.info("="*80)
        test_logger.info("")
        test_logger.info("VERIFICATION STEPS:")
        test_logger.info("-" * 80)
        test_logger.info("")
        test_logger.info(f"1. Open your baseline training results CSV:")
        test_logger.info(f"   {BASELINE_RESULTS_CSV}")
        test_logger.info("")
        test_logger.info(f"2. Open your test results CSV:")
        test_logger.info(f"   {test_config['training_res_path']}")
        test_logger.info("")
        test_logger.info(f"3. Compare epochs {RESUME_FROM_EPOCH + 1} through {TOTAL_EPOCHS}:")
        test_logger.info(f"   - Training losses should be IDENTICAL")
        test_logger.info(f"   - Validation losses should be IDENTICAL")
        test_logger.info(f"   - Behavioral RSA correlations should be IDENTICAL")
        test_logger.info("")
        test_logger.info("4. If all values match (within ~1e-7 precision), reproducibility is verified!")
        test_logger.info("")
        test_logger.info("Example comparison command:")
        test_logger.info(f"   # Show baseline epochs {RESUME_FROM_EPOCH + 1}-{TOTAL_EPOCHS}")
        test_logger.info(f"   head -n {TOTAL_EPOCHS + 1} BASELINE_CSV | tail -n {TOTAL_EPOCHS - RESUME_FROM_EPOCH}")
        test_logger.info("")
        test_logger.info(f"   # Show test epochs {RESUME_FROM_EPOCH + 1}-{TOTAL_EPOCHS}")
        test_logger.info(f"   head -n {TOTAL_EPOCHS - RESUME_FROM_EPOCH + 1} {test_config['training_res_path']} | tail -n {TOTAL_EPOCHS - RESUME_FROM_EPOCH}")
        test_logger.info("")
        test_logger.info("="*80)
        
    except Exception as e:
        test_logger.error("")
        test_logger.error("="*80)
        test_logger.error("✗ REPRODUCIBILITY TEST FAILED!")
        test_logger.error("="*80)
        test_logger.error(f"Error: {str(e)}")
        test_logger.error("")
        import traceback
        test_logger.error("Full traceback:")
        test_logger.error(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()

