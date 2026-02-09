"""
A python script that runs experiments sequentially. 
Contains error handling and logging.
"""

import yaml
import itertools
import logging
import traceback
from datetime import datetime
from pathlib import Path
from src.training.trainer import run_training_experiment

def setup_experiment_logger(log_dir):
    """Setup logger for the experiment queue."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"experiment_queue_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def run_sequential_experiments():
    """
    Run experiments with error handling and progress tracking.
    """
    
    # Setup logging
    logger = setup_experiment_logger("./logs/experiment_queues")
    
    # Load base config
    with open('configs/training_config.yaml', 'r') as f:
        base_config = yaml.safe_load(f)

    # Define parameter variations
    param_grid = {
        'perturb_type': ['none', 'random_target', 'label_shuffle', 'image_noise', 'uniform_images'],
        'perturb_epoch': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 89, 94, 99],
        'perturb_length': [1, 5, 10, 20, 30, 40, 50],
        'random_seed': [1, 2, 3, 4, 5],
    }

    # Generate all combinations of parameter variations
    keys = param_grid.keys()
    values = param_grid.values()
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Total experiments to run: {len(experiments)}")

    # Print the first 5 experiments
    for i in range(5):
        print(experiments[i])
    
    # Track results
    results = {
        'completed': [],
        'failed': [],
        'skipped': []
    }
    
    logger.info(f"Starting experiment queue: {len(experiments)} experiments")
    
    for idx, exp_params in enumerate(experiments, 1):
        exp_name = exp_params.get('description', exp_params['perturb_type'])
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Experiment {idx}/{len(experiments)}: {exp_name}")
        logger.info(f"Parameters: {exp_params}")
        logger.info(f"{'='*80}\n")
        
        # Create config for this experiment
        config = base_config.copy()
        
        # Update with experiment parameters
        for key, value in exp_params.items():
            if key != 'description':
                config[key] = value
        
        # Create unique save path
        save_name = f"{exp_params['perturb_type']}_seed{config['random_seed']}"
        config['save_path'] = f"./experiments/sequential/{save_name}"
        
        # Update W&B settings
        config['wandb_run_name'] = f"seq{idx:02d}_{save_name}"
        config['wandb_tags'] = ['sequential', exp_params['perturb_type']]
        config['wandb_notes'] = exp_name
        
        # Check if experiment already completed
        save_path = Path(config['save_path'])
        if (save_path / 'training_res.csv').exists():
            response = input(f"Experiment {exp_name} already exists. Skip? (y/n): ")
            if response.lower() == 'y':
                logger.info(f"Skipped: {exp_name}")
                results['skipped'].append(exp_name)
                continue
        
        # Run experiment
        try:
            logger.info(f"Starting training for: {exp_name}")
            run_training_experiment(config)
            logger.info(f"✓ Successfully completed: {exp_name}")
            results['completed'].append(exp_name)
            
        except KeyboardInterrupt:
            logger.warning(f"User interrupted during: {exp_name}")
            logger.info("Stopping experiment queue")
            break
            
        except Exception as e:
            logger.error(f"✗ Failed: {exp_name}")
            logger.error(f"Error: {str(e)}")
            logger.error(traceback.format_exc())
            results['failed'].append((exp_name, str(e)))
            
            # Ask user whether to continue
            response = input(f"Continue with remaining experiments? (y/n): ")
            if response.lower() != 'y':
                logger.info("Stopping experiment queue")
                break
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("EXPERIMENT QUEUE SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Completed: {len(results['completed'])}/{len(experiments)}")
    logger.info(f"Failed: {len(results['failed'])}/{len(experiments)}")
    logger.info(f"Skipped: {len(results['skipped'])}/{len(experiments)}")
    
    if results['completed']:
        logger.info("\nCompleted experiments:")
        for name in results['completed']:
            logger.info(f"  ✓ {name}")
    
    if results['failed']:
        logger.info("\nFailed experiments:")
        for name, error in results['failed']:
            logger.info(f"  ✗ {name}: {error}")
    
    if results['skipped']:
        logger.info("\nSkipped experiments:")
        for name in results['skipped']:
            logger.info(f"  ○ {name}")
    
    logger.info(f"\n{'='*80}\n")

if __name__ == "__main__":
    run_sequential_experiments()