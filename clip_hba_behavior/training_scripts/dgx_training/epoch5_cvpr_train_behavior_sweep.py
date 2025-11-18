from functions.new_cvpr_train_behavior_things_pipeline import run_behavioral_training
import torch.nn as nn
from datetime import datetime
import os
import logging
import sys

def setup_main_logger(log_file_path):
    logger = logging.getLogger('main_training_loop')
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    return logger


def main():

    # Unique timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Configuration
    config = {
        'csv_file': '/data/p_dsi/dhungs1/Data/spose_embedding66d_rescaled_1806train.csv',
        'img_dir': '/data/p_dsi/dhungs1/Data/Things1854',
        'inference_csv_file': '/data/p_dsi/dhungs1/Data/spose_embedding66d_rescaled_48val_reordered.csv',
        'RDM48_triplet_dir': '/data/p_dsi/dhungs1/Data/RDM48_triplet.mat',
        'backbone': 'ViT-L/14',
        'epochs': 500,                     # <-- full normal training
        'batch_size': 64,
        'train_portion': 0.8,
        'lr': 3e-4,
        'logger': None,
        'early_stopping_patience': 20,
        'random_seed': 2,
        'vision_layers': 2,
        'transformer_layers': 1,
        'rank': 32,
        'criterion': nn.MSELoss(),
        'cuda': 0,
        'baseline_dora_directory':
            '/data/p_dsi/dhungs1/baseline_runs/clip_hba_behavior_seed1/training_artifacts/dora_params/dora_params_seed1',
        'baseline_random_state_path':
            '/data/p_dsi/dhungs1/baseline_runs/clip_hba_behavior_seed1/training_artifacts/random_states/random_states_seed1',
        'baseline_split_indices_path':
            '/data/p_dsi/dhungs1/baseline_runs/clip_hba_behavior_seed1/training_artifacts/random_states/random_states_seed1/dataset_split_indices.pth',

        # --- Perturbation specifics ---
        'perturb_type': 'image_noise',     # only epoch 5
        'perturb_length': 1,
        'perturb_distribution': 'target',
        'perturb_seed': 42,

        # Output directory (everything goes inside training_run5)
        'output_base_directory':
            f'/data/p_dsi/dhungs1/image_noise_single_epoch_perturbation_sweeps/'
            f'perturb_sweep_baselineseed1_perturbseed42_{timestamp}',
    }

    # Logger
    main_log_path = os.path.join(
        config['output_base_directory'],
        f'main_training_log_{timestamp}.txt'
    )
    logger = setup_main_logger(main_log_path)

    logger.info("="*80)
    logger.info("STARTING SINGLE PERTURBATION RUN: EPOCH 5 ONLY")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Perturbation type: image_noise at epoch 5 only")
    logger.info("="*80)
    logger.info("")

    # ---------------------------------------------------------------------
    # NEW BEHAVIOR:
    #   One single training session:
    #     → resume from epoch 4
    #     → epoch 5 gets perturbation
    #     → continue normally to epoch 500
    # ---------------------------------------------------------------------

    training_run = 5
    logger.info(f"Preparing training_run5 directory")

    training_run_directory = os.path.join(
        config['output_base_directory'],
        f'training_run{training_run}'
    )
    os.makedirs(training_run_directory, exist_ok=True)

    config['training_run'] = training_run
    config['checkpoint_path'] = os.path.join(
        training_run_directory,
        f'model_checkpoint_run{training_run}.pth'
    )
    config['training_res_path'] = os.path.join(
        training_run_directory,
        f'training_res_run{training_run}.csv'
    )
    config['dora_parameters_path'] = os.path.join(
        training_run_directory,
        f'dora_params_run{training_run}'
    )
    config['random_state_path'] = os.path.join(
        training_run_directory,
        f'random_states_run{training_run}'
    )

    # ------------------------------------------------------------------
    # Key setting: resume from epoch *4*
    # ------------------------------------------------------------------
    config['resume_from_epoch'] = 4

    logger.info(f"Resume from epoch: 4")
    logger.info(f"Total epochs: {config['epochs']}")
    logger.info(f"Epoch 5 will contain perturbation, epochs 6–500 normal.")
    logger.info("Starting training now...")
    logger.info("")

    try:
        run_behavioral_training(config)
        logger.info("✓ FULL TRAINING (EPOCH 5 → 500) COMPLETED SUCCESSFULLY")
    except Exception as e:
        logger.error("✗ TRAINING FAILED")
        logger.error(str(e))
        raise

    logger.info("="*80)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Artifacts saved at: {training_run_directory}")
    logger.info("="*80)


if __name__ == '__main__':
    main()
