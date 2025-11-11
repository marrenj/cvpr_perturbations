from functions.cvpr_train_behavior_things_pipeline import run_behavioral_training
import torch.nn as nn
from datetime import datetime
import os
import logging
import sys
import argparse


def setup_main_logger(log_file_path):
    logger = logging.getLogger('main_training_loop')
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                  datefmt='%Y-%m-%d %H:%M:%S')
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def parse_args():
    parser = argparse.ArgumentParser(description='CVPR Behavior Training - SLURM Integration')
    parser.add_argument('--model', type=str, default='clip_hba')
    parser.add_argument('--perturb_type', type=str, default='random_target',
                        choices=['random_target', 'label_shuffle', 'baseline'])
    parser.add_argument('--perturb_epoch', type=int, required=True)
    parser.add_argument('--perturb_length', type=int, required=True)
    parser.add_argument('--perturb_distribution', type=str, default='target',
                        choices=['normal', 'target'])
    parser.add_argument('--perturb_seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--early_stopping_patience', type=int, default=20)
    parser.add_argument('--random_seed', type=int, default=1)
    parser.add_argument('--baseline_dora_directory', type=str, required=True)
    parser.add_argument('--baseline_random_state_path', type=str, required=True)
    parser.add_argument('--baseline_split_indices_path', type=str, required=True)
    parser.add_argument('--output_base_directory', type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ✅ Updated data paths for your directory layout
    config = {
        'csv_file': '/data/p_dsi/dhungs1/Data/spose_embedding66d_rescaled_1806train.csv',
        'img_dir': '/data/p_dsi/dhungs1/Data/Things1854',
        'inference_csv_file': '/data/p_dsi/dhungs1/Data/spose_embedding66d_rescaled_48val_reordered.csv',
        'RDM48_triplet_dir': '/data/p_dsi/dhungs1/Data/RDM48_triplet.mat',
        'backbone': 'ViT-L/14',
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'train_portion': 0.8,
        'lr': args.lr,
        'logger': None,
        'early_stopping_patience': args.early_stopping_patience,
        'random_seed': args.random_seed,
        'vision_layers': 2,
        'transformer_layers': 1,
        'rank': 32,
        'criterion': nn.MSELoss(),
        'cuda': args.cuda,
        'baseline_dora_directory': args.baseline_dora_directory,
        'baseline_random_state_path': args.baseline_random_state_path,
        'baseline_split_indices_path': args.baseline_split_indices_path,
        'perturb_type': args.perturb_type,
        'perturb_distribution': args.perturb_distribution,
        'perturb_seed': args.perturb_seed,
        'training_run': args.perturb_epoch,
        'resume_from_epoch': max(0, args.perturb_epoch - 1),
        'output_base_directory': args.output_base_directory,
        'output_directory': args.output_dir,
    }

    config['output_dir'] = os.path.join(config['output_base_directory'], config['output_directory'])

    existing_csv = os.path.join(config['output_dir'], 'training_res.csv')
    existing_ckpt = os.path.join(config['output_dir'], f"model_checkpoint_{args.perturb_epoch}.pth")
    existing_dora_dir = os.path.join(config['output_dir'], f"dora_params_{args.perturb_epoch}")
    existing_rs_dir = os.path.join(config['output_dir'], f"random_states_{args.perturb_epoch}")

    if os.path.isdir(config['output_dir']) or os.path.exists(existing_csv) or os.path.exists(existing_ckpt):
        print(f"Detected existing artifacts; skipping: {config['output_dir']}")
        sys.exit(2)

    os.makedirs(config['output_dir'], exist_ok=False)

    config['checkpoint_path'] = os.path.join(config['output_dir'], f'model_checkpoint_{args.perturb_epoch}.pth')
    config['training_res_path'] = os.path.join(config['output_dir'], 'training_res.csv')
    config['dora_parameters_path'] = os.path.join(config['output_dir'], f'dora_params_{args.perturb_epoch}')
    config['random_state_path'] = os.path.join(config['output_dir'], f'random_states_{args.perturb_epoch}')

    log_file = os.path.join(config['output_dir'], f'training_log_{timestamp}.txt')
    logger = setup_main_logger(log_file)

    logger.info("="*80)
    logger.info("STARTING TRAINING RUN")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Perturbation: {args.perturb_type} | Epoch: {args.perturb_epoch} | Length: {args.perturb_length}")
    logger.info(f"Distribution: {args.perturb_distribution} | Seed: {args.perturb_seed}")
    logger.info(f"Output Dir: {args.output_dir}")
    logger.info(f"CUDA: {args.cuda} | Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    logger.info("="*80)

    try:
        run_behavioral_training(config)
        logger.info("="*80)
        logger.info("✓ TRAINING COMPLETED SUCCESSFULLY")
        logger.info(f"Results saved to {config['output_dir']}")
        logger.info("="*80)
    except Exception as e:
        logger.error("✗ TRAINING RUN FAILED")
        logger.error(str(e))
        raise


if __name__ == '__main__':
    main()
