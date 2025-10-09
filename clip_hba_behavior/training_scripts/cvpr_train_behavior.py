from functions.cvpr_train_behavior_things_pipeline import run_behavioral_traning
import torch.nn as nn
from datetime import datetime
import os

def main():

    # Generate unique timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define base configuration shared across all runs
    base_config = {
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
        'perturb_seed': 42,
        'perturb_type': 'random_target', # either 'random_target' or 'label_shuffle',
        'perturb_distribution': 'global' # if using random_target, either 'normal' (for generating random targets from a normal distribution) or 'global' (for generating random targets from the target global statistics)
    }

    # Baseline checkpoint directory
    baseline_dora_dir = '/home/wallacelab/teba/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior/training_artifacts/dora_params/dora_params_20251008_211424'
    
    # Base directory for saving results
    base_dir = '/home/wallacelab/teba/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior_loops'
    
    # Create a parent timestamp for this batch of runs
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get list of available baseline checkpoints
    checkpoint_files = sorted([f for f in os.listdir(baseline_dora_dir) if f.endswith('_dora_params.pth')])
    available_epochs = [int(f.split('_')[0].replace('epoch', '')) for f in checkpoint_files]
    
    print(f"Found {len(available_epochs)} baseline checkpoints")
    print(f"Available epochs: {available_epochs[:10]}... {available_epochs[-1:]}")
    print()

    # Loop through epochs where we want to introduce perturbations
    # For epoch 1: start from scratch (no checkpoint)
    # For epoch N (N>1): load checkpoint from epoch N-1
    for perturb_epoch in range(1, 94):  # 1 to 93 (epoch numbers are 1-indexed)
        print("\n" + "="*80)
        print(f"STARTING TRAINING RUN {perturb_epoch}/93")
        print(f"Perturbation will be introduced at EPOCH {perturb_epoch}")
        
        # Determine if we should load from checkpoint
        if perturb_epoch == 1:
            print("Starting from scratch (epoch 1 perturbation)")
            load_checkpoint = None
        else:
            # Load from previous epoch
            checkpoint_epoch = perturb_epoch - 1
            if checkpoint_epoch in available_epochs:
                load_checkpoint = os.path.join(baseline_dora_dir, f'epoch{checkpoint_epoch}_dora_params.pth')
                print(f"Loading checkpoint from epoch {checkpoint_epoch}")
            else:
                print(f"WARNING: No checkpoint found for epoch {checkpoint_epoch}")
                print("Skipping this perturbation epoch...")
                continue  # Skip this iteration
        
        print("="*80 + "\n")

        print("="*80 + "\n")
    
        # Create a unique identifier for this run
        run_id = f"epoch{perturb_epoch}_{base_config['perturb_type']}_{batch_timestamp}"
    
        # Create config for this specific run
        config = base_config.copy()
    
        # Set unique paths for this run
        config['checkpoint_path'] = os.path.join(
            base_dir, 'models', f'cliphba_behavior_{run_id}.pth'
        )
        config['training_res_path'] = os.path.join(
            base_dir, 'training_results', f'training_res_{run_id}.csv'
        )
        config['dora_parameters_path'] = os.path.join(
            base_dir, 'training_artifacts', 'dora_params', f'dora_params_{run_id}'
        )

        # Set the epoch at which to introduce random targets
        config['perturb_epoch'] = perturb_epoch

        # Add checkpoint loading configuration
        config['load_checkpoint'] = load_checkpoint
        config['resume_from_epoch'] = perturb_epoch - 1 if load_checkpoint else 0

        # Create directories if they don't exist
        os.makedirs(os.path.dirname(config['checkpoint_path']), exist_ok=True)
        os.makedirs(os.path.dirname(config['training_res_path']), exist_ok=True)
        os.makedirs(config['dora_parameters_path'], exist_ok=True)

        # Run training
        try:
            run_behavioral_traning(config)
            print(f"\n✓ Successfully completed training with perturbation at epoch {perturb_epoch}\n")
        except Exception as e:
            print(f"\n✗ Error in training with perturbation at epoch {perturb_epoch}: {str(e)}\n")
            import traceback
            traceback.print_exc()
            # Continue to next iteration rather than stopping
            continue

    print("\n" + "="*80)
    print("ALL 93 TRAINING RUNS COMPLETED!")
    print(f"Batch ID: {batch_timestamp}")
    print("="*80)

if __name__ == '__main__':
    main()