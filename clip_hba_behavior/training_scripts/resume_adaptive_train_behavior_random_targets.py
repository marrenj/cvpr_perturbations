import torch.nn as nn
from datetime import datetime

def main():
    # Configuration for resuming training
    checkpoint_path = './models/cliphba_behavior_20250922_222651.pth'
    training_results_path = './training_results/training_res_20250922_222651.csv'
    start_epoch = 26
    
    # Generate unique timestamp for this resumed training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define configuration
    config = {
        'csv_file': './Data/spose_embedding66d_rescaled_1806train.csv', # csv data annotations of the training stimuli with the corresponding target embeddings
        'img_dir': './Data/Things1854', # path to the image directory
        'inference_csv_file': './Data/spose_embedding66d_rescaled_48val_reordered.csv', # csv data annotations of the inference stimuli with the corresponding target embeddings
        'RDM48_triplet_dir': './Data/RDM48_triplet.mat', # location of the reference behavioral RDM
        'backbone': 'ViT-L/14', # CLIP backbone model, ViT-L/14 is the CLIP-HBA model default
        'epochs': 500, 
        'batch_size': 64,
        'train_portion': 0.8,
        'lr': 3e-4, # learning rate
        'early_stopping_patience': 20, # early stopping patience
        'checkpoint_path': f'./models/cliphba_behavior_resumed_{timestamp}.pth', # path to save the resumed model weights
        'training_res_path': f'./training_results/training_res_resumed_{timestamp}.csv', # location to save the resumed training results
        'dora_parameters_path': f'./dora_params_resumed_{timestamp}', # location to save the DoRA parameters
        'random_seed': 1, # random seed, default CLIP-HBA is 1
        'vision_layers': 2, # Last n ViT layers of the model to be trained, default CLIP-HBA-Behavior is 2
        'transformer_layers': 1, # Last n transformer layers to be trained, default CLIP-HBA-Behavior is 1
        'rank': 32, # Rank of the feature reweighting matrix, default CLIP-HBA-Behavior is 32
        'criterion': nn.MSELoss(), # MSE Loss
        'cuda': 0,  # -1 for all GPUs, 0 for GPU 0, 1 for GPU 1, 2 for CPU
        'use_curriculum_learning': True,
        'curriculum_algorithm': 'anti_baby_step',
        'start_rate': 0.5,
        'grow_rate': 0.1,
        'grow_interval': 5,
        'difficulty_criterion': nn.MSELoss(reduction='none'),
        'model_embedding_path': f'./model_embeddings_resumed_{timestamp}',
        'curriculum_subset_path': f'./curriculum_subsets_resumed_{timestamp}',
        'model_rdm_path': f'./model_rdms_resumed_{timestamp}',
        'random_target_epoch': 12,
        'random_target_seed': 42,
        # Resume training specific parameters
        'resume_checkpoint_path': checkpoint_path,
        'resume_training_results_path': training_results_path,
        'start_epoch': start_epoch
    }

    # Import the resume training function based on curriculum algorithm
    if config['curriculum_algorithm'] == 'baby_step':
        from functions.train_behavior_things_baby_step_pipeline_random_targets_resume import run_behavioral_traning_resume
    elif config['curriculum_algorithm'] == 'anti_baby_step':
        from functions.train_behavior_things_anti_baby_step_pipeline_random_targets_resume import run_behavioral_traning_resume

    # Run resumed training
    run_behavioral_traning_resume(config)

if __name__ == '__main__':
    main()
