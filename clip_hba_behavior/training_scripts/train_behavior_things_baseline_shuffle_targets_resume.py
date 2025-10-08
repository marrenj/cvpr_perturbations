import torch.nn as nn
from functions.train_behavior_things_baseline_pipeline_marren_shuffle_targets_resume import run_behavioral_traning

def main():
    # Use the existing timestamp from the previous training run
    timestamp = "20250919_212822"

    # Define configuration for resuming training
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
        'checkpoint_path': f'/home/wallacelab/marren/adaptive-clip/CLIP-HBA/models/cliphba_behavior_{timestamp}.pth', # path to load the existing checkpoint
        'training_res_path': f'/home/wallacelab/marren/adaptive-clip/CLIP-HBA/training_results/training_res_{timestamp}.csv', # path to append to existing training results
        'dora_parameters_path': f'/home/wallacelab/marren/adaptive-clip/CLIP-HBA/dora_params_{timestamp}', # path to existing DoRA parameters folder
        'random_seed': 1, # random seed, default CLIP-HBA is 1
        'vision_layers': 2, # Last n ViT layers of the model to be trained, default CLIP-HBA-Behavior is 2
        'transformer_layers': 1, # Last n transformer layers to be trained, default CLIP-HBA-Behavior is 1
        'rank': 32, # Rank of the feature reweighting matrix, default CLIP-HBA-Behavior is 32
        'criterion': nn.MSELoss(), # MSE Loss
        'cuda': 0,  # -1 for all GPUs, 0 for GPU 0, 1 for GPU 1, 2 for CPU
        'difficulty_criterion': nn.MSELoss(reduction='none'),
        'model_embedding_path': f'/home/wallacelab/teba/marren/model_embeddings_{timestamp}',
        'curriculum_subset_path': f'/home/wallacelab/teba/marren/curriculum_subsets_{timestamp}',
        'model_rdm_path': f'/home/wallacelab/teba/marren/model_rdms_{timestamp}',
        'random_target_epoch': 12,
        'random_target_seed': 42,
        'resume_from_epoch': 119,  # Start training from epoch 107
        'resume_checkpoint_path': f'/home/wallacelab/marren/adaptive-clip/CLIP-HBA/models/cliphba_behavior_{timestamp}.pth'  # Path to load checkpoint from
    }

    # Run training
    run_behavioral_traning(config)

if __name__ == '__main__':
    main()
