import torch.nn as nn
from datetime import datetime
from functions.train_behavior_things_baseline_pipeline_marren_shuffle_targets import run_behavioral_traning

def main():
    # Generate unique timestamp for this training run
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
        'checkpoint_path': './models/cliphba_behavior_test.pth', # path to save the trained model weights
        'training_res_path': './training_results/training_res.csv', # location to save the training results
        'dora_parameters_path': './dora_params', # location to save the DoRA parameters
        'random_seed': 1, # random seed, default CLIP-HBA is 1
        'vision_layers': 2, # Last n ViT layers of the model to be trained, default CLIP-HBA-Behavior is 2
        'transformer_layers': 1, # Last n transformer layers to be trained, default CLIP-HBA-Behavior is 1
        'rank': 32, # Rank of the feature reweighting matrix, default CLIP-HBA-Behavior is 32
        'criterion': nn.MSELoss(), # MSE Loss
        'cuda': 1,  # -1 for all GPUs, 0 for GPU 0, 1 for GPU 1, 2 for CPU
        'difficulty_criterion': nn.MSELoss(reduction='none'),
        'model_embedding_path': './model_embeddings',
        'curriculum_subset_path': './curriculum_subsets',
        'model_rdm_path': './model_rdms',
        'random_target_epoch': 12,
        'random_target_seed': 42
    }

    # Update checkpoint path with timestamp
    config['checkpoint_path'] = f'./models/cliphba_behavior_{timestamp}.pth'

    # Update training results path with timestamp
    config['training_res_path'] = f'./training_results/training_res_{timestamp}.csv'

    # Update DoRA parameters path with timestamp
    config['dora_parameters_path'] = f'./dora_params_{timestamp}'

    # Update model embedding path with timestamp
    config['model_embedding_path'] = f'./model_embeddings_{timestamp}'

    # Update curriculum subset path with timestamp
    config['curriculum_subset_path'] = f'./curriculum_subsets_{timestamp}'

    # Update model RDM path with timestamp
    config['model_rdm_path'] = f'./model_rdms_{timestamp}'

    # Run training
    run_behavioral_traning(config)

if __name__ == '__main__':
    main()
