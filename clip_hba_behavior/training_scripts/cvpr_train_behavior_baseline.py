from functions.cvpr_train_behavior_things_pipeline_baseline import run_behavioral_training
import torch.nn as nn
from datetime import datetime

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
        'checkpoint_path': f'/home/wallacelab/teba/multimodal_brain_inspired/marren/cliphba_behavior_{timestamp}.pth', # path to save the trained model weights
        'training_res_path': f'/home/wallacelab/teba/multimodal_brain_inspired/marren/training_res_{timestamp}.csv', # location to save the training results
        'dora_parameters_path': f'/home/wallacelab/teba/multimodal_brain_inspired/marren/dora_params_{timestamp}', # location to save the DoRA parameters
        'random_state_path': f'/home/wallacelab/teba/multimodal_brain_inspired/marren/random_states_{timestamp}', # location to save the random states at every epoch
        'random_seed': 1, # random seed, default CLIP-HBA is 1
        'vision_layers': 2, # Last n ViT layers of the model to be trained, default CLIP-HBA-Behavior is 2
        'transformer_layers': 1, # Last n transformer layers to be trained, default CLIP-HBA-Behavior is 1
        'rank': 32, # Rank of the feature reweighting matrix, default CLIP-HBA-Behavior is 32
        'criterion': nn.MSELoss(), # MSE Loss
        'cuda': 0  # -1 for all GPUs, 0 for GPU 0, 1 for GPU 1, 2 for CPU
    }
    
    # Run training
    run_behavioral_training(config)

if __name__ == '__main__':
    main()