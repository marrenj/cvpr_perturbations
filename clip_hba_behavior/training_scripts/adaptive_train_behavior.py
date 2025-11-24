import torch.nn as nn
from datetime import datetime

from functions.train_behavior_things_baby_step_pipeline import run_behavioral_traning as run_behavioral_traning_baby_step
from functions.train_behavior_things_anti_baby_step_pipeline_revised import run_behavioral_traning as run_behavioral_traning_anti_baby_step
from functions.train_behavior_things_baseline_pipeline_marren import run_behavioral_traning as run_behavioral_traning_baseline

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
        'checkpoint_path': '/home/wallacelab/teba/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior/models/cliphba_behavior_{timestamp}.pth', # path to save the trained model weights
        'training_res_path': '/home/wallacelab/teba/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior/training_results/training_res_{timestamp}.csv', # location to save the training results
        'dora_parameters_path': '/home/wallacelab/teba/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior/training_artifacts/dora_params_{timestamp}', # location to save the DoRA parameters
        'random_seed': 1, # random seed, default CLIP-HBA is 1
        'vision_layers': 2, # Last n ViT layers of the model to be trained, default CLIP-HBA-Behavior is 2
        'transformer_layers': 1, # Last n transformer layers to be trained, default CLIP-HBA-Behavior is 1
        'rank': 32, # Rank of the feature reweighting matrix, default CLIP-HBA-Behavior is 32
        'criterion': nn.MSELoss(), # MSE Loss
        'cuda': 1,  # -1 for all GPUs, 0 for GPU 0, 1 for GPU 1, 2 for CPU
        'use_curriculum_learning': False,
        #'curriculum_algorithm': '',
        #'start_rate': 0.7,
        #'grow_rate': 0.02,
        #'grow_interval': 5,
        'difficulty_criterion': nn.MSELoss(reduction='none'),
        'model_embedding_path': '/home/wallacelab/teba/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior/training_artifacts/model_embeddings_{timestamp}',
        'curriculum_subset_path': '/home/wallacelab/teba/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior/training_artifacts/curriculum_subsets_{timestamp}',
        'model_rdm_path': '/home/wallacelab/teba/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior/training_artifacts/model_rdms_{timestamp}'
    }

    if config['use_curriculum_learning']:
        if config['curriculum_algorithm'] == 'baby_step':
            run_behavioral_traning_baby_step(config)
        elif config['curriculum_algorithm'] == 'anti_baby_step':
            run_behavioral_traning_anti_baby_step(config)
    else:
        run_behavioral_traning_baseline(config)

    # Run training
    

if __name__ == '__main__':
    main()