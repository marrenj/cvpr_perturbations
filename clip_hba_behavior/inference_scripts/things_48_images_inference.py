from functions.things_48_images_inference_pipeline import run_behavior_inference

def main(): 
    # Define configuration
    config = {
        'img_dir': '../Data/Things1854',  # input images directory,
        'inference_csv_file': '../Data/spose_embedding66d_rescaled_48val_reordered.csv', # csv data annotations of the inference stimuli with the corresponding target embeddings
        'RDM48_triplet_dir': '../Data/RDM48_triplet.mat', # location of the reference behavioral RDM
        'load_hba': True,  # False will load the original CLIP-ViT weights
        'backbone': 'ViT-L/14',  # CLIP backbone model
        'dora_params_path': '/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/baseline_runs/clip_hba_behavior_seed1/training_artifacts/dora_params/dora_params_seed1',  # path to the dora parameters
        'training_res_path': '/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/baseline_runs/clip_hba_behavior_seed1/training_results/training_res_seed1.csv', # path to the baseline training results
        'save_folder': '/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/baseline_runs/clip_hba_behavior_seed1/inference_results/things_48_inference_results',  # output path
        'batch_size': 48,  # batch size (increased for better GPU utilization)
        'cuda': 'cuda:0',  # 'cuda:0' for GPU 0, 'cuda:1' for GPU 1, '-1' for all GPUs
    }

    # Run inference with configuration
    run_behavior_inference(config)

if __name__ == '__main__':
    main()