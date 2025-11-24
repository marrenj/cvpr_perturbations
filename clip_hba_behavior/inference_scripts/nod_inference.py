from functions.nod_inference_pipeline import run_behavior_inference

def main(): 
    # Define configuration
    config = {
        'img_dir': '/home/wallacelab/teba/multimodal_brain_inspired/NOD/imagenet',  # input images directory,
        #'stimuli_file': '',  # input csv file
        'category_index_file': '../analysis/nod_2k_images.csv', 
        'load_hba': True,  # False will load the original CLIP-ViT weights
        'backbone': 'ViT-L/14',  # CLIP backbone model
        #'model_path': '/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior/models/cliphba_behavior_20250919_212822.pth',  # path to the final trained model
        'dora_params_path': '/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/baseline_runs/clip_hba_behavior_seed3/training_artifacts/dora_params/dora_params_seed3',  # path to the dora parameters
        'training_res_path': '/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/baseline_runs/clip_hba_behavior_seed3/training_results/training_res_20251105_171137.csv', # path to the baseline training results
        'save_folder': '/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/baseline_runs/clip_hba_behavior_seed3/inference_results/nod_inference_results',  # output path
        'batch_size': 256,  # batch size (increased for better GPU utilization)
        'cuda': 'cuda:0',  # 'cuda:0' for GPU 0, 'cuda:1' for GPU 1, '-1' for all GPUs
    }

    # Run inference with configuration
    run_behavior_inference(config)

if __name__ == '__main__':
    main()