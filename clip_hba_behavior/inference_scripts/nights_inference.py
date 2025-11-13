from functions.nights_inference_pipeline import run_nights_inference

def main(): 
    # Define configuration
    config = {
        'nights_dir': '/home/wallacelab/teba/multimodal_brain_inspired/dreamsim/dreamsim/nights',  # Path to NIGHTS dataset directory
        'backbone': 'ViT-L/14',  # CLIP backbone model: 'RN50', 'ViT-B/32', or 'ViT-B/16'
        'load_hba': True,  # False will load the original CLIP weights
        'dora_params_path': '/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/baseline_runs/clip_hba_behavior_seed1/training_artifacts/dora_params/dora_params_seed1',  # Path to DoRA parameters directory (required if load_hba=True)
        'training_res_path': '/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/baseline_runs/clip_hba_behavior_seed1/training_results/training_res_20251013_220330.csv',  # Path to training results CSV for filtering epochs by minimum test loss
        'use_image_features': False,  # If True, use image features instead of 66D behavior predictions
        'splits': ['test'],  # Which splits to evaluate on: 'train', 'val', 'test'
        'batch_size': 256,  # Batch size for evaluation
        'device': 'cuda:0',  # 'cuda:0' for GPU 0, 'cuda:1' for GPU 1, 'cpu' for CPU
        'output_dir': './nights_results',  # Directory to save results
        'save_predictions': False,  # If True, save detailed predictions to CSV
    }

    # Run inference with configuration
    run_nights_inference(config)

if __name__ == '__main__':
    main()

