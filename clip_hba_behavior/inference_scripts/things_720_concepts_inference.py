from functions.things_720_concepts_inference_pipeline import run_behavior_inference

def main(): 
    # Define configuration
    config = {
        'img_dir': '/home/wallacelab/teba/multimodal_brain_inspired/THINGS_images',
        'stimuli_file': '/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/things_720concepts_stimulus_metadata.csv',
        'batch_size': 64,
        'cuda': 'cuda:0',
        'load_hba': True,
        'backbone': 'ViT-L/14',
        'dora_params_path': '/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/baseline_runs/clip_hba_behavior_seed3/training_artifacts/dora_params/dora_params_seed3',  # path to the dora parameters
        'training_res_path': '/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/baseline_runs/clip_hba_behavior_seed3/training_results/training_res_seed3.csv', # path to the baseline training results
        'save_folder': '/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/baseline_runs/clip_hba_behavior_seed3/inference_results/things_720_concepts_inference_results'
    }

    # Run inference with configuration
    run_behavior_inference(config)

if __name__ == '__main__':
    main()