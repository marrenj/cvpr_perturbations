from functions.dora_inference_behavior_pipeline_720_concepts_optimized import run_behavior_inference

def main(): 
    # Define configuration
    config = {
        'img_dir': '../../../teba/multimodal_brain_inspired/THINGS_images',  # input images directory,
        'stimuli_file': '../../frrsa/sub-01_StimulusMetadata_train_only.csv',  # input csv file
        'concept_index_file': '../../frrsa/sub-01_lLOC_concept_index.npy',  # input npy file
        'load_hba': True,  # False will load the original CLIP-ViT weights
        'backbone': 'ViT-L/14',  # CLIP backbone model
        'model_path': './models/cliphba_behavior_20250919_212822.pth',  # path to the final trained model
        'dora_params_path': './dora_params_20250919_212822',  # path to the dora parameters
        'save_folder': './output/cliphba_behavior/20250919_212822',  # output path
        'batch_size': 64,  # batch size (increased for better GPU utilization)
        'cuda': 'cuda:1',  # 'cuda:0' for GPU 0, 'cuda:1' for GPU 1, '-1' for all GPUs
        'start_epoch': 28,  # start from epoch 7
        'max_batch_size': 256  # max batch size
    }

    # Run inference with configuration
    run_behavior_inference(config)

if __name__ == '__main__':
    main()