from functions.inference_behavior_pipeline import run_behavior_inference

def main(): 
    # Define configuration
    config = {
        'img_dir': './Data/Things1854',  # input images directory
        'load_hba': True,  # False will load the original CLIP-ViT weights
        'backbone': 'ViT-L/14',  # CLIP backbone model
        'model_path': './dora_params_20250828_134426/epoch4_dora_params.pth',  # path to the trained model
        'save_folder': './output/cliphba_behavior/20250828_134426/epoch4_output',  # output path
        'batch_size': 32,  # batch size
        'cuda': 'cuda:1'  # 'cuda:0' for GPU 0, 'cuda:1' for GPU 1, '-1' for all GPUs
    }

    # Run inference with configuration
    run_behavior_inference(config)

if __name__ == '__main__':
    main()