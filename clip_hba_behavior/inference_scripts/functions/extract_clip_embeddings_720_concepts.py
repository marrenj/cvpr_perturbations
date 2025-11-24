#!/usr/bin/env python3
"""
Extract embeddings for the 8640 THINGS images from CLIP checkpoints.
Cleaned and concise version.
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import re
import pickle
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
from tqdm import tqdm

# Import your CLIP components
pretrain_path = Path("/home/wallacelab/teba/Pretrain")
src_path = pretrain_path / "src"
datacomp_lib_path = pretrain_path / "lib" / "datacomp"

sys.path.insert(0, str(src_path))
sys.path.insert(0, str(datacomp_lib_path))
sys.path.insert(0, str(pretrain_path))

from model import CLIP  # type: ignore
from config import VIT_B_32_CONFIG  # type: ignore


def load_clip_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> Optional[torch.nn.Module]:
    """Load CLIP model from checkpoint with error handling."""
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        if not isinstance(state_dict, dict):
            print("Error: Invalid state_dict")
            return None

        # Initialize model
        model = CLIP(config=VIT_B_32_CONFIG, random_text_encoder=True)

        # Clean state dict
        cleaned_state_dict = OrderedDict()
        prefix = 'model.'
        has_prefix = any(k.startswith(prefix) for k in state_dict.keys())
        
        if has_prefix:
            for k, v in state_dict.items():
                if k.startswith(prefix):
                    cleaned_state_dict[k[len(prefix):]] = v
                else:
                    cleaned_state_dict[k] = v
        else:
            cleaned_state_dict = state_dict

        # Load weights
        missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing {len(missing_keys)} keys")
        if unexpected_keys:
            print(f"Warning: Unexpected {len(unexpected_keys)} keys")

        model = model.to(device)
        model.eval()
        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def register_vision_hooks(model: torch.nn.Module) -> Tuple[List[str], Dict[str, torch.Tensor]]:
    """Register forward hooks for ALL CLIP vision encoder layers."""
    activations = {}
    layer_names = []

    def get_hook(name: str):
        def hook(_, __, out):
            activations[name] = out.detach()
        return hook

    def add_hook(module: torch.nn.Module, name: str):
        layer_names.append(name)
        module.register_forward_hook(get_hook(name))

    if not hasattr(model, 'visual'):
        print("Error: Model does not have 'visual' attribute")
        return [], {}
    
    vision_model = model.visual
    
    try:
        # Register patch embedding
        if hasattr(vision_model, 'conv1'):
            add_hook(vision_model.conv1, 'vision.conv1')
            print("Registered patch embedding: conv1")
        
        # Register ALL transformer layers
        if hasattr(vision_model, 'transformer') and isinstance(vision_model.transformer, torch.nn.ModuleList):
            transformer_layers = vision_model.transformer
            num_layers = len(transformer_layers)
            print(f"Found {num_layers} transformer layers - registering ALL layers")
            
            # Register ALL transformer layers (0 to num_layers-1)
            for idx in range(num_layers):
                add_hook(transformer_layers[idx], f'vision.transformer.layer_{idx}')
                print(f"Registered transformer layer {idx}")
        
        # Register post-layer norm
        if hasattr(vision_model, 'ln_post'):
            add_hook(vision_model.ln_post, 'vision.ln_post')
            print("Registered post-norm: ln_post")
            
    except Exception as e:
        print(f"Error during hook registration: {e}")
        return [], {}
    
    print(f"Successfully registered {len(layer_names)} vision encoder hooks for ALL layers")
    return layer_names, activations


class THINGSInferenceDataset(Dataset):
    """Dataset for the 8640 THINGS images."""
    
    def __init__(self, stimuli_file, concept_index_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        # Load the full CSV
        self.stimuli_metadata = pd.read_csv(stimuli_file, index_col=0)
        self.concept_index = np.load(concept_index_file)

        print("🔍 SANITY CHECK: Dataset metadata loaded:")
        print(f"✅ CSV shape: {self.stimuli_metadata.shape}")
        print(f"✅ CSV columns: {list(self.stimuli_metadata.columns)}")
        print(f"✅ Concept index shape: {self.concept_index.shape}")
        print(f"✅ First few rows:")
        print(self.stimuli_metadata.head())
    
    def __len__(self):
        return len(self.stimuli_metadata)
        
    def __getitem__(self, index):
        # get the row at the specified index
        row = self.stimuli_metadata.iloc[index]

        # extract the concept and image name from the row
        concept = row['concept']
        image_name = row['stimulus']

        # Build the image path
        image_path = os.path.join(self.img_dir, concept, image_name)

        # Load and transform the image
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image, image_name, concept
            
        #return image, Path(img_name).stem


def feature_to_matrix(feat: torch.Tensor) -> torch.Tensor:
    """Convert feature tensor to [batch_size, feature_dim] matrix."""
    if feat.dim() == 4:  # Conv layers: [B, C, H, W]
        return F.adaptive_avg_pool2d(feat, (1, 1)).reshape(feat.size(0), -1)
    elif feat.dim() == 3:  # Transformer layers: [B, seq_len, hidden_dim]
        return feat[:, 0, :]  # Take CLS token
    elif feat.dim() == 2:  # Already [batch_size, features]
        return feat
    else:
        return feat.reshape(feat.size(0), -1)


def extract_embeddings_from_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    layer_names: List[str],
    activations: Dict[str, torch.Tensor],
    device: torch.device
) -> Dict[str, np.ndarray]:
    """Extract embeddings from all hooked layers."""
    
    layer_embeddings = {layer_name: [] for layer_name in layer_names}
    image_names = []
    concepts = []
    
    with torch.no_grad():
        for batch_images, batch_image_names, batch_concepts in data_loader:
            batch_images = batch_images.to(device)
            
            # Forward pass through vision encoder
            try:
                _ = model.encode_image(batch_images)
            except Exception as e:
                print(f"Error in model.encode_image(): {e}")
                continue
            
            # Collect activations from hooks
            for layer_name in layer_names:
                if layer_name in activations:
                    feat = activations[layer_name]
                    feat_matrix = feature_to_matrix(feat)
                    layer_embeddings[layer_name].append(feat_matrix.cpu().numpy())
            
            image_names.extend(batch_image_names)
            concepts.extend(batch_concepts)
    
    # Concatenate all batches
    final_embeddings = {}
    for layer_name in layer_names:
        if layer_embeddings[layer_name]:
            final_embeddings[layer_name] = np.vstack(layer_embeddings[layer_name])
            print(f"Layer {layer_name}: {final_embeddings[layer_name].shape}")
        else:
            print(f"Warning: No embeddings collected for layer {layer_name}")
    
    final_embeddings['image_names'] = np.array(image_names)
    final_embeddings['concepts'] = np.array(concepts)
    return final_embeddings


def run_clip_inference(model: torch.nn.Module, data_loader: DataLoader, device: torch.device) -> Dict[str, pd.DataFrame]:
    """Run inference using CLIP model and return embeddings from ALL layers as a dictionary."""
    print(f"🔍 SANITY CHECK: Setting up model for inference...")
    model.eval()
    model.to(device)
    print(f"✅ Model set to eval mode and moved to {device}")
    
    # SANITY CHECK: Register hooks for ALL layers
    print(f"🔍 SANITY CHECK: Registering hooks for ALL layers...")
    layer_names, activations = register_vision_hooks(model)
    print(f"✅ Available layers: {layer_names}")
    
    if not layer_names:
        raise ValueError("No layers available for embedding extraction")
    
    # Initialize storage for all layers
    layer_embeddings = {layer_name: [] for layer_name in layer_names}
    image_names = []
    concepts = []
    
    print(f"🔍 SANITY CHECK: Starting inference with {len(data_loader)} batches...")
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc="Processing images")
    
    with torch.no_grad():
        for _, (batch_images, batch_image_names, batch_concepts) in progress_bar:
            batch_images = batch_images.to(device)
            
            # Forward pass through vision encoder
            try:
                _ = model.encode_image(batch_images)
            except Exception as e:
                print(f"Error in model.encode_image(): {e}")
                continue
            
            # Extract embeddings from ALL layers
            for layer_name in layer_names:
                if layer_name in activations:
                    feat = activations[layer_name]
                    feat_matrix = feature_to_matrix(feat)
                    layer_embeddings[layer_name].append(feat_matrix.cpu().numpy())
            
            # Store image names and concepts for each batch
            image_names.extend(batch_image_names)
            concepts.extend(batch_concepts)
    
    # SANITY CHECK: Concatenate all embeddings for each layer
    print(f"🔍 SANITY CHECK: Processing extracted embeddings for {len(layer_names)} layers...")
    layer_dataframes = {}
    
    for layer_name in layer_names:
        if layer_embeddings[layer_name]:
            embeddings_array = np.vstack(layer_embeddings[layer_name])
            print(f"✅ Layer {layer_name}: embeddings shape {embeddings_array.shape}")
            
            # Create DataFrame for this layer
            layer_df = pd.DataFrame(embeddings_array)
            layer_df['image'] = image_names
            layer_df['concept'] = concepts
            layer_df = layer_df[['image', 'concept'] + [col for col in layer_df if col != 'image' and col != 'concept']]
            
            layer_dataframes[layer_name] = layer_df
            print(f"✅ Layer {layer_name}: DataFrame shape {layer_df.shape}")
        else:
            print(f"⚠️  Warning: No embeddings collected for layer {layer_name}")
    
    print(f"✅ Total images processed: {len(image_names)}")
    print(f"✅ Total concepts: {len(set(concepts))}")
    print(f"✅ Successfully extracted embeddings from {len(layer_dataframes)} layers")
    
    return layer_dataframes


def extract_training_metrics(checkpoint: dict) -> dict:
    """Extract training metrics from PyTorch Lightning checkpoint."""
    metrics = {}
    
    # Extract basic training info
    for key in ['epoch', 'global_step']:
        if key in checkpoint:
            metrics[key] = checkpoint[key]
    
    # Extract validation metrics from ModelCheckpoint callback
    if 'callbacks' in checkpoint:
        for callback_key, callback_data in checkpoint['callbacks'].items():
            if 'ModelCheckpoint' in callback_key and isinstance(callback_data, dict):
                if 'monitor' in callback_data:
                    metrics['monitored_metric'] = callback_data['monitor']
                if 'best_model_score' in callback_data:
                    score = callback_data['best_model_score']
                    if hasattr(score, 'item'):
                        metrics['best_score'] = float(score.item())
                    else:
                        metrics['best_score'] = float(score)
                if 'current_score' in callback_data:
                    score = callback_data['current_score']
                    if hasattr(score, 'item'):
                        metrics['current_score'] = float(score.item())
                    else:
                        metrics['current_score'] = float(score)
    
    # Extract hyperparameters
    if 'hyper_parameters' in checkpoint:
        hparams = checkpoint['hyper_parameters']
        if 'training_config' in hparams:
            config = hparams['training_config']
            if hasattr(config, 'optimizer') and hasattr(config.optimizer, 'lr'):
                metrics['learning_rate'] = float(config.optimizer.lr)
            if hasattr(config, 'data') and hasattr(config.data, 'batch_size'):
                metrics['batch_size'] = int(config.data.batch_size)
            if hasattr(config, 'epochs'):
                metrics['total_epochs'] = int(config.epochs)
    
    return metrics


def get_checkpoint_files(ckpt_root: Path) -> List[Path]:
    """Get and sort checkpoint files by epoch/step."""
    ckpt_files = list(ckpt_root.glob("*.ckpt"))
    
    if not ckpt_files:
        print(f"Error: No checkpoint files found in {ckpt_root}")
        return []
    
    def extract_epoch(path: Path) -> int:
        """Extract epoch/step number from checkpoint filename."""
        for pattern in [r'epoch[=_]?(\d+)', r'step[=_]?(\d+)', r'(\d+)']:
            match = re.search(pattern, path.name)
            if match:
                return int(match.group(1))
        return 0
    
    ckpt_files = sorted(ckpt_files, key=extract_epoch)
    print(f"Found {len(ckpt_files)} checkpoints")
    return ckpt_files


def cleanup_gpu_memory():
    """Clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    """Main function to extract CLIP embeddings for THINGS inference images."""
    
    # Configuration - same paths as CLIP-HBA pipeline
    img_dir = Path('/home/wallacelab/teba/multimodal_brain_inspired/THINGS_images')  # input images directory
    stimuli_file = Path('/home/wallacelab/marren/frrsa/sub-01_StimulusMetadata_train_only.csv')  # input csv file
    concept_index_file = Path('/home/wallacelab/marren/frrsa/sub-01_lLOC_concept_index.npy')  # input npy file
    ckpt_root = Path("/home/wallacelab/teba/pretrain_checkpoint/baseline_yfcc15m_vitb32_20250725_111728_ddp_nodeNone/checkpoints")
    save_folder = Path('/home/wallacelab/marren/adaptive-clip-from-scratch/CLIP-HBA/output/baseline_clip_720_concepts')  # output path
    
    # SANITY CHECK: Verify paths
    print("🔍 SANITY CHECK: Verifying input paths...")
    for path, name in [(stimuli_file, "Stimuli CSV"), (concept_index_file, "Concept Index"), (ckpt_root, "Checkpoints")]:
        if not path.exists():
            print(f"❌ Error: {name} path not found: {path}")
            return
        else:
            print(f"✅ {name} path exists: {path}")
    
    # SANITY CHECK: Verify images directory
    if not os.path.exists(img_dir):
        print(f"❌ Error: Images directory not found: {img_dir}")
        return
    else:
        print(f"✅ Images directory exists: {img_dir}")
        # Count images in directory
        image_count = sum(len(files) for _, _, files in os.walk(img_dir) if files)
        print(f"📊 Total images found in directory: {image_count}")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # SANITY CHECK: Create output directory
    print(f"📁 Creating output directory: {save_folder}")
    save_folder.mkdir(parents=True, exist_ok=True)
    print(f"✅ Output directory ready: {save_folder}")
    
    # Data preparation
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # SANITY CHECK: Load dataset
    print("📊 SANITY CHECK: Loading dataset...")
    dataset = THINGSInferenceDataset(stimuli_file, concept_index_file, img_dir, transform)
    print(f"✅ Dataset loaded with {len(dataset)} samples")
    
    # SANITY CHECK: Verify dataset structure
    print("🔍 SANITY CHECK: Verifying dataset structure...")
    sample_image, sample_image_name, sample_concept = dataset[0]
    print(f"✅ Sample image shape: {sample_image.shape}")
    print(f"✅ Sample image name: {sample_image_name}")
    print(f"✅ Sample concept: {sample_concept}")
    
    # num_workers specifies how many subprocesses to use for data loading.
    # Setting num_workers=4 means that 4 worker processes will load data in parallel,
    # which can speed up data loading, especially for large datasets.
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, 
                           num_workers=4, pin_memory=True)
    print(f"✅ DataLoader created with batch_size=32, num_workers=4")
    print(f"📊 Total batches: {len(dataloader)}")
    
    # SANITY CHECK: Get checkpoint files
    print("🔍 SANITY CHECK: Scanning for checkpoint files...")
    ckpt_files = get_checkpoint_files(ckpt_root)
    if not ckpt_files:
        print("❌ No checkpoint files found!")
        return
    
    print(f"✅ Found {len(ckpt_files)} checkpoint files to process")
    print("📋 Checkpoint files found:")
    for i, ckpt in enumerate(ckpt_files[:5]):  # Show first 5
        print(f"   {i+1}. {ckpt.name}")
    if len(ckpt_files) > 5:
        print(f"   ... and {len(ckpt_files) - 5} more")
    
    # Process each checkpoint
    for i, ckpt_path in enumerate(ckpt_files):
        print(f"\n{'='*60}")
        print(f"🔄 Processing checkpoint {i+1}/{len(ckpt_files)}: {ckpt_path.name}")
        print(f"{'='*60}")
        
        # SANITY CHECK: Extract epoch number from filename
        print(f"🔍 SANITY CHECK: Extracting epoch from filename...")
        epoch_match = re.search(r'epoch[=_]?(\d+)', ckpt_path.name)
        if epoch_match:
            epoch = epoch_match.group(1)
            print(f"✅ Extracted epoch: {epoch}")
        else:
            print(f"❌ Could not extract epoch from {ckpt_path.name}")
            continue
        
        # SANITY CHECK: Load model
        print(f"🔍 SANITY CHECK: Loading model from checkpoint...")
        model = load_clip_model_from_checkpoint(ckpt_path, device)
        if model is None:
            print(f"❌ Failed to load model from {ckpt_path.name}")
            continue
        print(f"✅ Model loaded successfully")
        
        try:
            # SANITY CHECK: Run inference
            print(f"🔍 SANITY CHECK: Running inference on epoch {epoch}...")
            print(f"📊 Extracting embeddings from ALL layers")
            layer_embeddings_dict = run_clip_inference(model, dataloader, device)
            
            # SANITY CHECK: Verify embedding output
            print(f"🔍 SANITY CHECK: Verifying embedding output...")
            print(f"✅ Number of layers processed: {len(layer_embeddings_dict)}")
            print(f"✅ Layer names: {list(layer_embeddings_dict.keys())}")
            
            # Show sample information for each layer
            for layer_name, layer_df in layer_embeddings_dict.items():
                print(f"✅ Layer {layer_name}: DataFrame shape {layer_df.shape}")
                print(f"   - Columns: ['image', 'concept', + {layer_df.shape[1]-2} embedding features]")
                print(f"   - Images: {len(layer_df)}, Concepts: {layer_df['concept'].nunique()}")
            
            # SANITY CHECK: Save embeddings dictionary
            print(f"🔍 SANITY CHECK: Saving embeddings dictionary...")
            embedding_save_path = save_folder / f"8640_image_embeddings_epoch{epoch}.pkl"
            
            with open(embedding_save_path, 'wb') as f:
                pickle.dump(layer_embeddings_dict, f)
            print(f"✅ Embeddings dictionary saved to {embedding_save_path}")
            
            # SANITY CHECK: Verify file was created and test loading
            if embedding_save_path.exists():
                file_size = embedding_save_path.stat().st_size / (1024*1024)  # MB
                print(f"✅ File created successfully, size: {file_size:.2f} MB")
                
                # Test loading the pickle file
                try:
                    with open(embedding_save_path, 'rb') as f:
                        test_dict = pickle.load(f)
                    print(f"✅ Pickle file loads correctly with {len(test_dict)} layers")
                except Exception as e:
                    print(f"❌ Warning: Could not load pickle file: {e}")
            else:
                print(f"❌ Warning: File was not created!")
            
            print("-----------------------------------------------\n")
            
        except Exception as e:
            print(f"❌ Error processing epoch {epoch}: {e}")
            import traceback
            print(f"❌ Full error traceback:")
            traceback.print_exc()
        finally:
            # SANITY CHECK: Cleanup
            print(f"🧹 SANITY CHECK: Cleaning up memory...")
            del model
            cleanup_gpu_memory()
            print(f"✅ Memory cleaned up")
    
    # SANITY CHECK: Final summary
    print(f"\n{'='*60}")
    print(f"🎉 EXTRACTION COMPLETE!")
    print(f"{'='*60}")
    print(f"📁 Results saved in: {save_folder}")
    print(f"📊 Total checkpoints processed: {len(ckpt_files)}")
    
    # SANITY CHECK: Count output files
    output_files = list(save_folder.glob("*.pkl"))
    print(f"📄 Output files created: {len(output_files)}")
    if output_files:
        print(f"📋 Output files:")
        for file in output_files:
            file_size = file.stat().st_size / (1024*1024)  # MB
            print(f"   - {file.name} ({file_size:.2f} MB)")
    
    print(f"✅ Script execution completed successfully!")


if __name__ == "__main__":
    main()