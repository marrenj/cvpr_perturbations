#!/usr/bin/env python3
"""
alexnet_720_concepts_extraction.py

Extract embeddings from all AlexNet layers for the 8640 THINGS images at every epoch checkpoint,
then generate 720 concept embeddings by aggregating image embeddings by concept.

This script combines functionality from:
- extract_alexnet_things.py: AlexNet checkpoint loading and layer hook registration
- extract_clip_embeddings_720_concepts.py: Concept embedding generation from image embeddings
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as Models
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import re
import pickle
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
from tqdm import tqdm


def register_alexnet_hooks_all_layers(model: torch.nn.Module) -> Tuple[List[str], Dict[str, torch.Tensor]]:
    """Register forward hooks for ALL AlexNet layers (not just selected ones)."""
    activations = {}
    layer_names = []

    def get_hook(name: str):
        def hook(_, __, out):
            activations[name] = out.detach()
        return hook

    def add_hook(module: torch.nn.Module, name: str):
        layer_names.append(name)
        module.register_forward_hook(get_hook(name))

    # Register ALL features layers (conv and pooling)
    if hasattr(model, 'features'):
        features = model.features
        for idx, layer in enumerate(features):
            layer_type = type(layer).__name__
            add_hook(layer, f'features.{layer_type.lower()}_{idx}')
            print(f"Registered features layer {idx}: {layer_type}")
    
    # Register avgpool
    if hasattr(model, 'avgpool'):
        add_hook(model.avgpool, 'avgpool')
        print("Registered avgpool")
    
    # Register ALL classifier layers (including dropout)
    if hasattr(model, 'classifier'):
        classifier = model.classifier
        for idx, layer in enumerate(classifier):
            layer_type = type(layer).__name__
            add_hook(layer, f'classifier.{layer_type.lower()}_{idx}')
            print(f"Registered classifier layer {idx}: {layer_type}")
    
    print(f"Successfully registered {len(layer_names)} AlexNet layer hooks for ALL layers")
    return layer_names, activations


class THINGSInferenceDataset(Dataset):
    """Dataset for the 8640 THINGS images with concept information."""
    
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


def feature_to_matrix(feat: torch.Tensor) -> torch.Tensor:
    """Convert feature tensor to [batch_size, feature_dim] matrix."""
    if feat.dim() == 4:  # Conv layers: [B, C, H, W]
        return F.adaptive_avg_pool2d(feat, (1, 1)).reshape(feat.size(0), -1)
    elif feat.dim() == 3:  # Should not happen for AlexNet, but just in case
        return feat.mean(dim=1)
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
        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc="Processing images")
        for batch_idx, (batch_images, batch_image_names, batch_concepts) in progress_bar:
            batch_images = batch_images.to(device)
            
            # Forward pass through AlexNet
            try:
                _ = model(batch_images)
            except Exception as e:
                print(f"Error in model forward pass: {e}")
                continue
            
            # Collect activations from hooks
            for layer_name in layer_names:
                if layer_name in activations:
                    feat = activations[layer_name]
                    feat_matrix = feature_to_matrix(feat)
                    layer_embeddings[layer_name].append(feat_matrix.cpu().numpy())
            
            image_names.extend(batch_image_names)
            concepts.extend(batch_concepts)
            
            # Update progress bar description with current batch info
            progress_bar.set_postfix({
                'batch': f'{batch_idx+1}/{len(data_loader)}',
                'images': len(image_names)
            })
    
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


def generate_concept_embeddings(layer_embeddings: Dict[str, np.ndarray]) -> Dict[str, pd.DataFrame]:
    """Generate concept embeddings by aggregating image embeddings by concept."""
    layer_dataframes = {}
    
    for layer_name, embeddings in layer_embeddings.items():
        if layer_name in ['image_names', 'concepts']:
            continue
            
        print(f"🔍 Generating concept embeddings for layer: {layer_name}")
        print(f"✅ Embeddings shape: {embeddings.shape}")
        
        # Create DataFrame with embeddings, image names, and concepts
        df = pd.DataFrame(embeddings)
        df['image'] = layer_embeddings['image_names']
        df['concept'] = layer_embeddings['concepts']
        
        # Group by concept and take mean of embeddings
        concept_embeddings = df.groupby('concept').mean().reset_index()
        
        # Reorder columns to have concept first
        embedding_cols = [col for col in concept_embeddings.columns if col != 'concept']
        concept_embeddings = concept_embeddings[['concept'] + embedding_cols]
        
        layer_dataframes[layer_name] = concept_embeddings
        
        print(f"✅ Concept embeddings shape: {concept_embeddings.shape}")
        print(f"✅ Number of concepts: {len(concept_embeddings)}")
        print(f"✅ Sample concepts: {concept_embeddings['concept'].tolist()[:10]}")
    
    return layer_dataframes


def run_alexnet_inference(model: torch.nn.Module, data_loader: DataLoader, device: torch.device) -> Dict[str, pd.DataFrame]:
    """Run inference using AlexNet model and return concept embeddings from ALL layers."""
    print(f"🔍 SANITY CHECK: Setting up AlexNet model for inference...")
    model.eval()
    model.to(device)
    print(f"✅ Model set to eval mode and moved to {device}")
    
    # Register hooks for ALL layers
    print(f"🔍 SANITY CHECK: Registering hooks for ALL layers...")
    layer_names, activations = register_alexnet_hooks_all_layers(model)
    print(f"✅ Available layers: {layer_names}")
    
    if not layer_names:
        raise ValueError("No layers available for embedding extraction")
    
    print(f"🔍 SANITY CHECK: Starting inference with {len(data_loader)} batches...")
    
    # Extract embeddings from all layers
    layer_embeddings = extract_embeddings_from_model(model, data_loader, layer_names, activations, device)
    
    print(f"🔍 SANITY CHECK: Generating concept embeddings...")
    layer_dataframes = generate_concept_embeddings(layer_embeddings)
    
    print(f"✅ Total images processed: {len(layer_embeddings['image_names'])}")
    print(f"✅ Total concepts: {len(set(layer_embeddings['concepts']))}")
    print(f"✅ Successfully extracted concept embeddings from {len(layer_dataframes)} layers")
    
    return layer_dataframes


def load_alexnet_checkpoint(ckpt_path: Path, device: torch.device, num_classes: int = 100):
    """Load AlexNet model from checkpoint with error handling."""
    print(f"Loading AlexNet model from checkpoint: {ckpt_path}")
    
    try:
        # Create AlexNet with modified head (based on extract_alexnet_things.py)
        model = Models.alexnet(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        
        # Load checkpoint
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        
        # Handle DDP checkpoints (remove 'module.' prefix if present)
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
        
        # Load weights
        model.load_state_dict(state_dict, strict=True)
        
        model = model.to(device)
        model.eval()
        print(f"✅ AlexNet model loaded successfully")
        return model

    except Exception as e:
        print(f"Error loading AlexNet model: {e}")
        return None


def get_checkpoint_files(ckpt_root: Path, seed: int = 1) -> List[Path]:
    """Get and sort AlexNet checkpoint files by epoch for a specific seed."""
    arch_dir = ckpt_root / "alexnet"
    seed_dir = arch_dir / f"seed_{seed}"
    
    if not seed_dir.exists():
        print(f"Error: Seed directory not found: {seed_dir}")
        return []
    
    ckpt_files = list(seed_dir.glob("alexnet_epoch*.pth"))
    
    if not ckpt_files:
        print(f"Error: No AlexNet checkpoint files found in {seed_dir}")
        return []
    
    def extract_epoch(path: Path) -> int:
        """Extract epoch number from checkpoint filename."""
        match = re.search(r'epoch(\d+)', path.name)
        return int(match.group(1)) if match else 0
    
    ckpt_files = sorted(ckpt_files, key=extract_epoch)
    print(f"Found {len(ckpt_files)} AlexNet checkpoints for seed {seed}")
    return ckpt_files


def cleanup_gpu_memory():
    """Clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    """Main function to extract AlexNet embeddings for THINGS inference images."""
    
    # Configuration - using the same paths as extract_clip_embeddings_720_concepts.py
    img_dir = Path('/home/wallacelab/teba/multimodal_brain_inspired/THINGS_images')
    stimuli_file = Path('/home/wallacelab/marren/frrsa/sub-01_StimulusMetadata_train_only.csv')
    concept_index_file = Path('/home/wallacelab/marren/frrsa/sub-01_lLOC_concept_index.npy')
    
    # AlexNet checkpoint paths - using the same structure as extract_alexnet_things.py
    ckpt_root = Path("/home/wallacelab/teba/seeds_checkpoints")
    save_folder = Path('/home/wallacelab/marren/adaptive-clip-from-scratch/CLIP-HBA/output/alexnet_720_concepts')
    seed = 1  # Focus on seed 1 as specified
    
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
    device = torch.device("cuda")
    print(f"🖥️  Using device: {device}")
    
    # Create output directory
    print(f"📁 Creating output directory: {save_folder}")
    save_folder.mkdir(parents=True, exist_ok=True)
    print(f"✅ Output directory ready: {save_folder}")
    
    # Data preparation - using the same transforms as extract_alexnet_things.py
    transform = T.Compose([
        T.Resize(256),        # Match training
        T.CenterCrop(224),    # Match training  
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    print("📊 SANITY CHECK: Loading dataset...")
    dataset = THINGSInferenceDataset(stimuli_file, concept_index_file, img_dir, transform)
    print(f"✅ Dataset loaded with {len(dataset)} samples")
    
    # Verify dataset structure
    print("🔍 SANITY CHECK: Verifying dataset structure...")
    sample_image, sample_image_name, sample_concept = dataset[0]
    print(f"✅ Sample image shape: {sample_image.shape}")
    print(f"✅ Sample image name: {sample_image_name}")
    print(f"✅ Sample concept: {sample_concept}")
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, 
                           num_workers=4, pin_memory=True)
    print(f"✅ DataLoader created with batch_size=32, num_workers=4")
    print(f"📊 Total batches: {len(dataloader)}")
    
    # Get checkpoint files
    print(f"🔍 SANITY CHECK: Scanning for AlexNet checkpoint files (seed {seed})...")
    ckpt_files = get_checkpoint_files(ckpt_root, seed)
    if not ckpt_files:
        print("❌ No AlexNet checkpoint files found!")
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
        
        # Extract epoch number from filename
        print(f"🔍 SANITY CHECK: Extracting epoch from filename...")
        epoch_match = re.search(r'epoch(\d+)', ckpt_path.name)
        if epoch_match:
            epoch = epoch_match.group(1)
            print(f"✅ Extracted epoch: {epoch}")
        else:
            print(f"❌ Could not extract epoch from {ckpt_path.name}")
            continue
        
        # Load model
        print(f"🔍 SANITY CHECK: Loading AlexNet model from checkpoint...")
        model = load_alexnet_checkpoint(ckpt_path, device, num_classes=100)
        if model is None:
            print(f"❌ Failed to load model from {ckpt_path.name}")
            continue
        
        try:
            # Run inference
            print(f"🔍 SANITY CHECK: Running inference on epoch {epoch}...")
            print(f"📊 Extracting concept embeddings from ALL layers")
            layer_embeddings_dict = run_alexnet_inference(model, dataloader, device)
            
            # Verify embedding output
            print(f"🔍 SANITY CHECK: Verifying embedding output...")
            print(f"✅ Number of layers processed: {len(layer_embeddings_dict)}")
            print(f"✅ Layer names: {list(layer_embeddings_dict.keys())}")
            
            # Show sample information for each layer
            for layer_name, layer_df in layer_embeddings_dict.items():
                print(f"✅ Layer {layer_name}: DataFrame shape {layer_df.shape}")
                print(f"   - Columns: ['concept', + {layer_df.shape[1]-1} embedding features]")
                print(f"   - Concepts: {len(layer_df)}")
            
            # Save embeddings dictionary
            print(f"🔍 SANITY CHECK: Saving embeddings dictionary...")
            embedding_save_path = save_folder / f"alexnet_720_concept_embeddings_epoch{epoch}.pkl"
            
            with open(embedding_save_path, 'wb') as f:
                pickle.dump(layer_embeddings_dict, f)
            print(f"✅ Embeddings dictionary saved to {embedding_save_path}")
            
            # Verify file was created and test loading
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
            # Cleanup
            print(f"🧹 SANITY CHECK: Cleaning up memory...")
            del model
            cleanup_gpu_memory()
            print(f"✅ Memory cleaned up")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"🎉 EXTRACTION COMPLETE!")
    print(f"{'='*60}")
    print(f"📁 Results saved in: {save_folder}")
    print(f"📊 Total checkpoints processed: {len(ckpt_files)}")
    
    # Count output files
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
