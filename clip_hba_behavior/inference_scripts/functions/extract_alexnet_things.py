#!/usr/bin/env python3
"""
Extract embeddings for the 48 THINGS inference images from AlexNet seed 1 checkpoints.
Uses your exact data paths and integrates with existing pipeline functions.
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as Models
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import re
from typing import Dict, List, Optional

# Define the hook registration function directly (to avoid import issues)
def _register_hooks_for_arch(model, arch):
    """Register forward hooks for AlexNet layers from SEEDS project"""
    key = arch.lower()
    
    activations = {}
    layer_names = []

    def get_hook(name):
        def hook(_, __, out):
            activations[name] = out.detach()
        return hook

    def add(m, name):
        layer_names.append(name)
        m.register_forward_hook(get_hook(name))

    if key == 'alexnet':
        # # AlexNet layer hooks based on your existing code from imagenet_weighted_embeddings.py
        add(model.features[2],  'features.pool1')   # stem/early boundary
        add(model.features[5],  'features.pool2')   # early boundary  
        add(model.features[6],  'features.conv3')   # mid
        add(model.features[8],  'features.conv4')   # mid-late
        add(model.features[10], 'features.conv5')   # late conv
        add(model.features[12], 'features.pool5')   # final spatial compress
        add(model.classifier[1],'classifier.fc1')   # penultimate stack (1)
        add(model.classifier[4],'classifier.fc2')   # penultimate stack (2)
        add(model.classifier[6],'classifier.fc3')   # Linear-num_classes (output)
    else:
        print(f"[WARN] No predefined hooks for {arch}")

    return layer_names, activations

class THINGSInferenceDataset(Dataset):
    """Define Dataset for the 48 THINGS inference images."""
    
    def __init__(self, csv_file: Path, images_dir: Path, transform=None):
        self.images_dir = Path(images_dir)
        self.transform = transform
        
        # Load CSV file (first column should contain image names)
        df = pd.read_csv(csv_file, index_col=0)
        self.image_names = df.iloc[:, 0].tolist()
        
        print(f"Loaded {len(self.image_names)} images from {csv_file}")
        print(f"Sample image names: {self.image_names[:3]}")
        
        # Verify images exist
        existing_images = []
        for img_name in self.image_names:
            img_path = self.images_dir / img_name
            if img_path.exists():
                existing_images.append(img_name)
            else:
                print(f"Warning: {img_path} not found")
        
        self.image_names = existing_images
        print(f"Found {len(existing_images)} existing images in {self.images_dir}")
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = self.images_dir / img_name
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, Path(img_name).stem

def _feat_to_matrix(feat: torch.Tensor, layer_name: str, batch_size: int) -> torch.Tensor:
    """Convert feature tensor to [batch_size, feature_dim] matrix."""
    if feat.dim() == 4:  # Conv layers: [B, C, H, W]
        feat = F.adaptive_avg_pool2d(feat, (1, 1)).reshape(feat.size(0), -1)
        return feat.contiguous()
    
    if feat.dim() == 3:  # Should not happen for AlexNet, but just in case
        feat = feat.mean(dim=1)
        return feat.contiguous()
    
    if feat.dim() == 2:  # Already [batch_size, features]
        return feat.contiguous()
    
    # Fallback: flatten
    return feat.reshape(feat.size(0), -1).contiguous()

def load_alexnet_checkpoint(ckpt_path: Path, device: torch.device, num_classes: int = 100):
    """Load AlexNet model from checkpoint."""
    try:
        # Create AlexNet with CIFAR-100 head (based on your existing code)
        model = Models.alexnet(weights=None)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
        model = model.to(device)
        
        # Load checkpoint parameters 
        state_dict = torch.load(ckpt_path, map_location=device)
        
        # Handle DDP checkpoints (remove 'module.' prefix if present)
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
        
        # Load checkpoint parameters into model
        model.load_state_dict(state_dict)
        
        # Keeps all neurons active
        # During inference, you want all learned features available
        model.eval()
        
        return model
        
    except Exception as e:
        print(f"Error loading AlexNet from {ckpt_path}: {e}")
        return None

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
    
    with torch.no_grad():# turn off gradient, don't need for inference
        for batch_images, batch_names in data_loader:
            batch_images = batch_images.to(device)
            
            # Forward pass, runs images through the model (triggers hooks)
            _ = model(batch_images)
            
            # Collect activations from each layer
            for layer_name in layer_names:
                if layer_name in activations:
                    feat = activations[layer_name]
                    feat_matrix = _feat_to_matrix(feat, layer_name, batch_images.size(0))
                    layer_embeddings[layer_name].append(feat_matrix.cpu().numpy())
                    
            # keeps track of which images correspond to which embeddings
            image_names.extend(batch_names) 
    
    # Concatenate all batches for each layer
    final_embeddings = {}
    for layer_name in layer_names:
        if layer_embeddings[layer_name]:
            # After processing all batches, combines all embeddings for each layer
            final_embeddings[layer_name] = np.vstack(layer_embeddings[layer_name])
            print(f"Layer {layer_name}: {final_embeddings[layer_name].shape}")
        else:
            print(f"Warning: No embeddings collected for layer {layer_name}")
    # Result: [total_images × feature_dimensions] array for each layer
    # Add image names to the results
    final_embeddings['image_names'] = np.array(image_names)
    
    return final_embeddings

def extract_alexnet_things_embeddings():
    """Main function to extract AlexNet embeddings for THINGS inference images."""
    
    # Your specific paths
    things_csv_file = Path("/home/wallacelab/marren/adaptive-clip-from-scratch/CLIP-HBA/Data/spose_embedding66d_rescaled_48val_reordered.csv")
    things_images_dir = Path("/home/wallacelab/marren/adaptive-clip-from-scratch/CLIP-HBA/Data/Things1854")
    ckpt_root = Path("/home/wallacelab/teba/seeds_checkpoints")
    
    # Verify paths once at the beginning
    if not things_csv_file.exists():
        print(f"ERROR: CSV file not found: {things_csv_file}")
        return
    if not things_images_dir.exists():
        print(f"ERROR: Images directory not found: {things_images_dir}")
        return
    
    # Configuration
    arch = 'alexnet'
    target_seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    num_workers = 4
    output_dir = Path("./10_seeds_alexnet_things_embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # # Create dataset and dataloader once (reuse for all seeds)
    # transform = T.Compose([
    #     T.Resize(224),  # Direct resize, no center crop
    #     T.ToTensor(),
    #     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    
    # Use same image crop as training
    transform = T.Compose([
    T.Resize(256),        # Match training
    T.CenterCrop(224),    # Match training  
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
    
    things_dataset = THINGSInferenceDataset(
        csv_file=things_csv_file,
        images_dir=things_images_dir,
        transform=transform
    )
    
    things_loader = DataLoader(
        things_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Store results for all seeds
    all_seeds_results = {}
    
    for target_seed in target_seeds:
        print(f"\n=== Extracting {arch} embeddings for seed {target_seed} ===")
        print(f"Device: {device}")
        
        # Find seed directory
        arch_dir = ckpt_root / arch
        seed_dir = arch_dir / f"seed_{target_seed}"
        
        if not seed_dir.exists():
            print(f"ERROR: Seed directory not found: {seed_dir}")
            continue  # Skip this seed, continue with next
        
        print(f"Using seed directory: {seed_dir}")
        
        # Find checkpoint files
        ckpt_files = list(seed_dir.glob(f"{arch}_epoch*.pth"))
        if not ckpt_files:
            print(f"ERROR: No checkpoint files found in {seed_dir}")
            continue  # Skip this seed, continue with next
        
        # Sort by epoch number
        def extract_epoch(path):
            match = re.search(r'epoch(\d+)', path.name)
            return int(match.group(1)) if match else 0
        
        ckpt_files = sorted(ckpt_files, key=extract_epoch)
        print(f"Found {len(ckpt_files)} checkpoints: {[f.name for f in ckpt_files]}")
        
        # Process each checkpoint for this seed
        seed_results = {}
        
        for ckpt_path in ckpt_files:
            print(f"\n--- Processing checkpoint: {ckpt_path.name} ---")
            
            # Load model
            model = load_alexnet_checkpoint(ckpt_path, device, num_classes=100)
            if model is None:
                continue
            
            # Register hooks for intermediate layers
            layer_names, activations = _register_hooks_for_arch(model, arch)
            print(f"Registered hooks for layers: {layer_names}")
            
            # Extract embeddings
            embeddings_dict = extract_embeddings_from_model(
                model, things_loader, layer_names, activations, device
            )
            
            # Store results
            checkpoint_name = ckpt_path.stem
            seed_results[checkpoint_name] = embeddings_dict
            
            # Save individual checkpoint results
            save_path = output_dir / f"{arch}_seed{target_seed}_{checkpoint_name}_things_embeddings.npz"
            np.savez_compressed(save_path, **embeddings_dict)
            print(f"Saved embeddings to: {save_path}")
            
            # Clear GPU memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Save consolidated results for this seed
        if seed_results:  # Only save if we have results
            consolidated_path = output_dir / f"{arch}_seed{target_seed}_all_checkpoints_things_embeddings.npz"
            consolidated = {}
            for checkpoint_name, embeddings_dict in seed_results.items():
                for layer_name, embeddings in embeddings_dict.items():
                    key = f"{checkpoint_name}_{layer_name}"
                    consolidated[key] = embeddings
            
            np.savez_compressed(consolidated_path, **consolidated)
            print(f"Saved consolidated embeddings to: {consolidated_path}")
            
            # Store in overall results
            all_seeds_results[target_seed] = seed_results
            
            print(f"=== Seed {target_seed} Complete: Processed {len(seed_results)} checkpoints ===")
        else:
            print(f"=== Seed {target_seed} Failed: No results ===")
    
    print(f"\n=== ALL SEEDS EXTRACTION COMPLETE ===")
    print(f"Successfully processed seeds: {list(all_seeds_results.keys())}")
    print(f"Results saved in: {output_dir}")
    
    return all_seeds_results

if __name__ == "__main__":
    results = extract_alexnet_things_embeddings()