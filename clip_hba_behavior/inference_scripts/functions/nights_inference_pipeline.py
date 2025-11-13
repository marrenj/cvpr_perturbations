import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
import json
from datetime import datetime
import sys
sys.path.append('../')
from src.models.CLIPs.clip_hba import clip
from functions.nod_inference_pipeline import CLIPHBA, DoRALayer, apply_dora_to_ViT, load_clip_to_cpu
from functions.spose_dimensions import classnames66


class NIGHTSTripletDataset(Dataset):
    """Dataset for NIGHTS triplet evaluation."""
    
    def __init__(self, nights_dir, split='test', transform=None):
        """
        Args:
            nights_dir: Path to NIGHTS dataset directory
            split: 'train', 'val', or 'test'
            transform: Image transformations
        """
        self.nights_dir = Path(nights_dir)
        self.split = split
        self.transform = transform
        
        # Load triplet metadata
        csv_path = self.nights_dir / 'data.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f"NIGHTS data.csv not found at {csv_path}")
        
        # Load all triplets and filter by split
        all_triplets = pd.read_csv(csv_path)

        # Filter to only include rows where split column matches requested split
        if 'split' not in all_triplets.columns:
            raise ValueError(f"Column 'split' not found in data.csv. Available columns: {all_triplets.columns.tolist()}")

        self.triplets = all_triplets[all_triplets['split'] == split].reset_index(drop=True)

        if len(self.triplets) == 0:
            raise ValueError(f"No triplets found for split '{split}' in data.csv. Available splits: {all_triplets['split'].unique().tolist()}")

        print(f"Loaded {len(self.triplets)} triplets from NIGHTS {split} split (filtered from {len(all_triplets)} total triplets)")
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        row = self.triplets.iloc[idx]
        
        # Load images
        ref_img = Image.open(self.nights_dir / row['ref_path']).convert('RGB')
        img0 = Image.open(self.nights_dir / row['left_path']).convert('RGB')
        img1 = Image.open(self.nights_dir / row['right_path']).convert('RGB')
        
        # Apply transforms
        if self.transform:
            ref_img = self.transform(ref_img)
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        # Determine winner: 0 if left won (left_vote=1), 1 if right won (right_vote=1)
        left_vote = int(row['left_vote'])
        right_vote = int(row['right_vote'])
    
        if left_vote == 1 and right_vote == 0:
            winner = 0  # Left image is the winner
        elif left_vote == 0 and right_vote == 1:
            winner = 1  # Right image is the winner
        else:
            raise ValueError(f"Invalid vote combination: left_vote={left_vote}, right_vote={right_vote}. Exactly one should be 1.")
        
        return ref_img, img0, img1, winner, idx


class CLIPHBAEmbedder:
    """Wrapper for CLIP-HBA model to extract embeddings."""
    
    def __init__(self, model, device, use_image_features=False):
        """
        Args:
            model: CLIPHBA model instance
            device: torch device
            use_image_features: If True, extract image features before classification.
                              If False, use the 66D behavior prediction as embedding.
        """
        self.model = model
        self.device = device
        self.use_image_features = use_image_features
        
        # Set model to eval mode
        self.model.eval()
        
        # Hook to extract intermediate features if needed
        self.features = None
        if use_image_features:
            self._register_feature_hook()
    
    def _register_feature_hook(self):
        """Register hook to extract image features from CLIP."""
        def hook_fn(module, input, output):
            self.features = output
        
        # Register hook on the image encoder output
        # This extracts features before the text-image similarity computation
        self.model.clip_model.visual.register_forward_hook(hook_fn)
    
    def embed(self, image):
        """
        Extract embedding for a batch of images.
        
        Args:
            image: Tensor of shape [batch_size, 3, 224, 224]
        
        Returns:
            embedding: Tensor of shape [batch_size, embedding_dim]
        """
        with torch.no_grad():
            if self.use_image_features:
                # Extract intermediate visual features
                _ = self.model(image)
                embedding = self.features.float()
                # Flatten if needed
                if len(embedding.shape) > 2:
                    embedding = embedding.view(embedding.size(0), -1)
            else:
                # Use the 66D behavior predictions as embeddings
                embedding = self.model(image)
        
        # Normalize embeddings for cosine similarity
        embedding = F.normalize(embedding, p=2, dim=-1)
        
        return embedding
    
    def compute_distance(self, ref_emb, comp_emb):
        """
        Compute perceptual distance between reference and comparison embeddings.
        
        Args:
            ref_emb: Reference embeddings [batch_size, embed_dim]
            comp_emb: Comparison embeddings [batch_size, embed_dim]
        
        Returns:
            distance: Cosine distance (1 - cosine_similarity)
        """
        # Cosine similarity
        cos_sim = F.cosine_similarity(ref_emb, comp_emb, dim=-1)
        # Convert to distance (lower = more similar)
        distance = 1 - cos_sim
        return distance


def cache_dataloader_to_pinned_memory(data_loader):
    """
    Cache all batches from DataLoader to pinned memory for faster GPU transfers.
    
    Args:
        data_loader: DataLoader instance
    
    Returns:
        cached_batches: List of tuples (ref_imgs, img0s, img1s, winners, indices)
                       with images in pinned memory
    """
    cached_batches = []
    caching_bar = tqdm(enumerate(data_loader),
                       total=len(data_loader),
                       desc="Caching images")

    with torch.no_grad():
        for _, (ref_imgs, img0s, img1s, winners, indices) in caching_bar:
            # Only pin if not already pinned (DataLoader pin_memory=False now)
            if not ref_imgs.is_pinned():
                ref_imgs = ref_imgs.pin_memory()
            if not img0s.is_pinned():
                img0s = img0s.pin_memory()
            if not img1s.is_pinned():
                img1s = img1s.pin_memory()
        
            cached_batches.append((
                ref_imgs,
                img0s,
                img1s,
                winners,  # Keep on CPU
                indices   # Keep on CPU
            ))

    return cached_batches


def cache_nights_dataset(nights_dir, split='test', batch_size=32):
    """
    Cache NIGHTS dataset batches without running evaluation.
    
    Args:
        nights_dir: Path to NIGHTS dataset
        split: 'train', 'val', or 'test'
        batch_size: Batch size for caching
    
    Returns:
        cached_batches: List of cached batches in pinned memory
    """
    # Setup transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.52997664, 0.48070561, 0.41943838],
                           std=[0.27608301, 0.26593025, 0.28238822])
    ])
    
    # Create dataset and dataloader
    dataset = NIGHTSTripletDataset(nights_dir, split=split, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=8, pin_memory=False, persistent_workers=True, prefetch_factor=2)

    # Cache batches to pinned memory
    print(f"Caching images for NIGHTS {split} split...")
    cached_batches = cache_dataloader_to_pinned_memory(dataloader)
    del dataloader  # Free memory
    
    return cached_batches


def evaluate_nights(model, nights_dir, split='test', batch_size=32, 
                    device='cuda', use_image_features=False, cached_batches=None):
    """
    Evaluate CLIP-HBA on NIGHTS dataset.
    
    Args:
        model: CLIPHBA model
        nights_dir: Path to NIGHTS dataset
        split: 'train', 'val', or 'test'
        batch_size: Batch size for evaluation
        device: Device to use
        use_image_features: Whether to use image features or behavior predictions
        cached_batches: Pre-cached batches (optional). If None, will create and cache.
    
    Returns:
        results: Dictionary containing accuracy and detailed results
        cached_batches: Cached batches for reuse (if created)
    """
    # Setup transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.52997664, 0.48070561, 0.41943838],
                           std=[0.27608301, 0.26593025, 0.28238822])
    ])
    
    # Create dataset and dataloader if cached_batches not provided
    if cached_batches is None:
        dataset = NIGHTSTripletDataset(nights_dir, split=split, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                               num_workers=8, pin_memory=False, persistent_workers=True, prefetch_factor=2)

        # Cache batches to pinned memory
        print(f"\nCaching images for NIGHTS {split} split...")
        cached_batches = cache_dataloader_to_pinned_memory(dataloader)
        del dataloader  # Free memory
    
    # Create embedder
    embedder = CLIPHBAEmbedder(model, device, use_image_features=use_image_features)
    
    # Evaluation loop
    correct = 0
    total = 0
    all_predictions = []
    all_ground_truths = []
    all_distances = []
    
    print(f"\nEvaluating on NIGHTS {split} split...")
    progress_bar = tqdm(enumerate(cached_batches), desc=f"Evaluating {split}")
    
    with torch.no_grad():
        for batch_idx, (pinned_ref_imgs, pinned_img0s, pinned_img1s, winners, indices) in progress_bar:
            # Transfer to GPU with non-blocking
            ref_imgs = pinned_ref_imgs.to(device, non_blocking=True)
            img0s = pinned_img0s.to(device, non_blocking=True)
            img1s = pinned_img1s.to(device, non_blocking=True)
        
            # Concatenate all images into one batch for a single forward pass
            all_imgs = torch.cat([ref_imgs, img0s, img1s], dim=0)  # [3*batch_size, 3, 224, 224]
        
            # Single forward pass for all images
            with torch.amp.autocast('cuda'):  # Add mixed precision
                all_embs = embedder.embed(all_imgs)
        
            # Split embeddings back
            batch_size = ref_imgs.size(0)
            ref_embs = all_embs[:batch_size]
            img0_embs = all_embs[batch_size:2*batch_size]
            img1_embs = all_embs[2*batch_size:]
        
            # Compute distances
            dist_0 = embedder.compute_distance(ref_embs, img0_embs)
            dist_1 = embedder.compute_distance(ref_embs, img1_embs)
            
            # Predict: which image is MORE SIMILAR (lower distance)
            predictions = (dist_1 < dist_0).long()  # 1 if img1 is closer, 0 if img0 is closer
            
            # Compute accuracy for this batch (keep on GPU, no sync)
            winners_gpu = winners.to(device, non_blocking=True)
            batch_correct = (predictions == winners_gpu).sum()
            correct += batch_correct.item()  # Only sync once
            total += len(winners)

            # Store results (batch CPU transfers)
            all_predictions.append(predictions.cpu())  # Append tensor, convert later
            all_ground_truths.append(winners.cpu())  # Append tensor
            all_distances.append((dist_0.cpu(), dist_1.cpu()))  # Append tuple of tensors
            
            # Update progress bar
            current_acc = (correct / total) * 100
            progress_bar.set_postfix({'Accuracy': f'{current_acc:.2f}%'})

    # Convert to numpy after loop (more efficient)
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_ground_truths = torch.cat(all_ground_truths, dim=0).numpy()
    all_distances = [(d0.numpy(), d1.numpy()) for d0, d1 in all_distances]
    
    # Compute final accuracy
    accuracy = (correct / total) * 100
    
    # Prepare results
    results = {
        'split': split,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'predictions': all_predictions,
        'ground_truths': all_ground_truths,
        'distances': all_distances,
        'use_image_features': use_image_features
    }
    
    print(f"\n{split.upper()} Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    return results, cached_batches


def run_nights_inference(config):
    """
    Run NIGHTS inference using the provided configuration.
    Loops through all epochs in dora_params_path directory.
    
    Args:
        config (dict): Configuration dictionary containing:
            - nights_dir (str or Path): Path to NIGHTS dataset directory
            - backbone (str): CLIP backbone model name (default: 'RN50')
            - load_hba (bool): Whether to load HBA (DoRA) weights
            - dora_params_path (str or Path, optional): Path to DoRA parameters directory
            - dora_weights (str or Path, optional): Path to single DoRA weights file (alternative to dora_params_path)
            - training_res_path (str or Path, optional): Path to training results CSV for filtering epochs
            - use_image_features (bool): Use image features instead of 66D behavior predictions
            - splits (list): Which splits to evaluate on (default: ['val', 'test'])
            - batch_size (int): Batch size for evaluation (default: 32)
            - device (str): Device to use (default: 'cuda:0')
            - output_dir (str or Path): Directory to save results (default: './nights_results')
            - save_predictions (bool): Save detailed predictions to CSV (default: False)
    """
    # Setup device
    device_str = config.get('device', 'cuda:0')
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup output directory
    output_dir = Path(config.get('output_dir', './nights_results'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    print("\nInitializing CLIP-HBA model...")
    classnames = classnames66
    backbone = config.get('backbone', 'RN50')
    pos_embedding = False if backbone == 'RN50' else True
    
    model = CLIPHBA(classnames=classnames, 
                    backbone_name=backbone,
                    pos_embedding=pos_embedding)
    
    # Apply DoRA if specified
    load_hba = config.get('load_hba', False)
    if load_hba:
        print("Applying DoRA layers...")
        apply_dora_to_ViT(model, 
                         n_vision_layers=2,
                         n_transformer_layers=1,
                         r=32,
                         dora_dropout=0.1)
    
    model.to(device)
    model.eval()
    
    use_image_features = config.get('use_image_features', False)
    print(f"Model: CLIP-HBA with {backbone} backbone")
    print(f"DoRA enabled: {load_hba}")
    print(f"Using image features: {use_image_features}")
    
    # Get configuration
    splits = config.get('splits', ['val', 'test'])
    batch_size = config.get('batch_size', 32)
    nights_dir = config['nights_dir']
    save_predictions = config.get('save_predictions', False)
    
    # Determine if we're processing a directory of epochs or a single file
    dora_params_path = config.get('dora_params_path', None)
    dora_weights = config.get('dora_weights', None)
    
    # Cache datasets once for all splits and epochs (only if processing multiple epochs)
    cached_batches_by_split = {}
    if dora_params_path:
        for split in splits:
            cached_batches = cache_nights_dataset(nights_dir, split=split, batch_size=batch_size)
            cached_batches_by_split[split] = cached_batches
            print(f"Cached {len(cached_batches)} batches for {split} split")
    
    if dora_params_path:
        # Process all epochs in directory
        dora_params_path = Path(dora_params_path)
        
        if not dora_params_path.exists():
            raise FileNotFoundError(f"DoRA parameters directory not found: {dora_params_path}")
        
        # List all the files in the dora_params_path
        dora_params_files = [f.name for f in dora_params_path.iterdir() if f.is_file() and f.name.endswith('.pth')]
        dora_params_files = sorted(dora_params_files, key=lambda f: int(f.split('_')[0].replace('epoch', '')))
        
        # Filter epochs based on minimum test loss if training results CSV is provided
        if 'training_res_path' in config and config['training_res_path']:
            training_res_path = Path(config['training_res_path'])
            if training_res_path.exists():
                training_df = pd.read_csv(training_res_path)
                
                # Find the epoch with minimum test loss
                if 'test_loss' in training_df.columns:
                    min_test_loss_idx = training_df['test_loss'].idxmin()
                    min_test_loss_epoch = int(training_df.loc[min_test_loss_idx, 'epoch'])
                    
                    print(f"Minimum test loss occurred at epoch {min_test_loss_epoch}")
                    print(f"Original epoch files: {len(dora_params_files)}")
                    
                    # Filter DoRA parameter files to only include epochs up to minimum test loss
                    filtered_dora_files = []
                    for file in dora_params_files:
                        epoch_num = int(file.split('_')[0].replace('epoch', ''))
                        if epoch_num <= min_test_loss_epoch:
                            filtered_dora_files.append(file)
                    
                    dora_params_files = filtered_dora_files
                    print(f"Filtered epoch files: {len(dora_params_files)}")
                    total_files = len([f for f in dora_params_path.iterdir() if f.is_file() and f.name.endswith('.pth')])
                    print(f"Skipping {total_files - len(dora_params_files)} epochs after minimum test loss\n")
        
        # Process each epoch
        for file in dora_params_files:
            # Extract epoch number from filename
            epoch = file.split('_')[0].replace('epoch', '')
            
            # Check if results already exist for this epoch
            existing_results = list(output_dir.glob(f'*epoch{epoch}*'))
            if existing_results:
                print(f"Epoch {epoch} already processed, skipping...")
                continue
            
            print(f"\n{'='*80}")
            print(f"Processing epoch {epoch}")
            print(f"{'='*80}\n")
            
            # Load DoRA parameters for this epoch
            dora_file_path = dora_params_path / file
            dora_params_state_dict = torch.load(dora_file_path, map_location='cpu', weights_only=True)
            model.load_state_dict(dora_params_state_dict, strict=False)
            
            # Run evaluation on each split using cached batches
            all_results = {}
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            for split in splits:
                results, _ = evaluate_nights(
                    model=model,
                    nights_dir=nights_dir,
                    split=split,
                    batch_size=batch_size,
                    device=device,
                    use_image_features=use_image_features,
                    cached_batches=cached_batches_by_split[split]  # Reuse cached batches
                )
                
                all_results[split] = results
                
                # Save predictions if requested
                if save_predictions:
                    pred_df = pd.DataFrame({
                        'prediction': results['predictions'],
                        'ground_truth': results['ground_truths'],
                        'dist_img0': [d[0] for d in results['distances']],
                        'dist_img1': [d[1] for d in results['distances']],
                        'correct': [p == g for p, g in zip(results['predictions'], 
                                                           results['ground_truths'])]
                    })
                    pred_path = output_dir / f'predictions_{split}_epoch{epoch}_{timestamp}.csv'
                    pred_df.to_csv(pred_path, index=False)
                    print(f"Saved predictions to {pred_path}")
            
            # Save summary results with epoch information
            summary = {
                'epoch': int(epoch),
                'timestamp': timestamp,
                'model_config': {
                    'backbone': backbone,
                    'load_hba': load_hba,
                    'dora_weights': str(dora_file_path),
                    'use_image_features': use_image_features
                },
                'results': {
                    split: {
                        'accuracy': results['accuracy'],
                        'correct': results['correct'],
                        'total': results['total']
                    }
                    for split, results in all_results.items()
                }
            }
            
            summary_path = output_dir / f'summary_epoch{epoch}_{timestamp}.json'
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\nSummary saved to {summary_path}")
            
            # Print final summary
            print("\n" + "="*60)
            print(f"EVALUATION SUMMARY - EPOCH {epoch}")
            print("="*60)
            for split, results in all_results.items():
                print(f"{split.upper()}: {results['accuracy']:.2f}% ({results['correct']}/{results['total']})")
            print("="*60)
            
            torch.cuda.empty_cache()
    
    elif dora_weights:
        # Process single file (backward compatibility)
        print(f"Loading DoRA weights from {dora_weights}")
        dora_state_dict = torch.load(dora_weights, map_location='cpu', weights_only=True)
        model.load_state_dict(dora_state_dict, strict=False)
        
        # Run evaluation on each split
        all_results = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for split in splits:
            results, _ = evaluate_nights(
                model=model,
                nights_dir=nights_dir,
                split=split,
                batch_size=batch_size,
                device=device,
                use_image_features=use_image_features,
                cached_batches=None
            )
            
            all_results[split] = results
            
            # Save predictions if requested
            if save_predictions:
                pred_df = pd.DataFrame({
                    'prediction': results['predictions'],
                    'ground_truth': results['ground_truths'],
                    'dist_img0': [d[0] for d in results['distances']],
                    'dist_img1': [d[1] for d in results['distances']],
                    'correct': [p == g for p, g in zip(results['predictions'], 
                                                       results['ground_truths'])]
                })
                pred_path = output_dir / f'predictions_{split}_{timestamp}.csv'
                pred_df.to_csv(pred_path, index=False)
                print(f"Saved predictions to {pred_path}")
        
        # Save summary results
        summary = {
            'timestamp': timestamp,
            'model_config': {
                'backbone': backbone,
                'load_hba': load_hba,
                'dora_weights': str(dora_weights),
                'use_image_features': use_image_features
            },
            'results': {
                split: {
                    'accuracy': results['accuracy'],
                    'correct': results['correct'],
                    'total': results['total']
                }
                for split, results in all_results.items()
            }
        }
        
        summary_path = output_dir / f'summary_{timestamp}.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary saved to {summary_path}")
        
        # Print final summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        for split, results in all_results.items():
            print(f"{split.upper()}: {results['accuracy']:.2f}% ({results['correct']}/{results['total']})")
        print("="*60)
    
    else:
        if load_hba:
            print("Warning: load_hba specified but no dora_params_path or dora_weights provided")
    
    # Compare to baselines
    print("\nBaseline Comparisons (from DreamSim paper):")
    print("  CLIP ViT-B/32:     Val: 94.9%, Test: 93.6%")
    print("  OpenCLIP ViT-B/32: Val: 95.6%, Test: 95.3%")
    print("  DINO ViT-B/16:     Val: 95.6%, Test: 94.8%")
    print("  DreamSim Ensemble: Val: 96.9%, Test: 96.2%")


def main():
    parser = argparse.ArgumentParser(description='Evaluate CLIP-HBA on NIGHTS dataset')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default='RN50',
                       choices=['RN50', 'ViT-B/32', 'ViT-B/16'],
                       help='CLIP backbone architecture')
    parser.add_argument('--load_hba', action='store_true',
                       help='Load HBA (DoRA) weights')
    parser.add_argument('--dora_weights', type=str, default=None,
                       help='Path to DoRA weights file')
    parser.add_argument('--use_image_features', action='store_true',
                       help='Use image features instead of 66D behavior predictions')
    
    # Dataset arguments
    parser.add_argument('--nights_dir', type=str, required=True,
                       help='Path to NIGHTS dataset directory')
    parser.add_argument('--splits', nargs='+', default=['val', 'test'],
                       choices=['train', 'val', 'test'],
                       help='Which splits to evaluate on')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use for evaluation')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./nights_results',
                       help='Directory to save results')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save detailed predictions to CSV')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    print("\nInitializing CLIP-HBA model...")
    classnames = classnames66
    pos_embedding = False if args.backbone == 'RN50' else True
    
    model = CLIPHBA(classnames=classnames, 
                    backbone_name=args.backbone,
                    pos_embedding=pos_embedding)
    
    # Apply DoRA if specified
    if args.load_hba:
        print("Applying DoRA layers...")
        apply_dora_to_ViT(model, 
                         n_vision_layers=2,
                         n_transformer_layers=1,
                         r=32,
                         dora_dropout=0.1)
        
        if args.dora_weights:
            print(f"Loading DoRA weights from {args.dora_weights}")
            dora_state_dict = torch.load(args.dora_weights, map_location='cpu', weights_only=True)
            model.load_state_dict(dora_state_dict, strict=False)
        else:
            print("Warning: --load_hba specified but no --dora_weights provided")
    
    model.to(device)
    model.eval()
    
    print(f"Model: CLIP-HBA with {args.backbone} backbone")
    print(f"DoRA enabled: {args.load_hba}")
    print(f"Using image features: {args.use_image_features}")
    
    # Run evaluation on each split
    all_results = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for split in args.splits:
        results = evaluate_nights(
            model=model,
            nights_dir=args.nights_dir,
            split=split,
            batch_size=args.batch_size,
            device=device,
            use_image_features=args.use_image_features
        )
        
        all_results[split] = results
        
        # Save predictions if requested
        if args.save_predictions:
            pred_df = pd.DataFrame({
                'prediction': results['predictions'],
                'ground_truth': results['ground_truths'],
                'dist_img0': [d[0] for d in results['distances']],
                'dist_img1': [d[1] for d in results['distances']],
                'correct': [p == g for p, g in zip(results['predictions'], 
                                                   results['ground_truths'])]
            })
            pred_path = output_dir / f'predictions_{split}_{timestamp}.csv'
            pred_df.to_csv(pred_path, index=False)
            print(f"Saved predictions to {pred_path}")
    
    # Save summary results
    summary = {
        'timestamp': timestamp,
        'model_config': {
            'backbone': args.backbone,
            'load_hba': args.load_hba,
            'dora_weights': args.dora_weights,
            'use_image_features': args.use_image_features
        },
        'results': {
            split: {
                'accuracy': results['accuracy'],
                'correct': results['correct'],
                'total': results['total']
            }
            for split, results in all_results.items()
        }
    }
    
    summary_path = output_dir / f'summary_{timestamp}.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to {summary_path}")
    
    # Print final summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for split, results in all_results.items():
        print(f"{split.upper()}: {results['accuracy']:.2f}% ({results['correct']}/{results['total']})")
    print("="*60)
    
    # Compare to baselines
    print("\nBaseline Comparisons (from DreamSim paper):")
    print("  CLIP ViT-B/32:     Val: 94.9%, Test: 93.6%")
    print("  OpenCLIP ViT-B/32: Val: 95.6%, Test: 95.3%")
    print("  DINO ViT-B/16:     Val: 95.6%, Test: 94.8%")
    print("  DreamSim Ensemble: Val: 96.9%, Test: 96.2%")


if __name__ == '__main__':
    main()
