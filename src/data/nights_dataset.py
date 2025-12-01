import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from pathlib import Path
import pandas as pd
from PIL import Image


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