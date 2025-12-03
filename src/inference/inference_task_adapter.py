from src.data.things_dataset import ThingsDataset
from src.data.nights_dataset import NightsTripletDataset


# InferenceTask abstract base class


# NightsTripletTask(InferenceTask)
def cache_nights_dataloader_to_pinned_memory(data_loader):
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
    
    # Create dataset and dataloader if cached_batches not provided
    if cached_batches is None:
        dataset = NightsTripletDataset(nights_dir, split=split)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                               num_workers=8, pin_memory=False, persistent_workers=True, prefetch_factor=2)

        # Cache batches to pinned memory
        print(f"\nCaching images for NIGHTS {split} split...")
        cached_batches = cache_nights_dataloader_to_pinned_memory(dataloader)
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


# NODInferenceTask(InferenceTask)



# Things48ImagesInferenceTask(InferenceTask)


# 720ThingsConceptsInferenceTask(InferenceTask)


# Initialize dataset
def adapt_inference_task(dataset_type):
    if dataset_type == 'things':
        inference_task = None
    elif dataset_type == 'nights':
        inference_task = evaluate_nights
    elif dataset_type == 'nod':
        inference_task = None
    else:
        raise ValueError(f"Dataset type {dataset_type} not supported")
    return inference_task