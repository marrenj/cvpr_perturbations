"""
Test checkpoint reproducibility using EXISTING checkpoints.

This test verifies that loading from an existing checkpoint and training for one epoch
produces the same result as the saved checkpoint for the next epoch.

For example:
- Load checkpoint from epoch 5
- Train for 1 epoch 
- Compare DoRA parameters with existing checkpoint for epoch 6

This validates that your checkpoint loading mechanism can reproduce the original training.
"""

import torch
import torch.nn as nn
import os
import sys

sys.path.append('../')
from functions.cvpr_train_behavior_things_pipeline import (
    seed_everything,
    CLIPHBA,
    apply_dora_to_ViT,
    switch_dora_layers,
    ThingsDataset,
    load_and_restore_rng_state,
    reconstruct_rng_state_for_epoch,
    restore_rng_state_dict
)
from functions.spose_dimensions import classnames66
from torch.utils.data import DataLoader, random_split


def extract_dora_params(model):
    """Extract DoRA parameters from model."""
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    
    params = {}
    modules = [
        "clip_model.visual.transformer.resblocks.22.attn.out_proj",
        "clip_model.visual.transformer.resblocks.23.attn.out_proj",
        "clip_model.transformer.resblocks.11.attn.out_proj",
    ]
    
    for module_path in modules:
        module = model
        for attr in module_path.split("."):
            module = getattr(module, attr)
        
        params[f'{module_path}.m'] = module.m.detach().cpu().clone()
        params[f'{module_path}.delta_D_A'] = module.delta_D_A.detach().cpu().clone()
        params[f'{module_path}.delta_D_B'] = module.delta_D_B.detach().cpu().clone()
    
    return params


def compare_params(params1, params2, tolerance=1e-8):
    """Compare two parameter dictionaries."""
    max_diff = 0
    all_match = True
    details = {}
    
    for key in params1.keys():
        if key not in params2:
            print(f"  ✗ {key}: MISSING in params2")
            all_match = False
            continue
            
        diff = torch.max(torch.abs(params1[key] - params2[key])).item()
        max_diff = max(max_diff, diff)
        
        if diff > tolerance:
            all_match = False
            status = "✗"
        else:
            status = "✓"
        
        details[key] = diff
        print(f"  {status} {key}: max diff = {diff:.2e}")
    
    return all_match, max_diff, details


def load_checkpoint_file(checkpoint_path):
    """Load DoRA parameters from a checkpoint file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint


def test_existing_checkpoint(checkpoint_dir, test_epoch, config):
    """
    Test that loading checkpoint N-1 and training 1 epoch reproduces checkpoint N.
    
    NOTE: Checkpoints are 1-indexed. epoch1_dora_params.pth is saved AFTER training epoch 1.
    
    Args:
        checkpoint_dir: Directory containing epoch checkpoints (1-indexed)
        test_epoch: The target epoch to test (1-indexed, e.g., 3 means epoch3_dora_params.pth)
                    Will load epoch{test_epoch-1} and compare with epoch{test_epoch}
        config: Training configuration
    """
    print("\n" + "="*80)
    print(f"TESTING EXISTING CHECKPOINTS - EPOCH {test_epoch}")
    print("="*80)
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Test: Load epoch{test_epoch-1}_dora_params.pth → Train 1 epoch → Compare with epoch{test_epoch}_dora_params.pth")
    print("="*80 + "\n")
    
    # Checkpoint paths (1-indexed naming convention)
    # epoch1_dora_params.pth = saved AFTER training epoch 1
    # epoch2_dora_params.pth = saved AFTER training epoch 2, etc.
    checkpoint_load = os.path.join(checkpoint_dir, f"epoch{test_epoch-1}_dora_params.pth")
    checkpoint_target = os.path.join(checkpoint_dir, f"epoch{test_epoch}_dora_params.pth")
    
    # Verify checkpoints exist
    if not os.path.exists(checkpoint_load):
        print(f"❌ ERROR: Checkpoint to load not found: {checkpoint_load}")
        return False
    
    if not os.path.exists(checkpoint_target):
        print(f"❌ ERROR: Target checkpoint not found: {checkpoint_target}")
        return False
    
    print(f"✓ Found checkpoint to load: epoch{test_epoch-1}_dora_params.pth")
    print(f"✓ Found target checkpoint: epoch{test_epoch}_dora_params.pth\n")
    
    # Load target checkpoint (what we're trying to reproduce)
    print(f"Loading target checkpoint (epoch{test_epoch}_dora_params.pth)...")
    target_params = load_checkpoint_file(checkpoint_target)
    print(f"✓ Loaded {len(target_params)} parameter tensors\n")
    
    # Set device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # ===================================================================
    # Initialize model and load checkpoint from epoch N-1
    # ===================================================================
    print("="*80)
    print(f"STEP 1: Initialize model and load epoch{test_epoch-1}_dora_params.pth")
    print("="*80)
    
    seed_everything(config['random_seed'])
    
    model = CLIPHBA(classnames=classnames66, backbone_name=config['backbone'], pos_embedding=True)
    apply_dora_to_ViT(model, 
                      n_vision_layers=config['vision_layers'],
                      n_transformer_layers=config['transformer_layers'],
                      r=config['rank'],
                      dora_dropout=0.1)
    switch_dora_layers(model, freeze_all=True, dora_state=True)
    model.to(device)
    
    # Prepare data BEFORE loading checkpoint and RNG state
    # This is critical! The RNG reconstruction simulates the random_split
    dataset = ThingsDataset(csv_file=config['csv_file'], img_dir=config['img_dir'])
    train_size = int(config['train_portion'] * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, _ = random_split(dataset, [train_size, test_size])
    
    # Load checkpoint
    checkpoint_data = load_checkpoint_file(checkpoint_load)
    model.load_state_dict(checkpoint_data, strict=False)
    print(f"✓ Loaded DoRA parameters from epoch{test_epoch-1}_dora_params.pth")
    
    # Try to restore RNG states for reproducible shuffle order
    # IMPORTANT: This must happen AFTER random_split!
    rng_restored = load_and_restore_rng_state(checkpoint_data)
    
    if not rng_restored:
        # Checkpoint doesn't have RNG states - try to reconstruct them!
        print("⚠️  No RNG states in checkpoint (old checkpoint)")
        print("🔧 Attempting to reconstruct RNG states from training parameters...")
        
        # Reconstruct the RNG state for this epoch
        # Note: We already did random_split above
        reconstructed_rng = reconstruct_rng_state_for_epoch(
            target_epoch=test_epoch,
            dataset_size=len(dataset),
            train_portion=config['train_portion'],
            batch_size=config['batch_size'],
            seed=config['random_seed']
        )
        
        # Restore the reconstructed RNG state
        restore_rng_state_dict(reconstructed_rng)
        print("✅ Successfully reconstructed and restored RNG states!")
    
    # Verify loaded parameters match checkpoint file
    loaded_params = extract_dora_params(model)
    print("\nVerifying loaded parameters match checkpoint file:")
    load_match, load_diff, _ = compare_params(checkpoint_data, loaded_params, tolerance=1e-12)
    
    if load_match:
        print(f"✓ Parameters loaded correctly (max diff: {load_diff:.2e})\n")
    else:
        print(f"✗ WARNING: Parameters differ after loading (max diff: {load_diff:.2e})\n")
    
    # ===================================================================
    # Train for 1 epoch
    # ===================================================================
    print("="*80)
    print(f"STEP 2: Train for 1 epoch (to match epoch{test_epoch}_dora_params.pth)")
    print("="*80)
    
    # NOW create DataLoader - the RNG state is correct at this point!
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    criterion = nn.MSELoss()
    
    # Train for specified number of steps (or full epoch)
    model.train()
    num_steps = config.get('num_training_steps', None)
    if num_steps is None:
        num_steps = len(train_loader)
    
    print(f"Training for {num_steps} batches...")
    total_loss = 0.0
    step_count = 0
    
    for batch_idx, (_, images, targets) in enumerate(train_loader):
        if step_count >= num_steps:
            break
        
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        step_count += 1
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx+1}/{num_steps}, Loss: {loss.item():.6f}")
    
    avg_loss = total_loss / step_count
    print(f"\n✓ Training complete. Average loss: {avg_loss:.6f}\n")
    
    # Extract parameters after training
    trained_params = extract_dora_params(model)
    
    # ===================================================================
    # Compare with target checkpoint
    # ===================================================================
    print("="*80)
    print(f"STEP 3: Compare with target checkpoint (epoch{test_epoch}_dora_params.pth)")
    print("="*80)
    print("")
    
    final_match, final_diff, details = compare_params(target_params, trained_params, tolerance=1e-8)
    
    # ===================================================================
    # Summary
    # ===================================================================
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    if final_match:
        print("✅ TEST PASSED: Checkpoint loading perfectly reproduces training!")
        print(f"   Maximum parameter difference: {final_diff:.2e}")
        print("   All parameters within tolerance (1e-8)")
    else:
        print("❌ TEST FAILED: Checkpoint loading does not reproduce training")
        print(f"   Maximum parameter difference: {final_diff:.2e}")
        print("\n   Parameters that differ significantly:")
        for key, diff in details.items():
            if diff > 1e-8:
                print(f"     - {key}: {diff:.2e}")
    
    print("="*80 + "\n")
    
    return final_match


def test_multiple_epochs(checkpoint_dir, test_epochs, config):
    """
    Test multiple epochs to ensure consistent reproducibility.
    
    Args:
        checkpoint_dir: Directory containing epoch checkpoints
        test_epochs: List of epochs to test
        config: Training configuration
    """
    print("\n" + "="*80)
    print("TESTING MULTIPLE EPOCHS")
    print("="*80)
    print(f"Testing epochs: {test_epochs}")
    print("="*80 + "\n")
    
    results = {}
    
    for epoch in test_epochs:
        result = test_existing_checkpoint(checkpoint_dir, epoch, config)
        results[epoch] = result
        
        if not result:
            print(f"\n⚠️  Epoch {epoch} failed. Continuing to next epoch...\n")
    
    # Overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    
    passed = sum(1 for r in results.values() if r)
    failed = len(results) - passed
    
    print(f"Total epochs tested: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    print("\nResults by epoch:")
    for epoch, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  Epoch {epoch}: {status}")
    
    print("="*80 + "\n")
    
    return all(results.values())


def main():
    """Run the checkpoint reproducibility test with existing checkpoints."""
    
    # Configuration
    config = {
        'csv_file': '../Data/spose_embedding66d_rescaled_1806train.csv',
        'img_dir': '../Data/Things1854',
        'backbone': 'ViT-L/14',
        'batch_size': 64,
        'train_portion': 0.8,
        'lr': 3e-4,
        'random_seed': 1,
        'vision_layers': 2,
        'transformer_layers': 1,
        'rank': 32,
        'num_training_steps': None,  # Use subset for faster testing (or None for full epoch)
    }
    
    # Your checkpoint directory
    checkpoint_dir = '/home/wallacelab/teba/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior/training_artifacts/dora_params/dora_params_20251008_211424'
    
    # Verify directory exists
    if not os.path.exists(checkpoint_dir):
        print(f"❌ ERROR: Checkpoint directory not found: {checkpoint_dir}")
        return 1
    
    # List available checkpoints
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith('epoch') and f.endswith('.pth')])
    
    if not checkpoint_files:
        print(f"❌ ERROR: No checkpoint files found in: {checkpoint_dir}")
        return 1
    
    print(f"\nFound {len(checkpoint_files)} checkpoints in directory:")
    for f in checkpoint_files[:5]:
        print(f"  - {f}")
    if len(checkpoint_files) > 5:
        print(f"  ... and {len(checkpoint_files) - 5} more")
    print()
    
    # Test a few epochs
    # Start with epoch 3 (safe bet that training was stable by then)
    test_epochs = [3, 10, 20]  # Test early, middle, and later epochs
    
    # Filter to only test epochs that have both N-1 and N checkpoints
    available_epochs = set()
    for f in checkpoint_files:
        epoch_num = int(f.replace('epoch', '').replace('_dora_params.pth', ''))
        available_epochs.add(epoch_num)
    
    test_epochs = [e for e in test_epochs if e in available_epochs and (e-1) in available_epochs]
    
    if not test_epochs:
        print("❌ ERROR: No suitable epochs found for testing")
        print("   Need at least 2 consecutive epochs (e.g., epoch2 and epoch3)")
        return 1
    
    print(f"Will test epochs: {test_epochs}\n")
    
    # Run tests
    all_passed = test_multiple_epochs(checkpoint_dir, test_epochs, config)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

