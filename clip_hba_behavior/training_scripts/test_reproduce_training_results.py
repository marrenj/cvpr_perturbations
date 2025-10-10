"""
Test that we can reproduce the actual training results from the CSV file.

This test verifies that:
1. Loading checkpoint from epoch N-1
2. Training for 1 epoch
3. Produces the SAME metrics (train loss, test loss, behavioral RSA) as recorded in the CSV for epoch N

This is the most comprehensive reproducibility test since it validates the entire training process,
not just the final parameters.
"""

import torch
import torch.nn as nn
import os
import sys
import pandas as pd

sys.path.append('../')
from functions.cvpr_train_behavior_things_pipeline import (
    seed_everything,
    CLIPHBA,
    apply_dora_to_ViT,
    switch_dora_layers,
    ThingsDataset,
    ThingsInferenceDataset,
    evaluate_model,
    behavioral_RSA,
    load_and_restore_rng_state,
    reconstruct_rng_state_for_epoch,
    restore_rng_state_dict,
    load_optimizer_state
)
from functions.spose_dimensions import classnames66
from torch.utils.data import DataLoader, random_split


def load_training_results(csv_path):
    """Load the training results CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training results CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded training results: {len(df)} epochs")
    print(f"  Columns: {list(df.columns)}")
    return df


def test_reproduce_epoch(checkpoint_dir, training_results_df, test_epoch, config):
    """
    Test that we can reproduce the training results for a specific epoch.
    
    Args:
        checkpoint_dir: Directory containing epoch checkpoints
        training_results_df: DataFrame with training results
        test_epoch: The epoch to test (1-indexed)
        config: Training configuration
    """
    print("\n" + "="*80)
    print(f"TESTING REPRODUCIBILITY OF EPOCH {test_epoch}")
    print("="*80)
    
    # Get expected results from CSV
    epoch_results = training_results_df[training_results_df['epoch'] == test_epoch]
    
    if epoch_results.empty:
        print(f"❌ ERROR: No results found in CSV for epoch {test_epoch}")
        return False, {}
    
    expected_train_loss = epoch_results['train_loss'].values[0]
    expected_test_loss = epoch_results['test_loss'].values[0]
    expected_rsa_rho = epoch_results['behavioral_rsa_rho'].values[0]
    expected_rsa_p = epoch_results['behavioral_rsa_p_value'].values[0]
    
    print(f"\nExpected results from CSV (epoch {test_epoch}):")
    print(f"  Train Loss: {expected_train_loss:.6f}")
    print(f"  Test Loss:  {expected_test_loss:.6f}")
    print(f"  RSA Rho:    {expected_rsa_rho:.6f}")
    print(f"  RSA P-val:  {expected_rsa_p:.6e}")
    
    # Checkpoint paths (1-indexed)
    checkpoint_load = os.path.join(checkpoint_dir, f"epoch{test_epoch-1}_dora_params.pth")
    
    if not os.path.exists(checkpoint_load):
        print(f"\n❌ ERROR: Checkpoint not found: {checkpoint_load}")
        return False, {}
    
    print(f"\n✓ Found checkpoint: epoch{test_epoch-1}_dora_params.pth")
    
    # Set device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}\n")
    
    # ===================================================================
    # Initialize model and load checkpoint
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
    
    # ===================================================================
    # Prepare data loaders (BEFORE loading checkpoint)
    # ===================================================================
    print("="*80)
    print("STEP 2: Prepare data loaders")
    print("="*80)
    
    # Training data - do random_split BEFORE restoring RNG state
    # This is important! The RNG reconstruction already simulates the split
    dataset = ThingsDataset(csv_file=config['csv_file'], img_dir=config['img_dir'])
    train_size = int(config['train_portion'] * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Test samples:  {len(test_dataset)}\n")
    
    # Load checkpoint (skip if epoch 1, as it starts from scratch)
    if test_epoch > 1:
        checkpoint_data = torch.load(checkpoint_load, map_location='cpu')
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
            # Note: We already did random_split above, so reconstruction
            # simulates that plus the previous epochs' shuffles
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
            print("   Training should now match original results.\n")
    else:
        print("✓ Starting from scratch (epoch 1 has no previous checkpoint)\n")
    
    # Create optimizer BEFORE loading optimizer state
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    
    # Load optimizer state if available (must happen AFTER creating optimizer, AFTER RNG restore)
    if test_epoch > 1:
        load_optimizer_state(checkpoint_data, optimizer)
    
    # NOW create DataLoaders - the RNG state is correct at this point!
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Inference data for behavioral RSA
    inference_dataset = ThingsInferenceDataset(
        inference_csv_file=config['inference_csv_file'],
        img_dir=config['img_dir'],
        RDM48_triplet_dir=config['RDM48_triplet_dir']
    )
    inference_loader = DataLoader(inference_dataset, batch_size=config['batch_size'], shuffle=False)
    
    print(f"✓ Inference samples: {len(inference_dataset)}\n")
    
    # ===================================================================
    # Train for 1 epoch
    # ===================================================================
    print("="*80)
    print(f"STEP 3: Train for 1 epoch (epoch {test_epoch})")
    print("="*80)
    
    criterion = nn.MSELoss()
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (_, images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        num_batches += 1
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.6f}")
    
    actual_train_loss = total_loss / len(train_loader.dataset)
    print(f"\n✓ Training complete. Average train loss: {actual_train_loss:.6f}\n")
    
    # ===================================================================
    # Evaluate on test set
    # ===================================================================
    print("="*80)
    print("STEP 4: Evaluate on test set")
    print("="*80)
    
    actual_test_loss = evaluate_model(model, test_loader, device, criterion)
    print(f"✓ Test loss: {actual_test_loss:.6f}\n")
    
    # ===================================================================
    # Compute behavioral RSA
    # ===================================================================
    print("="*80)
    print("STEP 5: Compute behavioral RSA")
    print("="*80)
    
    actual_rsa_rho, actual_rsa_p, _ = behavioral_RSA(model, inference_loader, device)
    print(f"✓ RSA Rho: {actual_rsa_rho:.6f}")
    print(f"✓ RSA P-value: {actual_rsa_p:.6e}\n")
    
    # ===================================================================
    # Compare results
    # ===================================================================
    print("="*80)
    print("STEP 6: Compare with expected results")
    print("="*80)
    print("")
    
    # Calculate differences
    train_loss_diff = abs(actual_train_loss - expected_train_loss)
    test_loss_diff = abs(actual_test_loss - expected_test_loss)
    rsa_rho_diff = abs(actual_rsa_rho - expected_rsa_rho)
    rsa_p_diff = abs(actual_rsa_p - expected_rsa_p)
    
    # Relative differences
    train_loss_rel = train_loss_diff / expected_train_loss if expected_train_loss != 0 else 0
    test_loss_rel = test_loss_diff / expected_test_loss if expected_test_loss != 0 else 0
    rsa_rho_rel = rsa_rho_diff / abs(expected_rsa_rho) if expected_rsa_rho != 0 else 0
    
    print("Metric               Expected      Actual        Abs Diff      Rel Diff")
    print("-" * 80)
    print(f"Train Loss:          {expected_train_loss:12.6f}  {actual_train_loss:12.6f}  {train_loss_diff:12.2e}  {train_loss_rel:10.2%}")
    print(f"Test Loss:           {expected_test_loss:12.6f}  {actual_test_loss:12.6f}  {test_loss_diff:12.2e}  {test_loss_rel:10.2%}")
    print(f"RSA Rho:             {expected_rsa_rho:12.6f}  {actual_rsa_rho:12.6f}  {rsa_rho_diff:12.2e}  {rsa_rho_rel:10.2%}")
    print(f"RSA P-value:         {expected_rsa_p:12.6e}  {actual_rsa_p:12.6e}  {rsa_p_diff:12.2e}")
    print("")
    
    # Determine if results match within tolerance
    # Tolerances are looser than parameter matching because:
    # 1. Different batch ordering due to DataLoader shuffle
    # 2. Accumulation of numerical errors
    # 3. Non-deterministic operations in some parts of training
    
    train_loss_match = train_loss_rel < 0.01  # 1% relative tolerance
    test_loss_match = test_loss_rel < 0.01    # 1% relative tolerance
    rsa_rho_match = rsa_rho_diff < 0.01       # 0.01 absolute tolerance for correlation
    
    print("Status:")
    print(f"  Train Loss: {'✓ MATCH' if train_loss_match else '✗ DIFFER'} (tolerance: 1% relative)")
    print(f"  Test Loss:  {'✓ MATCH' if test_loss_match else '✗ DIFFER'} (tolerance: 1% relative)")
    print(f"  RSA Rho:    {'✓ MATCH' if rsa_rho_match else '✗ DIFFER'} (tolerance: 0.01 absolute)")
    print("")
    
    all_match = train_loss_match and test_loss_match and rsa_rho_match
    
    # ===================================================================
    # Summary
    # ===================================================================
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    if all_match:
        print(f"✅ EPOCH {test_epoch} PASSED: Training results are reproducible!")
        print("   All metrics within tolerance")
    else:
        print(f"❌ EPOCH {test_epoch} FAILED: Training results differ")
        if not train_loss_match:
            print(f"   - Train loss differs by {train_loss_rel:.2%}")
        if not test_loss_match:
            print(f"   - Test loss differs by {test_loss_rel:.2%}")
        if not rsa_rho_match:
            print(f"   - RSA rho differs by {rsa_rho_diff:.4f}")
    
    print("="*80 + "\n")
    
    results = {
        'epoch': test_epoch,
        'passed': all_match,
        'train_loss_expected': expected_train_loss,
        'train_loss_actual': actual_train_loss,
        'train_loss_diff': train_loss_diff,
        'test_loss_expected': expected_test_loss,
        'test_loss_actual': actual_test_loss,
        'test_loss_diff': test_loss_diff,
        'rsa_rho_expected': expected_rsa_rho,
        'rsa_rho_actual': actual_rsa_rho,
        'rsa_rho_diff': rsa_rho_diff,
    }
    
    return all_match, results


def test_multiple_epochs(checkpoint_dir, training_results_path, test_epochs, config):
    """Test multiple epochs."""
    print("\n" + "="*80)
    print("TESTING MULTIPLE EPOCHS AGAINST TRAINING RESULTS")
    print("="*80)
    print(f"Training results CSV: {training_results_path}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Testing epochs: {test_epochs}")
    print("="*80 + "\n")
    
    # Load training results
    training_results_df = load_training_results(training_results_path)
    print()
    
    results = {}
    
    for epoch in test_epochs:
        passed, epoch_results = test_reproduce_epoch(
            checkpoint_dir, training_results_df, epoch, config
        )
        results[epoch] = (passed, epoch_results)
    
    # Overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    
    passed_epochs = [e for e, (p, _) in results.items() if p]
    failed_epochs = [e for e, (p, _) in results.items() if not p]
    
    print(f"Total epochs tested: {len(results)}")
    print(f"Passed: {len(passed_epochs)}")
    print(f"Failed: {len(failed_epochs)}")
    
    print("\nResults by epoch:")
    for epoch, (passed, epoch_results) in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  Epoch {epoch}: {status}")
        if not passed and epoch_results:
            if epoch_results.get('train_loss_diff', 0) > 0:
                rel_diff = epoch_results['train_loss_diff'] / epoch_results['train_loss_expected']
                print(f"    - Train loss: {rel_diff:.2%} difference")
            if epoch_results.get('test_loss_diff', 0) > 0:
                rel_diff = epoch_results['test_loss_diff'] / epoch_results['test_loss_expected']
                print(f"    - Test loss: {rel_diff:.2%} difference")
            if epoch_results.get('rsa_rho_diff', 0) > 0:
                print(f"    - RSA rho: {epoch_results['rsa_rho_diff']:.4f} difference")
    
    print("="*80 + "\n")
    
    return all(p for p, _ in results.values())


def main():
    """Run the test."""
    
    # Configuration
    config = {
        'csv_file': '../Data/spose_embedding66d_rescaled_1806train.csv',
        'img_dir': '../Data/Things1854',
        'inference_csv_file': '../Data/spose_embedding66d_rescaled_48val_reordered.csv',
        'RDM48_triplet_dir': '../Data/RDM48_triplet.mat',
        'backbone': 'ViT-L/14',
        'batch_size': 64,
        'train_portion': 0.8,
        'lr': 3e-4,
        'random_seed': 1,
        'vision_layers': 2,
        'transformer_layers': 1,
        'rank': 32,
    }
    
    # Paths
    checkpoint_dir = '/home/wallacelab/teba/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior/training_artifacts/dora_params/dora_params_20251008_211424'
    training_results_path = '/home/wallacelab/teba/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior/training_results/training_res_20251008_211424.csv'
    
    # Verify paths exist
    if not os.path.exists(checkpoint_dir):
        print(f"❌ ERROR: Checkpoint directory not found: {checkpoint_dir}")
        return 1
    
    if not os.path.exists(training_results_path):
        print(f"❌ ERROR: Training results CSV not found: {training_results_path}")
        return 1
    
    # Test a few epochs (early, middle, later)
    test_epochs = [3, 10, 20]
    
    # Run tests
    all_passed = test_multiple_epochs(checkpoint_dir, training_results_path, test_epochs, config)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

