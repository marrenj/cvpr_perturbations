#!/usr/bin/env python3
"""
Test script to verify DoRA parameter loading and training reproducibility.

This script tests that:
1. DoRA parameters can be loaded from the baseline directory
2. Loading the same parameters twice produces identical inference results
3. Training can progress from loaded checkpoints
4. Loading from epoch N-1 and training produces the same results as baseline epoch N

Test 4 is comprehensive: it loads DoRA parameters from epoch N-1, trains for one full 
epoch, and verifies both train and test losses match the baseline epoch N results.
This validates full training reproducibility from any checkpoint.
"""

from functions.cvpr_train_behavior_things_pipeline import (
    seed_everything, ThingsDataset,
    CLIPHBA, apply_dora_to_ViT, switch_dora_layers,
    evaluate_model
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
import os
import sys
import numpy as np
import pandas as pd
from functions.spose_dimensions import classnames66

class DoRAReproducibilityTest:
    def __init__(self, baseline_dora_dir, baseline_results_csv, test_epochs=None):
        """
        Initialize the reproducibility test.
        
        Args:
            baseline_dora_dir: Directory containing baseline DoRA parameters
            baseline_results_csv: Path to baseline training results CSV file
            test_epochs: List of epochs to test (e.g., [1, 10, 25, 50, 92])
        """
        self.baseline_dora_dir = baseline_dora_dir
        self.baseline_results_csv = baseline_results_csv
        self.test_epochs = test_epochs or [1, 10, 25, 50, 75, 92]
        self.results = {}
        
        # Load baseline results
        self.baseline_df = pd.read_csv(baseline_results_csv)
        print(f"Loaded baseline results: {len(self.baseline_df)} epochs")
        
        # Basic configuration for testing
        self.config = {
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
            'criterion': nn.MSELoss(),
            'cuda': 0,
        }
        
    def setup_model_and_data(self, shuffle_train=False):
        """
        Set up model and data loaders for testing.
        
        Args:
            shuffle_train: If True, shuffle training data (needed for baseline comparison test)
        """
        # Set seed for reproducibility
        seed_everything(self.config['random_seed'])
        
        # Initialize dataset
        dataset = ThingsDataset(
            csv_file=self.config['csv_file'],
            img_dir=self.config['img_dir']
        )
        
        # Split dataset
        train_size = int(self.config['train_portion'] * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        
        # Create data loaders
        # For baseline comparison, we need shuffle=True to match the original training
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=shuffle_train)
        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        # Set device
        if self.config['cuda'] == 0:
            device = torch.device("cuda:0")
        elif self.config['cuda'] == 1:
            device = torch.device("cuda:1")
        else:
            device = torch.device("cpu")
        
        return train_loader, test_loader, device
    
    def create_model(self, epoch_to_load=None):
        """
        Create model and optionally load DoRA parameters from specific epoch.
        
        Args:
            epoch_to_load: If provided, load DoRA parameters from this epoch
        """
        seed_everything(self.config['random_seed'])
        
        # Initialize model
        model = CLIPHBA(
            classnames=classnames66,
            backbone_name=self.config['backbone'],
            pos_embedding=True
        )
        
        # Apply DoRA
        apply_dora_to_ViT(
            model,
            n_vision_layers=self.config['vision_layers'],
            n_transformer_layers=self.config['transformer_layers'],
            r=self.config['rank'],
            dora_dropout=0.1
        )
        switch_dora_layers(model, freeze_all=True, dora_state=True)
        
        # Load DoRA parameters if specified
        if epoch_to_load is not None and epoch_to_load > 0:
            dora_params_path = os.path.join(
                self.baseline_dora_dir,
                f"epoch{epoch_to_load}_dora_params.pth"
            )
            if not os.path.exists(dora_params_path):
                raise FileNotFoundError(f"DoRA parameters not found: {dora_params_path}")
            
            dora_params_state_dict = torch.load(dora_params_path)
            model.load_state_dict(dora_params_state_dict, strict=False)
            print(f"    Loaded DoRA parameters from epoch {epoch_to_load}")
        
        return model
    
    def run_inference_steps(self, model, train_loader, test_loader, device, num_batches=5):
        """
        Run a few inference steps and collect predictions.
        
        Args:
            model: The model to test
            train_loader: Training data loader
            test_loader: Test data loader
            device: Device to run on
            num_batches: Number of batches to process
        """
        model.eval()
        model.to(device)
        
        train_predictions = []
        train_targets = []
        
        # Process a few training batches
        with torch.no_grad():
            for batch_idx, (_, images, targets) in enumerate(train_loader):
                if batch_idx >= num_batches:
                    break
                
                images = images.to(device)
                targets = targets.to(device)
                
                predictions = model(images)
                
                train_predictions.append(predictions.cpu().numpy())
                train_targets.append(targets.cpu().numpy())
        
        train_predictions = np.concatenate(train_predictions, axis=0)
        train_targets = np.concatenate(train_targets, axis=0)
        
        # Compute test loss
        criterion = nn.MSELoss()
        test_loss = evaluate_model(model, test_loader, device, criterion)
        
        return {
            'train_predictions': train_predictions,
            'train_targets': train_targets,
            'test_loss': test_loss,
        }
    
    def run_training_steps(self, model, train_loader, device, num_batches=5):
        """
        Run a few training steps and record losses.
        
        Args:
            model: The model to train
            train_loader: Training data loader
            device: Device to run on
            num_batches: Number of batches to train on
        """
        model.train()
        model.to(device)
        
        optimizer = AdamW(model.parameters(), lr=self.config['lr'])
        criterion = nn.MSELoss()
        
        losses = []
        
        for batch_idx, (_, images, targets) in enumerate(train_loader):
            if batch_idx >= num_batches:
                break
            
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        return losses
    
    def test_reproducibility_same_epoch(self, epoch):
        """
        Test that loading from the same epoch twice produces identical results.
        """
        print(f"\n  Test 1: Reproducibility from epoch {epoch}")
        print("  Loading DoRA parameters twice and comparing results...")
        
        train_loader, test_loader, device = self.setup_model_and_data()
        
        # First run
        model1 = self.create_model(epoch_to_load=epoch)
        results1 = self.run_inference_steps(model1, train_loader, test_loader, device)
        
        # Second run with same parameters
        model2 = self.create_model(epoch_to_load=epoch)
        results2 = self.run_inference_steps(model2, train_loader, test_loader, device)
        
        # Compare predictions
        pred_diff = np.abs(results1['train_predictions'] - results2['train_predictions']).max()
        test_loss_diff = abs(results1['test_loss'] - results2['test_loss'])
        
        passed = pred_diff < 1e-6 and test_loss_diff < 1e-6
        
        print(f"    Max prediction difference: {pred_diff:.2e}")
        print(f"    Test loss difference: {test_loss_diff:.2e}")
        print(f"    Status: {'✓ PASS' if passed else '✗ FAIL'}")
        
        return passed
    
    def test_loading_from_checkpoint(self, epoch):
        """
        Test that DoRA parameters can be loaded from checkpoint.
        """
        print(f"\n  Test 2: Loading checkpoint from epoch {epoch}")
        
        checkpoint_path = os.path.join(
            self.baseline_dora_dir,
            f"epoch{epoch}_dora_params.pth"
        )
        
        if not os.path.exists(checkpoint_path):
            print(f"    Status: ✗ FAIL - Checkpoint file not found: {checkpoint_path}")
            return False
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path)
            
            # Verify expected keys are present
            expected_keys = [
                'clip_model.visual.transformer.resblocks.22.attn.out_proj.m',
                'clip_model.visual.transformer.resblocks.22.attn.out_proj.delta_D_A',
                'clip_model.visual.transformer.resblocks.22.attn.out_proj.delta_D_B',
                'clip_model.visual.transformer.resblocks.23.attn.out_proj.m',
                'clip_model.visual.transformer.resblocks.23.attn.out_proj.delta_D_A',
                'clip_model.visual.transformer.resblocks.23.attn.out_proj.delta_D_B',
                'clip_model.transformer.resblocks.11.attn.out_proj.m',
                'clip_model.transformer.resblocks.11.attn.out_proj.delta_D_A',
                'clip_model.transformer.resblocks.11.attn.out_proj.delta_D_B',
            ]
            
            missing_keys = [key for key in expected_keys if key not in checkpoint]
            
            if missing_keys:
                print(f"    Status: ✗ FAIL - Missing keys: {missing_keys}")
                return False
            
            print(f"    Checkpoint contains {len(checkpoint)} parameters")
            print("    All expected DoRA parameters present")
            print("    Status: ✓ PASS")
            return True
            
        except Exception as e:  # pylint: disable=broad-except
            print(f"    Status: ✗ FAIL - Error loading checkpoint: {str(e)}")
            return False
    
    def test_training_progression(self, epoch):
        """
        Test that training can progress from a loaded checkpoint.
        """
        print(f"\n  Test 3: Training progression from epoch {epoch}")
        print("  Running 5 training steps...")
        
        train_loader, _, device = self.setup_model_and_data()
        
        # Create model and load checkpoint
        model = self.create_model(epoch_to_load=epoch)
        
        # Run training steps
        losses = self.run_training_steps(model, train_loader, device, num_batches=5)
        
        # Check that losses are reasonable (not NaN, not exploding)
        all_valid = all(not np.isnan(loss) and not np.isinf(loss) for loss in losses)
        losses_decreasing = losses[-1] <= losses[0] * 1.5  # Allow some increase
        
        passed = all_valid and losses_decreasing
        
        print(f"    Training losses: {[f'{l:.4f}' for l in losses]}")
        print(f"    All losses valid: {all_valid}")
        print(f"    Losses stable/decreasing: {losses_decreasing}")
        print(f"    Status: {'✓ PASS' if passed else '✗ FAIL'}")
        
        return passed
    
    def train_one_epoch(self, model, train_loader, device, optimizer, criterion):
        """
        Train model for one full epoch and return average training loss.
        """
        model.train()
        total_loss = 0.0
        
        for _, images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
        
        avg_loss = total_loss / len(train_loader.dataset)
        return avg_loss
    
    def test_baseline_comparison(self, epoch):
        """
        Test that loading from epoch N-1, training for one epoch, and evaluating 
        produces results matching baseline epoch N.
        
        This verifies full training reproducibility:
        - Load DoRA parameters from epoch N-1
        - Train for one full epoch
        - Compare train and test losses to baseline epoch N
        """
        print(f"\n  Test 4: Baseline comparison for epoch {epoch}")
        
        # Check if baseline data exists for this epoch
        baseline_row = self.baseline_df[self.baseline_df['epoch'] == epoch]
        if baseline_row.empty:
            print(f"    Status: ✗ SKIP - No baseline data for epoch {epoch}")
            return None
        
        baseline_row = baseline_row.iloc[0]
        baseline_train_loss = baseline_row['train_loss']
        baseline_test_loss = baseline_row['test_loss']
        
        print(f"    Baseline epoch {epoch} - Train loss: {baseline_train_loss:.4f}, Test loss: {baseline_test_loss:.4f}")
        
        # Use shuffle=True for training to match the original training pipeline
        train_loader, test_loader, device = self.setup_model_and_data(shuffle_train=True)
        
        # Create model and load checkpoint from epoch before (N-1)
        # This matches the training pipeline where epoch N starts from the state saved at epoch N-1
        epoch_to_load = epoch - 1 if epoch > 1 else None
        if epoch_to_load is not None:
            print(f"    Loading DoRA parameters from epoch {epoch_to_load} and training epoch {epoch}...")
        else:
            print(f"    Using initial DoRA parameters and training epoch {epoch}...")
        
        model = self.create_model(epoch_to_load=epoch_to_load)
        model.to(device)
        
        # Set up optimizer and criterion
        optimizer = AdamW(model.parameters(), lr=self.config['lr'])
        criterion = nn.MSELoss()
        
        # Train for one full epoch
        train_loss = self.train_one_epoch(model, train_loader, device, optimizer, criterion)
        
        # Evaluate on test set
        test_loss = evaluate_model(model, test_loader, device, criterion)
        
        print(f"    Current epoch {epoch} - Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}")
        
        # Compare both train and test losses with tolerance
        train_loss_diff = abs(train_loss - baseline_train_loss)
        test_loss_diff = abs(test_loss - baseline_test_loss)
        
        train_relative_diff = train_loss_diff / baseline_train_loss if baseline_train_loss > 0 else train_loss_diff
        test_relative_diff = test_loss_diff / baseline_test_loss if baseline_test_loss > 0 else test_loss_diff
        
        # Allow 1% relative difference or 0.1 absolute difference (slightly relaxed for full training)
        tolerance_relative = 0.01
        tolerance_absolute = 0.1
        
        train_passed = (train_relative_diff < tolerance_relative) or (train_loss_diff < tolerance_absolute)
        test_passed = (test_relative_diff < tolerance_relative) or (test_loss_diff < tolerance_absolute)
        
        print(f"    Train loss difference: {train_loss_diff:.6f} (relative: {train_relative_diff:.4%}) - {'✓' if train_passed else '✗'}")
        print(f"    Test loss difference: {test_loss_diff:.6f} (relative: {test_relative_diff:.4%}) - {'✓' if test_passed else '✗'}")
        
        passed = train_passed and test_passed
        print(f"    Status: {'✓ PASS' if passed else '✗ FAIL'}")
        
        return passed
    
    def run_all_tests(self):
        """Run all tests for all specified epochs."""
        print("="*80)
        print("DoRA REPRODUCIBILITY TEST SUITE")
        print(f"Baseline DoRA Directory: {self.baseline_dora_dir}")
        print(f"Baseline Results CSV: {self.baseline_results_csv}")
        print(f"Test Epochs: {self.test_epochs}")
        print("\nNOTE: Test 4 trains for a full epoch per test. This may take a while.")
        print("="*80)
        
        overall_results = {}
        all_exceptions = []
        
        for epoch in self.test_epochs:
            print(f"\n{'='*80}")
            print(f"TESTING EPOCH {epoch}")
            print(f"{'='*80}")
            
            epoch_results = {
                'reproducibility': False,
                'checkpoint_loading': False,
                'training_progression': False,
                'baseline_comparison': False,
            }
            
            try:
                epoch_results['reproducibility'] = self.test_reproducibility_same_epoch(epoch)
                epoch_results['checkpoint_loading'] = self.test_loading_from_checkpoint(epoch)
                epoch_results['training_progression'] = self.test_training_progression(epoch)
                baseline_result = self.test_baseline_comparison(epoch)
                if baseline_result is not None:  # None means skipped
                    epoch_results['baseline_comparison'] = baseline_result
                else:
                    # Don't count skipped tests as failures
                    epoch_results.pop('baseline_comparison')
                
                all_passed = all(epoch_results.values())
                overall_results[epoch] = all_passed
                
                print(f"\n  Overall for epoch {epoch}: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
                
            except Exception as e:  # pylint: disable=broad-except
                print(f"\n  ✗ ERROR during testing epoch {epoch}: {str(e)}")
                import traceback
                traceback.print_exc()
                overall_results[epoch] = False
                all_exceptions.append((epoch, str(e)))
        
        # Print summary
        print(f"\n{'='*80}")
        print("TEST SUMMARY")
        print(f"{'='*80}")
        
        for epoch, passed in overall_results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  Epoch {epoch:3d}: {status}")
        
        total_tests = len(overall_results)
        passed_tests = sum(overall_results.values())
        
        print(f"\n  Total: {passed_tests}/{total_tests} epochs passed all tests")
        
        if passed_tests == total_tests:
            print("\n  ✓✓✓ ALL TESTS PASSED ✓✓✓")
            print("  The DoRA loading pipeline is working correctly!")
        else:
            print("\n  ✗✗✗ SOME TESTS FAILED ✗✗✗")
            print("  Please check the failed epochs above.")
        
        print(f"{'='*80}\n")
        
        return overall_results


def main():
    """Main function to run the test suite."""
    
    # Configuration
    baseline_dora_dir = '/home/wallacelab/teba/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior/training_artifacts/dora_params/dora_params_20251008_211424'
    baseline_results_csv = '/home/wallacelab/teba/multimodal_brain_inspired/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior/training_results/training_res_20251008_211424.csv'
    
    # Test a selection of epochs throughout training
    # You can modify this list to test specific epochs
    test_epochs = [1, 10, 25, 50, 75, 92]
    
    # Check if baseline directory exists
    if not os.path.exists(baseline_dora_dir):
        print(f"ERROR: Baseline DoRA directory not found: {baseline_dora_dir}")
        print("Please update the baseline_dora_dir path in the script.")
        sys.exit(1)
    
    # Check if baseline CSV exists
    if not os.path.exists(baseline_results_csv):
        print(f"ERROR: Baseline results CSV not found: {baseline_results_csv}")
        print("Please update the baseline_results_csv path in the script.")
        sys.exit(1)
    
    # Run tests
    test_suite = DoRAReproducibilityTest(
        baseline_dora_dir=baseline_dora_dir,
        baseline_results_csv=baseline_results_csv,
        test_epochs=test_epochs
    )
    
    results = test_suite.run_all_tests()
    
    # Exit with appropriate code
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()

