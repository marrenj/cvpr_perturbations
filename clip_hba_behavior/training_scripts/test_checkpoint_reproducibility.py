"""
Test to verify that loading DoRA parameters from a checkpoint can perfectly reproduce 
training results from any given epoch.

This test ensures that:
1. Loading DoRA parameters from epoch N gives identical results as continuous training to epoch N
2. Model outputs are deterministic after loading checkpoints
3. Training can be resumed from any epoch without loss of reproducibility
"""

import torch
import torch.nn as nn
import os
import sys
import tempfile
import shutil

# Add parent directory to path to import necessary modules
sys.path.append('../')
from functions.cvpr_train_behavior_things_pipeline import (
    seed_everything,
    save_dora_parameters,
    CLIPHBA,
    apply_dora_to_ViT,
    switch_dora_layers,
    ThingsDataset
)
from functions.spose_dimensions import classnames66
from torch.utils.data import DataLoader, random_split


class CheckpointReproducibilityTest:
    """Test class to verify checkpoint loading reproduces results perfectly."""
    
    def __init__(self, test_config):
        self.test_config = test_config
        self.temp_dir = None
        self.continuous_dir = None
        self.checkpoint_dir = None
        
    def setup_temp_directories(self):
        """Create temporary directories for test artifacts."""
        self.temp_dir = tempfile.mkdtemp(prefix='checkpoint_test_')
        
        # Create subdirectories for different test runs
        self.continuous_dir = os.path.join(self.temp_dir, 'continuous_run')
        self.checkpoint_dir = os.path.join(self.temp_dir, 'checkpoint_run')
        
        os.makedirs(self.continuous_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        return self.temp_dir
    
    def cleanup(self):
        """Remove temporary directories."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def extract_dora_parameters(self, model):
        """
        Extract all DoRA parameters from a model.
        
        Returns:
            dict: Dictionary containing all DoRA parameters
        """
        dora_params = {}
        
        # Handle DataParallel wrapper
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        
        # Define the modules that have DoRA layers
        modules_to_extract = [
            ("clip_model.visual.transformer.resblocks.22.attn.out_proj", "visual_resblock_22_attn"),
            ("clip_model.visual.transformer.resblocks.23.attn.out_proj", "visual_resblock_23_attn"),
            ("clip_model.transformer.resblocks.11.attn.out_proj", "transformer_resblock_11_attn"),
        ]
        
        for module_path, _ in modules_to_extract:
            # Traverse the model to get the module
            module = model
            for attr in module_path.split("."):
                module = getattr(module, attr)
            
            # Extract DoRA parameters
            dora_params[f'{module_path}.m'] = module.m.detach().cpu().clone()
            dora_params[f'{module_path}.delta_D_A'] = module.delta_D_A.detach().cpu().clone()
            dora_params[f'{module_path}.delta_D_B'] = module.delta_D_B.detach().cpu().clone()
            dora_params[f'{module_path}.D'] = module.D.detach().cpu().clone()
        
        return dora_params
    
    def compare_dora_parameters(self, params1, params2, tolerance=1e-10):
        """
        Compare two sets of DoRA parameters.
        
        Args:
            params1: First set of parameters
            params2: Second set of parameters
            tolerance: Numerical tolerance for comparison
            
        Returns:
            tuple: (bool, dict) - (are_equal, differences)
        """
        differences = {}
        all_equal = True
        
        for key in params1.keys():
            if key not in params2:
                differences[key] = "Missing in params2"
                all_equal = False
                continue
            
            # Compare tensors
            param1 = params1[key]
            param2 = params2[key]
            
            # Check shapes match
            if param1.shape != param2.shape:
                differences[key] = f"Shape mismatch: {param1.shape} vs {param2.shape}"
                all_equal = False
                continue
            
            # Check values match
            max_diff = torch.max(torch.abs(param1 - param2)).item()
            if max_diff > tolerance:
                differences[key] = f"Max difference: {max_diff:.2e} (tolerance: {tolerance:.2e})"
                all_equal = False
            else:
                differences[key] = f"Match (max diff: {max_diff:.2e})"
        
        return all_equal, differences
    
    def compare_model_outputs(self, model1, model2, test_input, tolerance=1e-10):
        """
        Compare outputs of two models on the same input.
        
        Args:
            model1: First model
            model2: Second model
            test_input: Input tensor
            tolerance: Numerical tolerance for comparison
            
        Returns:
            tuple: (bool, float) - (are_equal, max_difference)
        """
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            output1 = model1(test_input)
            output2 = model2(test_input)
        
        max_diff = torch.max(torch.abs(output1 - output2)).item()
        are_equal = max_diff <= tolerance
        
        return are_equal, max_diff
    
    def initialize_model(self, config, device):
        """Initialize a model with the given configuration."""
        # Determine pos_embedding based on backbone
        pos_embedding = False if config['backbone'] == 'RN50' else True
        
        # Initialize model
        model = CLIPHBA(classnames=classnames66, backbone_name=config['backbone'], 
                        pos_embedding=pos_embedding)
        
        # Apply DoRA
        apply_dora_to_ViT(model, 
                          n_vision_layers=config['vision_layers'],
                          n_transformer_layers=config['transformer_layers'],
                          r=config['rank'],
                          dora_dropout=0.1)
        switch_dora_layers(model, freeze_all=True, dora_state=True)
        
        model.to(device)
        
        return model
    
    def test_single_epoch_reproducibility(self, test_epoch=3, num_training_steps=10):
        """
        Test that loading from a checkpoint at epoch N gives identical results 
        to continuous training through epoch N.
        
        Args:
            test_epoch: The epoch to test (will train from 0 to test_epoch)
            num_training_steps: Number of batches to train for (for faster testing)
            
        Returns:
            dict: Test results with detailed comparison
        """
        print("\n" + "="*80)
        print(f"TESTING CHECKPOINT REPRODUCIBILITY AT EPOCH {test_epoch}")
        print("="*80)
        
        self.setup_temp_directories()
        
        # Set random seed for reproducibility
        seed_everything(self.test_config['random_seed'])
        
        # Set device
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {device}")
        
        # ===================================================================
        # PART 1: Continuous training from epoch 0 to test_epoch
        # ===================================================================
        print(f"\n{'='*80}")
        print(f"PART 1: Continuous training from epoch 0 to {test_epoch}")
        print(f"{'='*80}")
        
        seed_everything(self.test_config['random_seed'])
        
        # Initialize model
        continuous_model = self.initialize_model(self.test_config, device)
        
        # Create simple training data for testing
        train_dataset = ThingsDataset(
            csv_file=self.test_config['csv_file'],
            img_dir=self.test_config['img_dir']
        )
        train_size = int(0.8 * len(train_dataset))
        test_size = len(train_dataset) - train_size
        train_dataset, _ = random_split(train_dataset, [train_size, test_size])
        
        train_loader = DataLoader(train_dataset, batch_size=self.test_config['batch_size'], shuffle=True)
        
        # Initialize optimizer
        optimizer_continuous = torch.optim.AdamW(continuous_model.parameters(), lr=self.test_config['lr'])
        criterion = nn.MSELoss()
        
        # Store DoRA params at each epoch
        continuous_params_by_epoch = {}
        
        # Train for test_epoch epochs
        for epoch in range(test_epoch):
            continuous_model.train()
            print(f"\n--- Epoch {epoch+1}/{test_epoch} (Continuous) ---")
            
            step_count = 0
            for batch_idx, (_, images, targets) in enumerate(train_loader):
                if step_count >= num_training_steps:
                    break
                
                images = images.to(device)
                targets = targets.to(device)
                
                optimizer_continuous.zero_grad()
                predictions = continuous_model(images)
                loss = criterion(predictions, targets)
                loss.backward()
                optimizer_continuous.step()
                
                step_count += 1
                
                if batch_idx % 5 == 0:
                    print(f"  Step {step_count}/{num_training_steps}, Loss: {loss.item():.6f}")
            
            # Save DoRA parameters after this epoch
            epoch_params = self.extract_dora_parameters(continuous_model)
            continuous_params_by_epoch[epoch] = epoch_params
            
            # Also save to file (simulating real checkpoint saving)
            continuous_dora_path = os.path.join(self.continuous_dir, 'dora_params')
            os.makedirs(continuous_dora_path, exist_ok=True)
            save_dora_parameters(continuous_model, continuous_dora_path, epoch)
            print(f"  Saved DoRA params for epoch {epoch+1}")
        
        # Extract final parameters after continuous training
        final_continuous_params = self.extract_dora_parameters(continuous_model)
        
        # ===================================================================
        # PART 2: Train to checkpoint epoch, save, then resume
        # ===================================================================
        checkpoint_epoch = test_epoch - 1  # Train up to this epoch, then resume
        
        print(f"\n{'='*80}")
        print(f"PART 2: Training with checkpoint at epoch {checkpoint_epoch}, then resume to {test_epoch}")
        print(f"{'='*80}")
        
        # Reset random seed to ensure same initialization
        seed_everything(self.test_config['random_seed'])
        
        # Initialize model with same seed
        checkpoint_model = self.initialize_model(self.test_config, device)
        
        # Verify initial models are identical
        initial_checkpoint_params = self.extract_dora_parameters(checkpoint_model)
        initial_continuous_params = continuous_params_by_epoch[0] if 0 in continuous_params_by_epoch else self.extract_dora_parameters(continuous_model)
        
        # Actually we need to extract from continuous model before training
        seed_everything(self.test_config['random_seed'])
        temp_model = self.initialize_model(self.test_config, device)
        initial_continuous_params = self.extract_dora_parameters(temp_model)
        del temp_model
        
        print("\nVerifying initial models are identical...")
        initial_match, initial_diffs = self.compare_dora_parameters(
            initial_checkpoint_params, initial_continuous_params, tolerance=1e-12
        )
        if initial_match:
            print("✓ Initial models are identical")
        else:
            print("✗ Initial models differ!")
            for key, diff in initial_diffs.items():
                print(f"  {key}: {diff}")
        
        # Recreate data loader with same seed
        seed_everything(self.test_config['random_seed'])
        train_dataset = ThingsDataset(
            csv_file=self.test_config['csv_file'],
            img_dir=self.test_config['img_dir']
        )
        train_size = int(0.8 * len(train_dataset))
        test_size = len(train_dataset) - train_size
        train_dataset, _ = random_split(train_dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=self.test_config['batch_size'], shuffle=True)
        
        optimizer_checkpoint = torch.optim.AdamW(checkpoint_model.parameters(), lr=self.test_config['lr'])
        
        # Step 2a: Train to checkpoint_epoch
        print(f"\nStep 2a: Training to checkpoint epoch {checkpoint_epoch}...")
        for epoch in range(checkpoint_epoch):
            checkpoint_model.train()
            print(f"\n--- Epoch {epoch+1}/{checkpoint_epoch} (Pre-checkpoint) ---")
            
            step_count = 0
            for batch_idx, (_, images, targets) in enumerate(train_loader):
                if step_count >= num_training_steps:
                    break
                
                images = images.to(device)
                targets = targets.to(device)
                
                optimizer_checkpoint.zero_grad()
                predictions = checkpoint_model(images)
                loss = criterion(predictions, targets)
                loss.backward()
                optimizer_checkpoint.step()
                
                step_count += 1
                
                if batch_idx % 5 == 0:
                    print(f"  Step {step_count}/{num_training_steps}, Loss: {loss.item():.6f}")
        
        # Save checkpoint
        checkpoint_dora_path = os.path.join(self.checkpoint_dir, 'dora_params')
        os.makedirs(checkpoint_dora_path, exist_ok=True)
        save_dora_parameters(checkpoint_model, checkpoint_dora_path, checkpoint_epoch - 1)
        checkpoint_file = os.path.join(checkpoint_dora_path, f"epoch{checkpoint_epoch}_dora_params.pth")
        print(f"\nSaved checkpoint: {checkpoint_file}")
        
        # Extract parameters at checkpoint
        checkpoint_params_at_checkpoint = self.extract_dora_parameters(checkpoint_model)
        
        # Compare with continuous training at same epoch
        print(f"\nComparing parameters at epoch {checkpoint_epoch}...")
        checkpoint_match, checkpoint_diffs = self.compare_dora_parameters(
            checkpoint_params_at_checkpoint,
            continuous_params_by_epoch[checkpoint_epoch - 1],
            tolerance=1e-10
        )
        
        if checkpoint_match:
            print(f"✓ Parameters at epoch {checkpoint_epoch} match!")
        else:
            print(f"✗ Parameters at epoch {checkpoint_epoch} differ!")
            for key, diff in checkpoint_diffs.items():
                print(f"  {key}: {diff}")
        
        # Step 2b: Create new model and load from checkpoint
        print(f"\nStep 2b: Loading fresh model from checkpoint and resuming to epoch {test_epoch}...")
        
        seed_everything(self.test_config['random_seed'])
        resumed_model = self.initialize_model(self.test_config, device)
        
        # Load DoRA parameters from checkpoint
        dora_checkpoint_state = torch.load(checkpoint_file)
        resumed_model.load_state_dict(dora_checkpoint_state, strict=False)
        print(f"Loaded DoRA parameters from checkpoint")
        
        # Verify loaded parameters match the checkpoint
        resumed_params_after_load = self.extract_dora_parameters(resumed_model)
        load_match, load_diffs = self.compare_dora_parameters(
            resumed_params_after_load,
            checkpoint_params_at_checkpoint,
            tolerance=1e-12
        )
        
        if load_match:
            print("✓ Loaded parameters match checkpoint exactly")
        else:
            print("✗ Loaded parameters differ from checkpoint!")
            for key, diff in load_diffs.items():
                print(f"  {key}: {diff}")
        
        # Recreate optimizer and data loader with appropriate random state
        # We need to advance the random state to where it would be after checkpoint_epoch
        seed_everything(self.test_config['random_seed'])
        train_dataset = ThingsDataset(
            csv_file=self.test_config['csv_file'],
            img_dir=self.test_config['img_dir']
        )
        train_size = int(0.8 * len(train_dataset))
        test_size = len(train_dataset) - train_size
        train_dataset, _ = random_split(train_dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=self.test_config['batch_size'], shuffle=True)
        
        # Skip through data to match random state
        for epoch in range(checkpoint_epoch):
            step_count = 0
            for batch_idx, (_, images, targets) in enumerate(train_loader):
                if step_count >= num_training_steps:
                    break
                step_count += 1
        
        optimizer_resumed = torch.optim.AdamW(resumed_model.parameters(), lr=self.test_config['lr'])
        
        # Continue training for remaining epochs
        for epoch in range(checkpoint_epoch, test_epoch):
            resumed_model.train()
            print(f"\n--- Epoch {epoch+1}/{test_epoch} (Resumed) ---")
            
            step_count = 0
            for batch_idx, (_, images, targets) in enumerate(train_loader):
                if step_count >= num_training_steps:
                    break
                
                images = images.to(device)
                targets = targets.to(device)
                
                optimizer_resumed.zero_grad()
                predictions = resumed_model(images)
                loss = criterion(predictions, targets)
                loss.backward()
                optimizer_resumed.step()
                
                step_count += 1
                
                if batch_idx % 5 == 0:
                    print(f"  Step {step_count}/{num_training_steps}, Loss: {loss.item():.6f}")
        
        # Extract final parameters after resumed training
        final_resumed_params = self.extract_dora_parameters(resumed_model)
        
        # ===================================================================
        # PART 3: Compare final results
        # ===================================================================
        print(f"\n{'='*80}")
        print(f"PART 3: Comparing final results at epoch {test_epoch}")
        print(f"{'='*80}")
        
        final_match, final_diffs = self.compare_dora_parameters(
            final_continuous_params,
            final_resumed_params,
            tolerance=1e-8
        )
        
        print("\nFINAL COMPARISON:")
        print("-" * 80)
        for key, diff in final_diffs.items():
            status = "✓" if "Match" in diff else "✗"
            print(f"{status} {key}: {diff}")
        
        # Test model outputs on same input
        print("\nTesting model outputs on identical input...")
        test_input = torch.randn(4, 3, 224, 224).to(device)
        output_match, output_diff = self.compare_model_outputs(
            continuous_model, resumed_model, test_input, tolerance=1e-6
        )
        
        if output_match:
            print(f"✓ Model outputs match (max diff: {output_diff:.2e})")
        else:
            print(f"✗ Model outputs differ (max diff: {output_diff:.2e})")
        
        # ===================================================================
        # Summary
        # ===================================================================
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        test_passed = final_match and output_match
        
        if test_passed:
            print("✓ TEST PASSED: Checkpoint loading perfectly reproduces results!")
        else:
            print("✗ TEST FAILED: Checkpoint loading does not reproduce results perfectly")
            print("\nIssues found:")
            if not final_match:
                print("  - DoRA parameters differ after resumed training")
            if not output_match:
                print("  - Model outputs differ after resumed training")
        
        print("="*80 + "\n")
        
        # Cleanup
        self.cleanup()
        
        return {
            'passed': test_passed,
            'final_params_match': final_match,
            'final_output_match': output_match,
            'final_output_diff': output_diff,
            'parameter_differences': final_diffs
        }


def main():
    """Run the checkpoint reproducibility test."""
    
    # Configuration for testing (using a subset of data and fewer epochs)
    test_config = {
        'csv_file': '../Data/spose_embedding66d_rescaled_1806train.csv',
        'img_dir': '../Data/Things1854',
        'backbone': 'ViT-L/14',
        'batch_size': 32,
        'lr': 3e-4,
        'random_seed': 1,
        'vision_layers': 2,
        'transformer_layers': 1,
        'rank': 32,
    }
    
    # Create test instance
    test = CheckpointReproducibilityTest(test_config)
    
    # Run test for epoch 3 (train from 0->1->2->3 vs 0->1->2, save, load, 2->3)
    results = test.test_single_epoch_reproducibility(test_epoch=3, num_training_steps=10)
    
    # Return results
    if results['passed']:
        print("\n🎉 All tests passed! Checkpoint loading is reproducible.")
        return 0
    else:
        print("\n❌ Tests failed! Check the output above for details.")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

