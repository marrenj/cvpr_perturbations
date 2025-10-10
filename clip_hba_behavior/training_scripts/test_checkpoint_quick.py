"""
Quick test for checkpoint reproducibility.

This is a faster version of the full test that runs with minimal data and epochs.
Use this for rapid iteration and debugging.
"""

import torch
import torch.nn as nn
import os
import sys
import tempfile
import shutil

sys.path.append('../')
from functions.cvpr_train_behavior_things_pipeline import (
    seed_everything,
    CLIPHBA,
    apply_dora_to_ViT,
    switch_dora_layers
)
from functions.spose_dimensions import classnames66


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
    
    for key in params1.keys():
        diff = torch.max(torch.abs(params1[key] - params2[key])).item()
        max_diff = max(max_diff, diff)
        if diff > tolerance:
            all_match = False
            print(f"  ✗ {key}: max diff = {diff:.2e}")
        else:
            print(f"  ✓ {key}: max diff = {diff:.2e}")
    
    return all_match, max_diff


def quick_test():
    """
    Quick test: Train for 2 steps, save, load, train 1 more step.
    Compare with continuous 3-step training.
    """
    print("\n" + "="*80)
    print("QUICK CHECKPOINT REPRODUCIBILITY TEST")
    print("="*80)
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Test parameters
    seed = 42
    lr = 1e-4
    batch_size = 8
    
    # Create dummy data
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_targets = torch.randn(batch_size, 66)
    
    temp_dir = tempfile.mkdtemp(prefix='quick_test_')
    checkpoint_path = os.path.join(temp_dir, 'checkpoint.pth')
    
    try:
        # ================================================================
        # CONTINUOUS TRAINING: 3 steps
        # ================================================================
        print("PART 1: Continuous training for 3 steps")
        print("-" * 80)
        
        seed_everything(seed)
        model_continuous = CLIPHBA(classnames=classnames66, backbone_name='ViT-L/14', pos_embedding=True)
        apply_dora_to_ViT(model_continuous, n_vision_layers=2, n_transformer_layers=1, r=32, dora_dropout=0.1)
        switch_dora_layers(model_continuous, freeze_all=True, dora_state=True)
        model_continuous.to(device)
        
        optimizer_continuous = torch.optim.AdamW(model_continuous.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Train for 3 steps
        model_continuous.train()
        for step in range(3):
            images = dummy_images.to(device)
            targets = dummy_targets.to(device)
            
            optimizer_continuous.zero_grad()
            predictions = model_continuous(images)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer_continuous.step()
            
            print(f"  Step {step+1}/3: Loss = {loss.item():.6f}")
        
        params_continuous = extract_dora_params(model_continuous)
        print("\nContinuous training complete")
        
        # ================================================================
        # CHECKPOINT TRAINING: 2 steps + save + load + 1 step
        # ================================================================
        print("\nPART 2: Training with checkpoint (2 steps → save → load → 1 step)")
        print("-" * 80)
        
        seed_everything(seed)
        model_checkpoint = CLIPHBA(classnames=classnames66, backbone_name='ViT-L/14', pos_embedding=True)
        apply_dora_to_ViT(model_checkpoint, n_vision_layers=2, n_transformer_layers=1, r=32, dora_dropout=0.1)
        switch_dora_layers(model_checkpoint, freeze_all=True, dora_state=True)
        model_checkpoint.to(device)
        
        optimizer_checkpoint = torch.optim.AdamW(model_checkpoint.parameters(), lr=lr)
        
        # Train for 2 steps
        model_checkpoint.train()
        for step in range(2):
            images = dummy_images.to(device)
            targets = dummy_targets.to(device)
            
            optimizer_checkpoint.zero_grad()
            predictions = model_checkpoint(images)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer_checkpoint.step()
            
            print(f"  Step {step+1}/2: Loss = {loss.item():.6f}")
        
        # Save checkpoint
        params_before_save = extract_dora_params(model_checkpoint)
        
        checkpoint_dict = {}
        for key, value in params_before_save.items():
            checkpoint_dict[key] = value
        
        torch.save(checkpoint_dict, checkpoint_path)
        print(f"\nCheckpoint saved to: {checkpoint_path}")
        
        # Create new model and load checkpoint
        seed_everything(seed)
        model_resumed = CLIPHBA(classnames=classnames66, backbone_name='ViT-L/14', pos_embedding=True)
        apply_dora_to_ViT(model_resumed, n_vision_layers=2, n_transformer_layers=1, r=32, dora_dropout=0.1)
        switch_dora_layers(model_resumed, freeze_all=True, dora_state=True)
        model_resumed.to(device)
        
        # Load checkpoint
        loaded_checkpoint = torch.load(checkpoint_path)
        model_resumed.load_state_dict(loaded_checkpoint, strict=False)
        print("Checkpoint loaded into new model")
        
        # Verify loaded params match saved params
        params_after_load = extract_dora_params(model_resumed)
        print("\nVerifying checkpoint load:")
        load_match, load_diff = compare_params(params_before_save, params_after_load, tolerance=1e-12)
        
        if load_match:
            print(f"✓ Checkpoint loaded correctly (max diff: {load_diff:.2e})")
        else:
            print(f"✗ Checkpoint loading has errors (max diff: {load_diff:.2e})")
        
        # Continue training for 1 more step
        optimizer_resumed = torch.optim.AdamW(model_resumed.parameters(), lr=lr)
        model_resumed.train()
        
        print("\nResuming training for 1 more step:")
        images = dummy_images.to(device)
        targets = dummy_targets.to(device)
        
        optimizer_resumed.zero_grad()
        predictions = model_resumed(images)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer_resumed.step()
        
        print(f"  Step 3/3: Loss = {loss.item():.6f}")
        
        params_resumed = extract_dora_params(model_resumed)
        print("\nCheckpoint training complete")
        
        # ================================================================
        # COMPARISON
        # ================================================================
        print("\n" + "="*80)
        print("FINAL COMPARISON")
        print("="*80)
        
        final_match, final_diff = compare_params(params_continuous, params_resumed, tolerance=1e-8)
        
        # Compare model outputs
        print("\nComparing model outputs:")
        test_input = torch.randn(4, 3, 224, 224).to(device)
        
        model_continuous.eval()
        model_resumed.eval()
        
        with torch.no_grad():
            output_continuous = model_continuous(test_input)
            output_resumed = model_resumed(test_input)
        
        output_diff = torch.max(torch.abs(output_continuous - output_resumed)).item()
        output_match = output_diff < 1e-6
        
        if output_match:
            print(f"✓ Outputs match (max diff: {output_diff:.2e})")
        else:
            print(f"✗ Outputs differ (max diff: {output_diff:.2e})")
        
        # ================================================================
        # SUMMARY
        # ================================================================
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        test_passed = final_match and output_match
        
        if test_passed:
            print("✅ TEST PASSED: Checkpoint loading is reproducible!")
        else:
            print("❌ TEST FAILED: Checkpoint loading has issues")
            if not final_match:
                print(f"  - Parameters differ (max: {final_diff:.2e})")
            if not output_match:
                print(f"  - Outputs differ (max: {output_diff:.2e})")
        
        print("="*80 + "\n")
        
        return test_passed
        
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    passed = quick_test()
    sys.exit(0 if passed else 1)

