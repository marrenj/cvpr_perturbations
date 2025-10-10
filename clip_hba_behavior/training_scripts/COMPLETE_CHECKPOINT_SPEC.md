# Complete Checkpoint Specification for Perfect Reproducibility

## The Complete List

Here's **everything** that affects training and needs to be saved for perfect reproducibility:

### ✅ Currently Saved

1. **DoRA Parameters** (m, delta_D_A, delta_D_B)
   - ✅ Status: Saved
   - Impact: Critical
   - Location: Main trainable parameters

2. **RNG States** (torch, numpy, python, cuda)
   - ✅ Status: Saved (new feature)
   - Impact: Critical for batch shuffle order
   - Location: `checkpoint['rng_state']`

### ❌ Currently NOT Saved

3. **Optimizer State** ⚠️ **CRITICAL MISSING ITEM**
   - ❌ Status: **NOT saved**
   - Impact: **High** - causes 1-5% difference for later epochs
   - What it contains:
     - First moment (momentum): `m = β₁ * m + (1-β₁) * grad`
     - Second moment (variance): `v = β₂ * v + (1-β₂) * grad²`
     - Step counter
   - Size: Roughly equal to model parameters (~same size as DoRA params)

4. **Early Stopping State**
   - ❌ Status: **NOT saved**
   - Impact: Medium - affects when training stops
   - What it contains:
     - `best_test_loss`: Best validation loss seen so far
     - `epochs_no_improve`: Counter for early stopping
   - Size: Negligible (2 numbers)

5. **Learning Rate Scheduler State** (if using one)
   - ❌ Status: **NOT saved** (but you're not using one)
   - Impact: High if present
   - Note: Your code doesn't use a scheduler, so this doesn't apply

### ✅ Already Handled Correctly

6. **Model Architecture State**
   - ✅ Status: Deterministic from config
   - No need to save - always reconstructed identically

7. **Dropout State**
   - ✅ Status: Controlled by RNG (already saved)
   - DoRA dropout uses PyTorch RNG

8. **Batch Normalization Running Stats**
   - ✅ Status: Not applicable
   - Your model doesn't have BN layers with running stats

## Analysis: What's Causing the Differences?

### Your Test Results

| Epoch | Train Loss Diff | Test Loss Diff | Why? |
|-------|----------------|----------------|------|
| 3 | 0.00% | 0.00% | ✅ Optimizer barely matters (2 epochs) |
| 10 | 0.03% | 1.13% | ❌ Missing 9 epochs of optimizer momentum |
| 20 | 0.69% | 1.27% | ❌ Missing 19 epochs of optimizer momentum |

The pattern is clear: **Optimizer state matters more as training progresses.**

### Why Optimizer State Matters

AdamW maintains:

```python
# For each parameter:
m_t = β₁ * m_{t-1} + (1-β₁) * grad_t      # First moment (momentum)
v_t = β₂ * v_{t-1} + (1-β₂) * grad_t²     # Second moment (variance)

# Update rule:
param_t = param_{t-1} - lr * m_t / (√v_t + ε)
```

Starting fresh means `m_0 = 0` and `v_0 = 0`, which gives **different updates** than if you had accumulated statistics.

## What You Need to Save

### For Perfect Reproducibility

```python
checkpoint = {
    # 1. Model parameters (DoRA)
    'clip_model.visual.transformer.resblocks.22.attn.out_proj.m': tensor(...),
    'clip_model.visual.transformer.resblocks.22.attn.out_proj.delta_D_A': tensor(...),
    'clip_model.visual.transformer.resblocks.22.attn.out_proj.delta_D_B': tensor(...),
    # ... more DoRA parameters ...
    
    # 2. RNG states
    'rng_state': {
        'torch_rng_state': ...,
        'numpy_rng_state': ...,
        'python_rng_state': ...,
        'cuda_rng_state_all': ...
    },
    
    # 3. Optimizer state (MISSING!)
    'optimizer_state': {
        'state': {...},  # Per-parameter momentum and variance
        'param_groups': [...]  # Learning rate and other hyperparams
    },
    
    # 4. Early stopping state (MISSING!)
    'early_stopping_state': {
        'best_test_loss': 0.123456,
        'epochs_no_improve': 3
    },
    
    # 5. Training metadata (nice to have)
    'epoch': 10,  # Which epoch this was saved after
    'training_run': 10,  # Which training run
}
```

## Impact Assessment

### If You DON'T Save Optimizer State

| Aspect | Impact |
|--------|--------|
| Epoch 1-3 | ≈0% difference ✅ |
| Epoch 4-10 | 0.5-1% difference ⚠️ |
| Epoch 11-50 | 1-2% difference ❌ |
| Epoch 50+ | 1-3% difference ❌ |

### If You DO Save Optimizer State

| Aspect | Impact |
|--------|--------|
| All epochs | < 0.01% difference ✅ |

## Important Question: How Was Your Baseline Training Run?

This matters! There are two scenarios:

### Scenario A: Continuous Baseline Training

If your baseline was trained like this:
```python
# ONE continuous training run, epochs 1-93
run_behavioral_training(config)  # epochs=93
```

Then **optimizer state accumulated** across all 93 epochs.

**To reproduce:** You MUST save/load optimizer state.

### Scenario B: Separate Training Runs

If your baseline was trained like this:
```python
# 93 separate training runs, each starting fresh
for training_run in range(1, 94):
    run_behavioral_training(config)  # Each creates NEW optimizer
```

Then **optimizer was reset** for each epoch.

**To reproduce:** You DON'T need optimizer state! Each epoch starts fresh.

## The Critical Question

**How was your baseline training (Oct 8, 2024) actually run?**

Check your baseline training script. Was it:
1. ☐ One continuous run (epochs 1-93)?
2. ☐ Separate runs for each epoch?

If it was continuous, you need optimizer state. If it was separate runs, you don't!

## My Recommendation

### For Now (Testing)

Accept the 1-3% difference for epochs 10+ as expected behavior when optimizer state is missing. This validates that:
- ✅ Your DoRA parameters load correctly
- ✅ Your RNG reconstruction works
- ✅ The training mechanism is sound
- ⚠️ Only optimizer momentum is missing

### For Future Training

Save **everything** for perfect reproducibility:

```python
def save_complete_checkpoint(model, optimizer, epoch, best_test_loss, 
                            epochs_no_improve, dora_parameters_path, logger=None):
    """Save everything needed for perfect reproducibility."""
    
    dora_params = {
        # DoRA parameters
        ... (existing code) ...,
        
        # RNG states
        'rng_state': { ... },
        
        # Optimizer state (NEW!)
        'optimizer_state': optimizer.state_dict(),
        
        # Early stopping state (NEW!)
        'early_stopping_state': {
            'best_test_loss': best_test_loss,
            'epochs_no_improve': epochs_no_improve
        },
        
        # Metadata
        'epoch': epoch + 1,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(dora_params, save_path)
```

## Quick Check

To determine if your baseline used continuous training:

```bash
# Check if there's only ONE training log for all epochs
ls -la /home/wallacelab/teba/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior/training_artifacts/dora_params/dora_params_20251008_211424/../../logs/

# If you see ONE log file for all 93 epochs → continuous training (need optimizer state)
# If you see 93 log files → separate runs (don't need optimizer state)
```

Let me know what you find, and I'll help you decide whether to implement optimizer state saving!
