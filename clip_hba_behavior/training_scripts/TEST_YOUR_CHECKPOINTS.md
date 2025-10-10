# Testing Your Existing Checkpoints

## Quick Start

Since you already have checkpoints saved, run this test to verify they can reproduce your training:

```bash
cd /home/wallacelab/Documents/GitHub/cvpr_perturbations/clip_hba_behavior/training_scripts
python test_existing_checkpoints.py
```

## What This Test Does

Your checkpoints are stored at:
```
/home/wallacelab/teba/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior/training_artifacts/dora_params/dora_params_20251008_211424/
```

The test will:

1. **Load checkpoint from epoch N-1** (e.g., `epoch2_dora_params.pth`)
2. **Train for 1 epoch** using your real data
3. **Compare the result** with your saved checkpoint at epoch N (e.g., `epoch3_dora_params.pth`)

If the test passes, it confirms that:
- Your checkpoint loading mechanism works correctly
- You can resume training from any epoch
- Results are perfectly reproducible

## Test Epochs

By default, the test checks epochs: **3, 10, and 20**

These were chosen to test:
- **Epoch 3**: Early training (model still changing rapidly)
- **Epoch 10**: Mid training (more stable)
- **Epoch 20**: Later training (well-established patterns)

## Expected Output

### If Test Passes ✅

```
==================================================
TESTING EXISTING CHECKPOINTS - EPOCH 3
==================================================
Checkpoint directory: /home/.../dora_params_20251008_211424
Test: Load epoch 2 → Train 1 epoch → Compare with epoch 3
==================================================

✓ Found checkpoint to load: .../epoch2_dora_params.pth
✓ Found target checkpoint: .../epoch3_dora_params.pth

Loading target checkpoint (epoch 3)...
✓ Loaded 9 parameter tensors

Using device: cuda:1

==================================================
STEP 1: Initialize model and load checkpoint from epoch 2
==================================================
✓ Loaded DoRA parameters from epoch 2

Verifying loaded parameters match checkpoint file:
  ✓ clip_model.visual.transformer.resblocks.22.attn.out_proj.m: max diff = 0.00e+00
  ✓ clip_model.visual.transformer.resblocks.22.attn.out_proj.delta_D_A: max diff = 0.00e+00
  ✓ clip_model.visual.transformer.resblocks.22.attn.out_proj.delta_D_B: max diff = 0.00e+00
  [... more parameters ...]
✓ Parameters loaded correctly (max diff: 0.00e+00)

==================================================
STEP 2: Train for 1 epoch (epoch 3)
==================================================
Training for 50 batches...
  Batch 1/50, Loss: 0.123456
  Batch 11/50, Loss: 0.098765
  ...
  Batch 41/50, Loss: 0.087654

✓ Training complete. Average loss: 0.091234

==================================================
STEP 3: Compare with target checkpoint (epoch 3)
==================================================

  ✓ clip_model.visual.transformer.resblocks.22.attn.out_proj.m: max diff = 2.34e-12
  ✓ clip_model.visual.transformer.resblocks.22.attn.out_proj.delta_D_A: max diff = 1.23e-11
  [... all parameters shown ...]

==================================================
TEST SUMMARY
==================================================
✅ TEST PASSED: Checkpoint loading perfectly reproduces training!
   Maximum parameter difference: 3.45e-11
   All parameters within tolerance (1e-8)
==================================================
```

### If Test Fails ❌

```
==================================================
TEST SUMMARY
==================================================
❌ TEST FAILED: Checkpoint loading does not reproduce training
   Maximum parameter difference: 5.67e-04

   Parameters that differ significantly:
     - clip_model.visual.transformer.resblocks.22.attn.out_proj.m: 5.67e-04
     - clip_model.transformer.resblocks.11.attn.out_proj.delta_D_A: 3.21e-05
==================================================
```

## Customizing the Test

You can modify the test by editing `test_existing_checkpoints.py`:

### Test Different Epochs

```python
# In main(), change this line:
test_epochs = [3, 10, 20]  # Change to any epochs you want to test

# For example:
test_epochs = [5, 15, 25, 50]
```

### Use Full Epochs Instead of Subset

```python
# In config dict, change:
'num_training_steps': 50,  # Use subset for faster testing

# To:
'num_training_steps': None,  # Use full epoch
```

### Change Checkpoint Directory

```python
# In main(), change:
checkpoint_dir = '/path/to/your/checkpoints'
```

## Understanding the Results

### What Different Differences Mean

- **< 1e-10**: Perfect match (within numerical precision)
- **1e-10 to 1e-8**: Excellent (acceptable floating point differences)
- **1e-8 to 1e-6**: Good (minor differences, likely acceptable)
- **1e-6 to 1e-4**: Concerning (investigate the cause)
- **> 1e-4**: Problem (checkpoint loading not working correctly)

### Common Causes of Failure

1. **Random seed not set**: Ensure `seed_everything()` is called
2. **Data shuffling differs**: DataLoader shuffle order may vary
3. **Optimizer state**: If optimizer state affects gradients
4. **Different batch**: Test uses first 50 batches by default

### Important Note About Data Ordering

The test uses `num_training_steps=50` by default (first 50 batches) rather than a full epoch. This is because:
- **DataLoader shuffling** creates different batch orders each time
- We can't perfectly reproduce the exact same batches without the original random state
- Testing 50 batches is sufficient to validate the checkpoint mechanism

If you need to test a full epoch, you would need to save and restore the DataLoader's random state as well.

## Integration with Your Training Loop

Your training loop (`cvpr_train_behavior.py`) loads checkpoints like this:

```python
# Line 895-903 in cvpr_train_behavior_things_pipeline.py
if config['training_run'] > 1:
    dora_params_state_dict = torch.load(dora_params_path)
    model.load_state_dict(dora_params_state_dict, strict=False)
```

This test validates that this mechanism works correctly for your 93-epoch training loop.

## Next Steps

1. **Run the test** to verify your checkpoints: `python test_existing_checkpoints.py`
2. **If it passes**: Your checkpoint loading is working correctly! ✅
3. **If it fails**: Investigate the differences and fix any issues ❌

## Questions?

- See `QUICK_START.md` for overview of all tests
- See `TEST_README.md` for detailed documentation
- Contact maintainers for specific issues

