# Checkpoint Reproducibility Tests

This directory contains tests to verify that DoRA parameter checkpoint loading allows for perfect reproducibility of training results at any given epoch.

## Overview

When training neural networks with checkpointing, it's crucial that loading from a checkpoint produces identical results to continuous training. These tests verify that your DoRA parameter loading mechanism maintains this property.

## Test Files

### 1. `test_checkpoint_quick.py` - Quick Test (Recommended for development)

**Purpose**: Fast test using dummy data and minimal training steps.

**What it tests**:
- Trains a model for 3 steps continuously
- Trains another model for 2 steps, saves checkpoint, loads it, then trains 1 more step
- Verifies both approaches produce identical DoRA parameters and model outputs

**When to use**: 
- During development when making changes to checkpoint loading code
- Quick verification that basic checkpoint mechanism works
- CI/CD pipelines

**Runtime**: ~30 seconds (depends on GPU availability)

**Usage**:
```bash
cd /home/wallacelab/Documents/GitHub/cvpr_perturbations/clip_hba_behavior/training_scripts
python test_checkpoint_quick.py
```

**Expected output**:
- ✅ TEST PASSED: Checkpoint loading is reproducible!

### 2. `test_checkpoint_reproducibility.py` - Full Test

**Purpose**: Comprehensive test using real training data and multiple epochs.

**What it tests**:
- Part 1: Trains continuously from epoch 0 to epoch N
- Part 2: Trains from epoch 0 to epoch N-1, saves checkpoint
- Part 3: Loads checkpoint and continues from epoch N-1 to epoch N
- Part 4: Verifies DoRA parameters at epoch N are identical between approaches

**When to use**:
- Before deploying to production
- After making significant changes to training pipeline
- Periodic validation (e.g., weekly)

**Runtime**: ~5-10 minutes (depends on `num_training_steps` parameter)

**Usage**:
```bash
cd /home/wallacelab/Documents/GitHub/cvpr_perturbations/clip_hba_behavior/training_scripts
python test_checkpoint_reproducibility.py
```

**Configuration**: Edit the `test_config` dictionary in `main()` to adjust:
- `num_training_steps`: Number of batches per epoch (default: 10)
- `test_epoch`: Which epoch to test (default: 3)
- Other hyperparameters as needed

## What Gets Tested

Both tests verify:

1. **DoRA Parameter Reproducibility**
   - `m` (magnitude parameters)
   - `delta_D_A` (low-rank adaptation matrix A)
   - `delta_D_B` (low-rank adaptation matrix B)
   
2. **Model Output Consistency**
   - Forward pass outputs on identical inputs
   
3. **Checkpoint Loading Correctness**
   - Parameters are correctly saved to disk
   - Parameters are correctly loaded from disk
   - No information is lost during save/load cycle

## Understanding Test Results

### Successful Test Output

```
✓ clip_model.visual.transformer.resblocks.22.attn.out_proj.m: max diff = 1.23e-12
✓ clip_model.visual.transformer.resblocks.22.attn.out_proj.delta_D_A: max diff = 2.45e-12
...
✓ Outputs match (max diff: 3.14e-09)
✅ TEST PASSED: Checkpoint loading is reproducible!
```

### Failed Test Output

```
✗ clip_model.visual.transformer.resblocks.22.attn.out_proj.m: max diff = 1.23e-04
✗ Outputs differ (max diff: 5.67e-02)
❌ TEST FAILED: Checkpoint loading has issues
  - Parameters differ (max: 1.23e-04)
  - Outputs differ (max: 5.67e-02)
```

## Numerical Tolerances

The tests use different tolerance levels for different comparisons:

- **Parameter loading verification**: `1e-12` (nearly exact match expected)
- **DoRA parameter comparison**: `1e-8` to `1e-10` (accounts for floating point arithmetic)
- **Model output comparison**: `1e-6` (slightly more lenient due to accumulated differences)

## Common Issues and Solutions

### Issue: Test fails with large parameter differences

**Possible causes**:
1. Random seed not being properly restored
2. Optimizer state not being saved/loaded
3. Model architecture mismatch between save and load

**Solution**: Check that `seed_everything()` is called before model initialization

### Issue: Test fails with small differences (e.g., 1e-7)

**Possible causes**:
1. Different CUDA versions or hardware
2. Non-deterministic CUDA operations
3. Different floating point precision handling

**Solution**: This may be acceptable depending on your use case. Adjust tolerance levels if needed.

### Issue: Parameters match but outputs differ

**Possible causes**:
1. Dropout is enabled during evaluation
2. Batch normalization stats differ
3. Random transformations in data loading

**Solution**: Ensure `model.eval()` is called before output comparison

## Extending the Tests

To add additional checks:

1. **Check optimizer state**:
```python
# Save optimizer state
checkpoint_dict['optimizer_state'] = optimizer.state_dict()

# Load optimizer state
optimizer.load_state_dict(checkpoint_dict['optimizer_state'])
```

2. **Check learning rate scheduler**:
```python
# Similar to optimizer state
checkpoint_dict['scheduler_state'] = scheduler.state_dict()
```

3. **Check random number generator state**:
```python
checkpoint_dict['rng_state'] = torch.get_rng_state()
checkpoint_dict['cuda_rng_state'] = torch.cuda.get_rng_state()
```

## Integration with Training Loop

These tests verify the checkpoint mechanism used in your training loop:

```python
# From cvpr_train_behavior.py, lines 895-903
if config['training_run'] > 1:
    dora_params_state_dict = torch.load(dora_params_path)
    model.load_state_dict(dora_params_state_dict, strict=False)
    logger.info(f"Loaded DoRA parameters from {dora_params_path}")
```

The tests ensure this loading mechanism allows you to:
- Resume training from any epoch
- Reproduce results exactly as if training had never been interrupted
- Verify that perturbations applied at specific epochs are correctly handled

## Running Tests in CI/CD

Add to your CI/CD pipeline:

```yaml
- name: Test checkpoint reproducibility
  run: |
    cd clip_hba_behavior/training_scripts
    python test_checkpoint_quick.py
```

## Troubleshooting

If tests fail:

1. **Run the quick test first** to isolate the issue
2. **Check the detailed output** for which parameters differ
3. **Verify CUDA determinism** is enabled (`torch.backends.cudnn.deterministic = True`)
4. **Ensure consistent environments** between test runs
5. **Check disk I/O** - corrupt checkpoints can cause issues

## Notes

- These tests use `strict=False` in `load_state_dict()` to match your training loop behavior
- Only DoRA parameters are saved/loaded, not the full model state
- The tests verify that this partial loading is sufficient for reproducibility
- Random seed management is crucial for reproducibility

## Contact

For questions or issues with these tests, please refer to the main project documentation or contact the maintainers.

