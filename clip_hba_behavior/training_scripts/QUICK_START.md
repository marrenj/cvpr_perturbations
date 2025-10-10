# Quick Start Guide - Checkpoint Reproducibility Tests

## TL;DR

```bash
cd /home/wallacelab/Documents/GitHub/cvpr_perturbations/clip_hba_behavior/training_scripts

# BEST: Test against your actual training results CSV (RECOMMENDED!)
python test_reproduce_training_results.py

# Or test with your existing checkpoints
python test_existing_checkpoints.py

# Or run the quick test (uses dummy data)
python test_checkpoint_quick.py

# Or use the test runner
./run_tests.sh results   # Test against training results CSV (recommended!)
./run_tests.sh existing  # Test existing checkpoints
./run_tests.sh quick     # Quick test with dummy data
```

## What These Tests Do

These tests verify that when you:
1. Train your model to epoch N
2. Save a checkpoint
3. Load that checkpoint and continue training

You get **exactly the same results** as if you had trained continuously without interruption.

### Four Types of Tests

1. **Training Results Test** (`test_reproduce_training_results.py`) - **MOST RECOMMENDED** ⭐
   - Validates against your actual training results CSV file
   - Loads checkpoint from epoch N-1, trains for 1 epoch
   - Compares train loss, test loss, and behavioral RSA with recorded values
   - **Most comprehensive** - validates the entire training process, not just parameters
   - This is what you should run to ensure full reproducibility!

2. **Existing Checkpoint Test** (`test_existing_checkpoints.py`)
   - Uses your actual saved checkpoints from real training runs
   - Loads checkpoint from epoch N-1, trains for 1 epoch
   - Compares DoRA parameters with your saved checkpoint at epoch N
   - Good for validating parameter consistency

3. **Quick Test** (`test_checkpoint_quick.py`)
   - Uses dummy data for fast validation
   - Tests the basic checkpoint mechanism (3 training steps)
   - Good for development and debugging

4. **Full Test** (`test_checkpoint_reproducibility.py`)
   - Creates new checkpoints with real data
   - More comprehensive but slower
   - Good for validating the full training pipeline

## Running the Tests

### Option 1: Direct Python Execution

**Training Results Test** (5-10 minutes, validates against CSV) ⭐ **RECOMMENDED**:
```bash
python test_reproduce_training_results.py
```
This validates train/test loss AND behavioral RSA metrics!

**Existing Checkpoint Test** (2-3 minutes, validates parameters):
```bash
python test_existing_checkpoints.py
```

**Quick Test** (30 seconds, uses dummy data):
```bash
python test_checkpoint_quick.py
```

**Full Test** (5-10 minutes, creates new checkpoints):
```bash
python test_checkpoint_reproducibility.py
```

### Option 2: Using the Test Runner Script

**Training results test** (recommended) ⭐:
```bash
./run_tests.sh results
```

**Existing checkpoint test**:
```bash
./run_tests.sh existing
```

**Quick test only**:
```bash
./run_tests.sh quick
```

**Full test only**:
```bash
./run_tests.sh full
```

**All tests**:
```bash
./run_tests.sh
```

## Expected Output

### Successful Test
```
==================================================
QUICK CHECKPOINT REPRODUCIBILITY TEST
==================================================
Device: cuda:1

PART 1: Continuous training for 3 steps
------------------------------------------------
  Step 1/3: Loss = 0.123456
  Step 2/3: Loss = 0.098765
  Step 3/3: Loss = 0.087654

Continuous training complete

PART 2: Training with checkpoint (2 steps → save → load → 1 step)
------------------------------------------------
  Step 1/2: Loss = 0.123456
  Step 2/2: Loss = 0.098765

Checkpoint saved to: /tmp/quick_test_xyz/checkpoint.pth
Checkpoint loaded into new model

Verifying checkpoint load:
  ✓ clip_model.visual.transformer.resblocks.22.attn.out_proj.m: max diff = 0.00e+00
  ✓ clip_model.visual.transformer.resblocks.22.attn.out_proj.delta_D_A: max diff = 0.00e+00
  ✓ clip_model.visual.transformer.resblocks.22.attn.out_proj.delta_D_B: max diff = 0.00e+00
  [... more parameters ...]
✓ Checkpoint loaded correctly (max diff: 0.00e+00)

Resuming training for 1 more step:
  Step 3/3: Loss = 0.087654

Checkpoint training complete

==================================================
FINAL COMPARISON
==================================================
  ✓ clip_model.visual.transformer.resblocks.22.attn.out_proj.m: max diff = 1.23e-12
  [... all parameters shown ...]

Comparing model outputs:
✓ Outputs match (max diff: 3.45e-09)

==================================================
TEST SUMMARY
==================================================
✅ TEST PASSED: Checkpoint loading is reproducible!
==================================================
```

### Failed Test
```
==================================================
FINAL COMPARISON
==================================================
  ✗ clip_model.visual.transformer.resblocks.22.attn.out_proj.m: max diff = 1.23e-04
  [... problem parameters shown ...]

Comparing model outputs:
✗ Outputs differ (max diff: 5.67e-02)

==================================================
TEST SUMMARY
==================================================
❌ TEST FAILED: Checkpoint loading has issues
  - Parameters differ (max: 1.23e-04)
  - Outputs differ (max: 5.67e-02)
==================================================
```

## When to Run These Tests

### Always Run:
- Before starting a new training loop that depends on checkpoints
- After modifying checkpoint loading/saving code
- After updating PyTorch or CUDA versions

### Good Practice:
- Weekly, to ensure environment hasn't changed
- After any changes to the DoRA layer implementation
- Before long training runs (93 epochs in your case)

## Troubleshooting

### Test Fails with Parameter Differences

**Problem**: Parameters differ after loading checkpoint

**Solutions**:
1. Check that `seed_everything()` is called before model initialization
2. Verify `strict=False` is used in `load_state_dict()`
3. Ensure DoRA parameters are correctly saved/loaded
4. Check for any random operations in model initialization

### Test Fails with Output Differences

**Problem**: Model outputs differ even though parameters match

**Solutions**:
1. Ensure `model.eval()` is called before output comparison
2. Check for non-deterministic operations (e.g., dropout during eval)
3. Verify CUDA determinism is enabled

### Test is Too Slow

**Problem**: Full test takes too long

**Solutions**:
1. Use the quick test for rapid iteration: `python test_checkpoint_quick.py`
2. Reduce `num_training_steps` in the full test
3. Use a smaller subset of data

## Integration with Your Training Loop

Your current training loop in `cvpr_train_behavior.py` loads checkpoints like this:

```python
# Lines 895-903 in cvpr_train_behavior_things_pipeline.py
if config['training_run'] > 1:
    dora_params_state_dict = torch.load(dora_params_path)
    model.load_state_dict(dora_params_state_dict, strict=False)
    logger.info(f"Loaded DoRA parameters from {dora_params_path}")
```

These tests verify that this mechanism works correctly and allows you to:
- Resume from any epoch (training runs 1-93)
- Reproduce results exactly
- Apply perturbations at specific epochs with confidence

## Questions?

See `TEST_README.md` for detailed documentation or contact the maintainers.

