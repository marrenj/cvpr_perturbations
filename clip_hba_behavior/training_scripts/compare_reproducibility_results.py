"""
Helper script to compare baseline training results with reproducibility test results.

Usage:
    python compare_reproducibility_results.py \\
        /path/to/baseline/training_res.csv \\
        /path/to/test/test_training_res.csv \\
        --resume-from 5 \\
        --total-epochs 8
"""

import pandas as pd
import numpy as np
import sys
import argparse


def compare_results(baseline_csv, test_csv, resume_from, total_epochs):
    """
    Compare baseline and test results to verify reproducibility.
    
    Args:
        baseline_csv: Path to baseline training results CSV
        test_csv: Path to test training results CSV
        resume_from: Epoch number that was resumed from
        total_epochs: Total epochs run in test
    
    Returns:
        bool: True if results match (reproducible), False otherwise
    """
    print("="*80)
    print("REPRODUCIBILITY COMPARISON")
    print("="*80)
    print()
    
    # Load CSVs
    print(f"Loading baseline results: {baseline_csv}")
    baseline = pd.read_csv(baseline_csv)
    
    print(f"Loading test results: {test_csv}")
    test = pd.read_csv(test_csv)
    print()
    
    # Determine which epochs to compare
    epochs_to_compare = list(range(resume_from + 1, total_epochs + 1))
    
    print(f"Comparing epochs: {epochs_to_compare}")
    print(f"(Resumed from epoch {resume_from}, ran through epoch {total_epochs})")
    print()
    
    # Filter to relevant epochs
    baseline_slice = baseline[baseline['epoch'].isin(epochs_to_compare)].copy()
    test_slice = test[test['epoch'].isin(epochs_to_compare)].copy()
    
    # Check that we have the same number of epochs
    if len(baseline_slice) != len(test_slice):
        print(f"❌ ERROR: Different number of epochs!")
        print(f"   Baseline has {len(baseline_slice)} epochs")
        print(f"   Test has {len(test_slice)} epochs")
        return False
    
    if len(baseline_slice) == 0:
        print(f"❌ ERROR: No epochs found in range {epochs_to_compare}!")
        print(f"   Baseline epochs available: {baseline['epoch'].values}")
        print(f"   Test epochs available: {test['epoch'].values}")
        return False
    
    # Sort both by epoch to ensure alignment
    baseline_slice = baseline_slice.sort_values('epoch').reset_index(drop=True)
    test_slice = test_slice.sort_values('epoch').reset_index(drop=True)
    
    print(f"Found {len(baseline_slice)} epochs to compare")
    print()
    
    # Compare metrics
    all_match = True
    tolerance = 1e-6  # Floating point tolerance
    
    metrics = ['train_loss', 'test_loss', 'behavioral_rsa_rho', 'behavioral_rsa_p_value']
    
    for metric in metrics:
        if metric not in baseline_slice.columns or metric not in test_slice.columns:
            print(f"⚠️  Skipping {metric} (not found in both CSVs)")
            continue
        
        baseline_values = baseline_slice[metric].values
        test_values = test_slice[metric].values
        
        diff = np.abs(baseline_values - test_values)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        matches = max_diff < tolerance
        
        if matches:
            print(f"✅ {metric:25s} MATCH (max diff: {max_diff:.2e})")
        else:
            print(f"❌ {metric:25s} MISMATCH (max diff: {max_diff:.2e})")
            all_match = False
            
            # Show details
            print(f"   Mean difference: {mean_diff:.2e}")
            print(f"   Baseline values: {baseline_values}")
            print(f"   Test values:     {test_values}")
            print(f"   Differences:     {diff}")
            print()
    
    print()
    print("="*80)
    
    if all_match:
        print("🎉 SUCCESS: All metrics match! Reproducibility verified!")
        print("="*80)
        print()
        print("Your checkpoint/resume system is working perfectly.")
        print("You can confidently run perturbation experiments.")
        return True
    else:
        print("❌ FAILURE: Metrics do not match. Reproducibility NOT verified.")
        print("="*80)
        print()
        print("Possible issues:")
        print("  1. Wrong checkpoint epoch loaded")
        print("  2. Random states not properly restored")
        print("  3. Different random seed used in baseline vs test")
        print("  4. Baseline training may have been interrupted or corrupted")
        print()
        print("Check your checkpoint paths and random seed configuration.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Compare baseline and test training results for reproducibility verification'
    )
    parser.add_argument('baseline_csv', help='Path to baseline training_res.csv')
    parser.add_argument('test_csv', help='Path to test training_res.csv')
    parser.add_argument('--resume-from', type=int, required=True,
                       help='Epoch number that was resumed from (e.g., 5)')
    parser.add_argument('--total-epochs', type=int, required=True,
                       help='Total epochs run in test (e.g., 8)')
    
    args = parser.parse_args()
    
    # Run comparison
    success = compare_results(
        args.baseline_csv,
        args.test_csv,
        args.resume_from,
        args.total_epochs
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

