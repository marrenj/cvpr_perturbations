#!/bin/bash
# Simple wrapper to run experiments sequentially
# This replaces the SLURM script for non-SLURM environments

echo "🚀 Starting sequential experiment run..."
echo "This will run all 32 experiments one by one."
echo "Estimated total time: ~64 hours (32 experiments × ~2 hours each)"
echo ""
echo "Press Ctrl+C to stop at any time."
echo ""

# Activate conda environment
echo "Activating conda environment..."
module load anaconda
conda activate adaptive-clip

# Run the sequential experiment script
python run_experiments_sequential.py
