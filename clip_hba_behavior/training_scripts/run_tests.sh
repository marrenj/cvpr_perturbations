#!/bin/bash

# Script to run checkpoint reproducibility tests

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=================================================="
echo "Checkpoint Reproducibility Test Suite"
echo "=================================================="
echo ""

# Check if we're in the right directory
if [ ! -f "test_checkpoint_quick.py" ]; then
    echo -e "${RED}Error: Must be run from the training_scripts directory${NC}"
    exit 1
fi

# Function to run a test
run_test() {
    local test_name=$1
    local test_file=$2
    
    echo ""
    echo -e "${YELLOW}Running $test_name...${NC}"
    echo "--------------------------------------------------"
    
    python "$test_file"
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ $test_name PASSED${NC}"
        return 0
    else
        echo -e "${RED}✗ $test_name FAILED${NC}"
        return 1
    fi
}

# Parse command line arguments
if [ "$1" == "quick" ]; then
    echo "Running quick test only..."
    run_test "Quick Test" "test_checkpoint_quick.py"
    exit $?
elif [ "$1" == "full" ]; then
    echo "Running full test only..."
    run_test "Full Test" "test_checkpoint_reproducibility.py"
    exit $?
elif [ "$1" == "existing" ]; then
    echo "Running existing checkpoint test only..."
    run_test "Existing Checkpoint Test" "test_existing_checkpoints.py"
    exit $?
elif [ "$1" == "results" ]; then
    echo "Running training results reproducibility test only..."
    run_test "Training Results Test" "test_reproduce_training_results.py"
    exit $?
else
    # Run all tests
    echo "Running tests..."
    echo ""
    
    failed=0
    
    run_test "Quick Test" "test_checkpoint_quick.py"
    if [ $? -ne 0 ]; then
        failed=$((failed + 1))
    fi
    
    echo ""
    echo "=================================================="
    echo ""
    
    # Optionally run existing checkpoint test if checkpoints are available
    checkpoint_dir="/home/wallacelab/teba/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior/training_artifacts/dora_params/dora_params_20251008_211424"
    training_results="/home/wallacelab/teba/marren/temporal_dynamics_of_human_alignment/clip_hba_behavior/training_results/training_res_20251008_211424.csv"
    
    if [ -d "$checkpoint_dir" ] && [ -f "$training_results" ]; then
        echo "Existing training data found. Running training results test..."
        echo ""
        run_test "Training Results Test" "test_reproduce_training_results.py"
        if [ $? -ne 0 ]; then
            failed=$((failed + 1))
        fi
        echo ""
        echo "=================================================="
        echo ""
    elif [ -d "$checkpoint_dir" ]; then
        echo "Existing checkpoints found. Running existing checkpoint test..."
        echo ""
        run_test "Existing Checkpoint Test" "test_existing_checkpoints.py"
        if [ $? -ne 0 ]; then
            failed=$((failed + 1))
        fi
        echo ""
        echo "=================================================="
        echo ""
    fi
    
    if [ $failed -eq 0 ]; then
        echo -e "${GREEN}All tests passed!${NC}"
        exit 0
    else
        echo -e "${RED}$failed test(s) failed${NC}"
        exit 1
    fi
fi

