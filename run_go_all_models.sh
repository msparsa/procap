#!/bin/bash

# GO All Models Benchmark Runner
# This script ensures proper environment activation before running the benchmark

# Find and activate conda environment with torch
CONDA_BASE="./mparsa/miniconda3"

# Source conda
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Try to find an environment with torch
for env_dir in "${CONDA_BASE}"/envs/*; do
    if [ -d "$env_dir" ]; then
        env_name=$(basename "$env_dir")
        echo "Testing environment: $env_name"
        conda activate "$env_name" 2>/dev/null
        if python -c "import torch" 2>/dev/null; then
            echo "Found torch in environment: $env_name"
            echo "Running benchmark..."
            cd ./mparsa/bench/procap-v2
            python run_go_all_models.py
            exit $?
        fi
        conda deactivate 2>/dev/null
    fi
done

# If no conda env found, try base environment
echo "No conda environment with torch found, trying base environment..."
cd ./mparsa/bench/procap-v2
python run_go_all_models.py
