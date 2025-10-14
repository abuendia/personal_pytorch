#!/bin/bash

# Activate conda environment
source ~/.bashrc
conda activate personal

# Change to script directory
cd /oak/stanford/groups/akundaje/abuen/personal_genome/personal_pytorch

# Run the truly optimized Python script
python scripts/run_personal_eval_truly_optimized.py \
    --auto-samples \
    --gpus 1 2 \
    --max-workers 5 \
    --log-dir ./logs \
    --output-dir /oak/stanford/groups/akundaje/abuen/personal_genome/personal_pytorch/output/personal_eval_ENCFF142IOR_corrected

echo "Optimized evaluation completed"