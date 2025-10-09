#!/bin/bash

samples=$(ls /oak/stanford/groups/akundaje/ziwei75/african-omics/data/processed | grep -E '^[GH]')
output_dir="/oak/stanford/groups/akundaje/abuen/personal_genome/personal_pytorch/output/personal_eval"

# Create logs directory if it doesn't exist
mkdir -p ./logs

# GPU queue config (edit as needed)
GPUS=(0 1 2 3)

# Use pipefail for safer pipelines
set -o pipefail

# FIFO as a GPU token pool
fifo=$(mktemp -u)
mkfifo "$fifo"
exec 3<>"$fifo"
rm -f "$fifo"

# seed tokens
for g in "${GPUS[@]}"; do echo "$g" >&3; done

# launch jobs (one per GPU at a time)
for sample_id in $samples; do
    # wait for a free GPU token
    read -r gpu <&3

    {
        echo "[$(date +%T)] start ${sample_id} on GPU ${gpu}"

        CUDA_VISIBLE_DEVICES="$gpu" chrombpnet predict \
            --peaks /oak/stanford/groups/akundaje/abuen/personal_genome/personal_pytorch/data/consensus_peaks/overlap.optimal_peak.narrowPeak.merged.bed \
            --bigwig /oak/stanford/groups/akundaje/ziwei75/african-omics/data/processed/${sample_id}/${sample_id}_unstranded.bw \
            --bias /oak/stanford/groups/akundaje/abuen/personal_genome/personal_pytorch/models/bias.h5 \
            --adjust_bias \
            --checkpoint /oak/stanford/groups/akundaje/abuen/personal_genome/personal_pytorch/output/reference/fold_0/checkpoints/best_model.ckpt \
            -o ${output_dir}/${sample_id} \
            --sample_id ${sample_id} \
            --vcf_dir /oak/stanford/groups/akundaje/ziwei75/african-omics/data/genotype/individual_vcf \
            --rename_map /oak/stanford/groups/akundaje/ziwei75/african-omics/data/genotype/individual_vcf/rename.txt \
            > ./logs/personal_eval_${sample_id}_$(date +%Y%m%d_%H%M%S).log 2>&1

        status=$?
        echo "$gpu" >&3   # return GPU token
        echo "[$(date +%T)] done  ${sample_id} on GPU ${gpu} (exit ${status})"
        exit $status
    } &
done

wait
exec 3>&- 3<&-
echo "All samples finished."