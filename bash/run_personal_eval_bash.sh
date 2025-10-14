#!/bin/bash

samples=$(ls /oak/stanford/groups/akundaje/ziwei75/african-omics/data/processed | grep -E '^[GH]')
output_dir="/oak/stanford/groups/akundaje/abuen/personal_genome/personal_pytorch/output/personal_eval_ENCFF142IOR"
folds=(0 1 2 3 4)

# Create logs directory if it doesn't exist
mkdir -p ./logs

# GPU queue config (edit as needed)
GPUS=(0 1 2)

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
for fold in ${folds[@]}; do
    for sample_id in $samples; do
        # wait for a free GPU token
        read -r gpu <&3

        {
            log_file="./logs/personal_eval_${sample_id}_fold_${fold}_$(date +%Y%m%d_%H%M%S).log"
            echo "[$(date +%T)] start ${sample_id} on GPU ${gpu}"
            echo "Log file: $(pwd)/${log_file}"

            CUDA_VISIBLE_DEVICES="$gpu" chrombpnet predict \
                --peaks /oak/stanford/groups/akundaje/abuen/personal_genome/personal_pytorch/data/consensus_peaks/overlap.optimal_peak.narrowPeak.merged.bed \
                --bigwig /oak/stanford/groups/akundaje/ziwei75/african-omics/data/processed/${sample_id}/${sample_id}_unstranded.bw \
                --bias /oak/stanford/groups/akundaje/abuen/personal_genome/personal_pytorch/models/ENCFF142IOR/fold_${fold}/model.bias_scaled.fold_${fold}.ENCSR637XSC.h5 \
                --adjust_bias \
                --checkpoint /oak/stanford/groups/akundaje/abuen/personal_genome/personal_pytorch/models/ENCFF142IOR/fold_${fold}/model.chrombpnet_nobias.fold_${fold}.ENCSR637XSC.h5 \
                --fold ${fold} \
                -o ${output_dir}/${sample_id} \
                --sample_id ${sample_id} \
                --vcf_dir /oak/stanford/groups/akundaje/ziwei75/african-omics/data/genotype/individual_vcf \
                --rename_map /oak/stanford/groups/akundaje/ziwei75/african-omics/data/genotype/individual_vcf/rename.txt \
                > "${log_file}" 2>&1

            status=$?
            echo "$gpu" >&3   # return GPU token
            echo "[$(date +%T)] done  ${sample_id} on GPU ${gpu} (exit ${status})"
            exit $status
        } &
    done
done

wait
exec 3>&- 3<&-
echo "All samples finished."
