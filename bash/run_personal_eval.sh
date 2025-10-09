output_dir="/oak/stanford/groups/akundaje/abuen/personal_genome/personal_pytorch/output/personal_eval"
sample_id=GM18508

CUDA_VISIBLE_DEVICES=1 chrombpnet predict \
    --peaks /oak/stanford/groups/akundaje/abuen/personal_genome/personal_pytorch/data/consensus_peaks/overlap.optimal_peak.narrowPeak.merged.bed \
    --bigwig /oak/stanford/groups/akundaje/ziwei75/african-omics/data/processed/${sample_id}/${sample_id}_unstranded.bw \
    --bias /oak/stanford/groups/akundaje/abuen/personal_genome/personal_pytorch/models/bias.h5 \
    --adjust_bias \
    --checkpoint /oak/stanford/groups/akundaje/abuen/personal_genome/personal_pytorch/output/reference/fold_0/checkpoints/best_model.ckpt \
    -o ${output_dir}/${sample_id} \
    --sample_id ${sample_id} \
    --vcf_dir /oak/stanford/groups/akundaje/ziwei75/african-omics/data/genotype/individual_vcf \
    --rename_map /oak/stanford/groups/akundaje/ziwei75/african-omics/data/genotype/individual_vcf/rename.txt
