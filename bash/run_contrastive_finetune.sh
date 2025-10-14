CUDA_VISIBLE_DEVICES=0 python -m chrombpnet.contrastive_finetune \
    --base_model_checkpoint /oak/stanford/groups/akundaje/abuen/personal_genome/personal_pytorch/output/reference/fold_0/checkpoints/best_model.ckpt \
    --consensus_peaks_path /oak/stanford/groups/akundaje/abuen/personal_genome/personal_pytorch/data/consensus_peaks/overlap.optimal_peak.narrowPeak.merged.bed \
    --individual_bigwig_dir /oak/stanford/groups/akundaje/ziwei75/african-omics/data/processed/ \
    --individual_vcf_dir /oak/stanford/groups/akundaje/ziwei75/african-omics/data/genotype/individual_vcf \
    --output_dir /oak/stanford/groups/akundaje/abuen/personal_genome/personal_pytorch/output/contrastive_finetuning \
    --max_epochs 50 \
    --batch_size 16 \
    --fast_dev_run False
