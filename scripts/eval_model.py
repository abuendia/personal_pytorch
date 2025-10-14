# %%
import os 
import pandas as pd
from tqdm import tqdm

# %%
personal_eval_dir = "/oak/stanford/groups/akundaje/abuen/personal_genome/personal_pytorch/output/personal_eval_ENCFF142IOR"
results_dict = {}

for sample_id in os.listdir(personal_eval_dir):
    cur_sample_dfs = []
    for fold in range(5):
        pred_file = f"/oak/stanford/groups/akundaje/abuen/personal_genome/personal_pytorch/output/personal_eval_ENCFF142IOR/{sample_id}/fold_{fold}/predict/test/regions.csv"
        pred_df = pd.read_csv(pred_file, sep="\t")
        pred_df = pred_df[pred_df["is_peak"] == 1]
        cur_sample_dfs.append(pred_df)
    results_dict[sample_id] = pd.concat(cur_sample_dfs, axis=0)
        

# %%
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

# Build matrices: rows = regions, cols = individuals
# Assume all individuals' dataframes have the same regions in the same order
sample_ids = list(results_dict.keys())
pred_mat = pd.concat([results_dict[sid]["pred_count"].reset_index(drop=True)
                      for sid in sample_ids], axis=1)
true_mat = pd.concat([results_dict[sid]["true_count"].reset_index(drop=True)
                      for sid in sample_ids], axis=1)

pred_mat.columns = sample_ids
true_mat.columns = sample_ids

# %% 
# Compute cross-individual correlations per region
# check if there is nan in any cell 

print(pred_mat.isna().sum().sum())
print(true_mat.isna().sum().sum())

# %%
# compute correlation with respect to the rows
per_sample_corrs = []

for sample_id in sample_ids:
    preds = pred_mat[sample_id].values
    trues = true_mat[sample_id].values
    r, _ = spearmanr(preds, trues)
    per_sample_corrs.append(r)

per_peak_corrs = []
for i in range(len(pred_mat)):
    preds = pred_mat.iloc[i, :].values
    trues = true_mat.iloc[i, :].values
    r, _ = spearmanr(preds, trues)
    per_peak_corrs.append(r)

# %%
print(np.nanmean(per_sample_corrs))
# %%
print(np.nanmean(per_peak_corrs))
# %%

import seaborn as sns
# flatten matrix to one vector
preds = pred_mat.values.flatten()
trues = true_mat.values.flatten()
sns.scatterplot(x=preds, y=trues, alpha=0.1)
# %% 

# number of nan in per_peak_corrs
print(np.sum(np.isnan(per_peak_corrs)))
print(len(per_peak_corrs))
# %%
# number of nan in per_sample_corrs

import matplotlib.pyplot as plt

plt.hist(per_peak_corrs, bins=100)
plt.title("Cross-individual correlation (per peak)")
plt.xlabel("Spearman correlation")
plt.axvline(x=np.nanmean(per_peak_corrs), color='red', linestyle='--')
plt.show()
# %%


plt.hist(per_sample_corrs, bins=20)
plt.title("Cross-peak correlation (per sample)")
plt.xlabel("Spearman correlation")
plt.axvline(x=np.nanmean(per_sample_corrs), color='red', linestyle='--')
plt.show()
# %%
caqtl_path = "/oak/stanford/groups/akundaje/abuen/personal_genome/legacy/personal_genome/data/variant_benchmark/caqtls.african.lcls.asb.benchmarking.all.tsv"
caqtl_df = pd.read_csv(caqtl_path, sep='\t')
# %%
caqtl_df["chrom"] = caqtl_df["var.chr"].apply(lambda x: x.split("chr")[1])
# %%

consensus_peaks = "/oak/stanford/groups/akundaje/abuen/personal_genome/personal_pytorch/data/consensus_peaks/overlap.optimal_peak.narrowPeak.merged.bed"
consensus_peaks_df = pd.read_csv(consensus_peaks, sep='\t', header=None)
# %%

# filter rows of pred_df such that the peak contains at least one caQTL in caqtl_df

def peaks_contain_caqtls(pred_df, caqtl_df):
    """
    Filter pred_df to only include peaks that contain at least one caQTL.
    
    Args:
        pred_df: DataFrame with columns ['chr', 'start', 'end', ...]
        caqtl_df: DataFrame with columns ['var.chr', 'var.pos_hg38', ...]
    
    Returns:
        Boolean mask for pred_df rows that contain at least one caQTL
    """
    # Create a list to store boolean masks for each peak
    peak_masks = []
    
    for idx, peak in pred_df.iterrows():
        # Get peak coordinates
        peak_chr = peak['chr']
        peak_start = peak['start']
        peak_end = peak['end']
        
        # Filter caQTLs for the same chromosome
        chrom_caqtls = caqtl_df[caqtl_df['var.chr'] == peak_chr]
        
        if len(chrom_caqtls) == 0:
            peak_masks.append(False)
            continue
        
        # Check if any caQTL position falls within the peak
        caqtl_in_peak = ((chrom_caqtls['var.pos_hg38'] >= peak_start) & 
                         (chrom_caqtls['var.pos_hg38'] <= peak_end))
        
        peak_masks.append(caqtl_in_peak.any())
    
    return pd.Series(peak_masks, index=pred_df.index)

# Apply the filter
peak_contains_caqtl_mask = peaks_contain_caqtls(pred_df, caqtl_df)
pred_df_filtered = pred_df[peak_contains_caqtl_mask]

print(f"Original number of peaks: {len(pred_df)}")
print(f"Number of peaks containing caQTLs: {len(pred_df_filtered)}")
print(f"Percentage of peaks with caQTLs: {len(pred_df_filtered)/len(pred_df)*100:.2f}%")
# %%

# use indices of pred_df_filtered to filter pred_mat and true_mat
pred_mat_filtered = pred_mat.iloc[pred_df_filtered.index, :]
true_mat_filtered = true_mat.iloc[pred_df_filtered.index, :]
# %%

# compute correlation with respect to the rows
per_sample_corrs = []

for sample_id in sample_ids:
    preds = pred_mat_filtered[sample_id].values
    trues = true_mat_filtered[sample_id].values
    r, _ = spearmanr(preds, trues)
    per_sample_corrs.append(r)

per_peak_corrs = []
for i in range(len(pred_mat)):
    preds = pred_mat_filtered.iloc[i, :].values
    trues = true_mat_filtered.iloc[i, :].values
    r, _ = spearmanr(preds, trues)
    per_peak_corrs.append(r)

# %%
print(np.nanmean(per_sample_corrs))
# %%
print(np.nanmean(per_peak_corrs))# %%
# %%
