# %%
import os 
import pandas as pd


# %%
personal_eval_dir = "/oak/stanford/groups/akundaje/abuen/personal_genome/personal_pytorch/output/personal_eval"
results_dict = {}

for sample_id in os.listdir(personal_eval_dir):
    pred_file = f"/oak/stanford/groups/akundaje/abuen/personal_genome/personal_pytorch/output/personal_eval/{sample_id}/fold_0/predict/test/regions.csv"
    
    pred_df = pd.read_csv(pred_file, sep="\t")
    pred_df = pred_df[pred_df["is_peak"] == 1]
    results_dict[sample_id] = pred_df[["pred_count", "true_count"]]

# %%
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

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
    r, _ = pearsonr(preds, trues)
    per_sample_corrs.append(r)

per_peak_corrs = []
for i in range(len(pred_mat)):
    preds = pred_mat.iloc[i, :].values
    trues = true_mat.iloc[i, :].values
    r, _ = pearsonr(preds, trues)
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
plt.xlabel("Pearson correlation")
plt.axvline(x=np.nanmean(per_peak_corrs), color='red', linestyle='--')
plt.show()
# %%


plt.hist(per_sample_corrs, bins=20)
plt.title("Cross-peak correlation (per sample)")
plt.xlabel("Pearson correlation")
plt.axvline(x=np.nanmean(per_sample_corrs), color='red', linestyle='--')
plt.show()
# %%