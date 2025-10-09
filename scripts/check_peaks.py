# %%
import pandas as pd
import os 
from tqdm import tqdm

# %%
sequence_length = 2114
consensus_peaks = "/oak/stanford/groups/akundaje/abuen/personal_genome/personal_pytorch/data/consensus_peaks/overlap.optimal_peak.narrowPeak.merged.bed"
peaks = pd.read_csv(consensus_peaks, sep="\t", header=None)
peaks.columns = ["chrom", "start", "end"]

# Extend each peak to exactly 2114 base pairs
extended_peaks = []
for _, peak in peaks.iterrows():
    chrom = str(peak["chrom"].split("chr")[1])
    start = int(peak["start"])
    end = int(peak["end"])
    
    # # Calculate current peak length
    # current_length = end - start
    
    # # Calculate how much to extend on each side
    # extension_needed = sequence_length - current_length
    # extension_per_side = extension_needed // 2
    
    # # Handle odd extension by adding 1 to the right side
    # left_extension = extension_per_side
    # right_extension = extension_needed - left_extension
    
    # # Calculate new start and end positions
    # new_start = start - left_extension
    # new_end = end + right_extension
    
    extended_peaks.append({
        "chrom": chrom,
        "start": start,
        "end": end
    })

# Create extended peaks DataFrame
extended_peaks_df = pd.DataFrame(extended_peaks)

# Create a set of genomic positions for efficient lookup
peak_positions = set()
for _, peak in extended_peaks_df.iterrows():
    chrom = peak["chrom"]
    for pos in range(int(peak["start"]), int(peak["end"]) + 1):
        peak_positions.add((chrom, pos))

print(f"Extended {len(peaks)} peaks to {sequence_length} base pairs each")
print(f"Loaded {len(peak_positions)} genomic positions from extended consensus peaks")

# %%

individual_vcf_dir = "/oak/stanford/groups/akundaje/ziwei75/african-omics/data/genotype/individual_vcf"

# Dictionary to store counts for each user
user_variant_counts = {}

for vcf_file in tqdm(os.listdir(individual_vcf_dir)):
    if vcf_file.endswith(".csv"):
        vcf_path = os.path.join(individual_vcf_dir, vcf_file)
        vcf_df = pd.read_csv(vcf_path)
        
        # Subset to first 5 columns and last column
        vcf_df = vcf_df.iloc[:, list(range(5)) + [-1]]
        vcf_df.columns = ["chrom", "pos", "id", "ref", "alt", "genotype"]
        
        # Extract user ID from filename (assuming format like "user_id.csv")
        user_id = vcf_file.replace(".csv", "")
        
        # Count variants in peaks for this user
        variants_in_peaks = 0
        
        for _, row in vcf_df.iterrows():
            chrom = str(row["chrom"])
            pos = int(row["pos"])
            genotype = str(row["genotype"])

            # Check if position is in any peak
            if (chrom, pos) in peak_positions:
                # Check if genotype indicates alternate allele (1|1, 1|0, or 0|1)
                if genotype.startswith(("1|1", "1|0", "0|1")):
                    variants_in_peaks += 1
        
        user_variant_counts[user_id] = variants_in_peaks
        print(f"User {user_id}: {variants_in_peaks} variants in peaks")

# Count users with at least one variant in peaks
users_with_variants = sum(1 for count in user_variant_counts.values() if count > 0)
total_users = len(user_variant_counts)

print(f"\nSummary:")
print(f"Total users: {total_users}")
print(f"Users with at least one variant in peaks: {users_with_variants}")
print(f"Percentage: {users_with_variants/total_users*100:.2f}%")

# %%
