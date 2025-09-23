"""
Personalized Genome Implementation for ChromBPNet

This module provides functionality for creating personalized genomes by applying
variants from VCF files to reference genomes, enabling personalized model training.
"""

import numpy as np
import pandas as pd
import pysam
from typing import Dict, List, Optional, Tuple
import pyfaidx
from tqdm import tqdm
import os


class PersonalizedGenome:
    """
    A class to handle personalized genomes by applying variants from VCF to reference genome.
    """
    
    def __init__(self, reference_genome: str, vcf_dir: Optional[str] = None, sample_id: Optional[str] = None, rename_map: Optional[str] = None):
        """
        Initialize personalized genome.
        
        Args:
            reference_genome: Path to reference genome fasta file
            vcf_dir: Path to directory containing VCF/BCF files
            sample_id: Sample ID to extract genotypes from VCF
        """
        self.reference_genome_path = reference_genome
        self.reference_genome = pyfaidx.Fasta(reference_genome)
        self.sample_id = sample_id
        self.rename_map = {}
        self.variants_cache = {}

        if rename_map:
            with open(rename_map, 'r') as f:
                for line in f:
                    old_name, new_name = line.strip().split('\t')
                    self.rename_map[old_name] = new_name
        self.vcf_file = os.path.join(vcf_dir, self.rename_map.get(self.sample_id, self.sample_id) + '.csv')

        if self.vcf_file and self.sample_id and self.rename_map:
            self._load_variants()
    
    def _load_variants(self):
        """Load variants from VCF file for the specified sample."""
        print(f"Loading variants from {self.vcf_file} for sample {self.sample_id}")

        # Load VCF data - CSV with columns like CHROM, POS, REF, ALT, and sample columns
        vcf_df = pd.read_csv(self.vcf_file)
                
        # Process each variant
        for _, variant in tqdm(vcf_df.iterrows(), total=len(vcf_df), desc="Processing variants"):
            chrom = variant['CHROM']    
            pos = variant['POS'] - 1  # Convert to 0-based
            ref_allele = variant['REF']
            alt_allele = variant['ALT']
            
            if len(ref_allele) > 1 or len(alt_allele) > 1:
                continue
            
            # Extract genotype for the sample
            if self.sample_id and self.sample_id in variant.index:
                genotype_str = variant[self.sample_id]
            else:
                genotype_str = variant.iloc[-1]
            
            # Parse genotype (assuming format like "0|1:...")
            genotype_parts = genotype_str.split(":")[0]
            if "|" in genotype_parts:
                first_hap, second_hap = genotype_parts.split("|")
                first_hap, second_hap = int(first_hap), int(second_hap)
                genotype = (first_hap, second_hap)
            elif "/" in genotype_parts:
                first_hap, second_hap = genotype_parts.split("/")
                first_hap, second_hap = int(first_hap), int(second_hap)
                genotype = (first_hap, second_hap)
            else:
                # Skip if genotype format is not recognized
                continue
            
            # Only store variants that have alternate alleles
            if genotype[0] == 1 or genotype[1] == 1:
                # Store variant information
                if chrom not in self.variants_cache:
                    self.variants_cache[chrom] = []
                self.variants_cache[chrom].append({
                    'pos': pos,
                    'ref': ref_allele,
                    'alt': alt_allele,
                    'genotype': genotype
                })
        
        # Sort variants by position for each chromosome
        for chrom in self.variants_cache:
            self.variants_cache[chrom].sort(key=lambda x: x['pos'])
        
        print(f"Loaded {sum(len(vars) for vars in self.variants_cache.values())} variants")
    
    def get_sequence(self, chrom: str, start: int, end: int) -> str:
        """
        Get personalized sequence for a genomic region.
        
        Args:
            chrom: Chromosome name
            start: Start position (0-based)
            end: End position (0-based)
        
        Returns:
            Personalized DNA sequence string
        """
        # Get reference sequence
        ref_seq = str(self.reference_genome[chrom][start:end])
        
        if not self.vcf_file or chrom not in self.variants_cache:
            return ref_seq
        
        # Apply variants in the region
        personalized_seq = list(ref_seq)
        
        for variant in self.variants_cache[chrom]:
            var_pos = variant['pos']
            if start <= var_pos < end:
                # Adjust position relative to our region
                rel_pos = var_pos - start
                
                # Apply the variant
                ref_allele = variant['ref']
                alt_allele = variant['alt']
                
                # Check if reference allele matches
                if rel_pos + len(ref_allele) <= len(personalized_seq):
                    region_ref = ''.join(personalized_seq[rel_pos:rel_pos + len(ref_allele)])
                    if region_ref == ref_allele:
                        # Replace reference with alternate
                        if len(alt_allele) == len(ref_allele):
                            # Simple substitution
                            for i, base in enumerate(alt_allele):
                                personalized_seq[rel_pos + i] = base
                        elif len(alt_allele) > len(ref_allele):
                            # Insertion
                            for i, base in enumerate(alt_allele):
                                if i < len(ref_allele):
                                    personalized_seq[rel_pos + i] = base
                                else:
                                    personalized_seq.insert(rel_pos + len(ref_allele), base)
                        else:
                            # Deletion
                            for i in range(len(ref_allele) - len(alt_allele)):
                                if rel_pos + len(alt_allele) < len(personalized_seq):
                                    personalized_seq.pop(rel_pos + len(alt_allele))
        
        return ''.join(personalized_seq)
    
    def get_sequence_with_haplotypes(self, chrom: str, start: int, end: int) -> Tuple[str, str]:
        """
        Get personalized sequence for a genomic region, returning both haplotypes.
        
        Args:
            chrom: Chromosome name
            start: Start position (0-based)
            end: End position (0-based)
        
        Returns:
            Tuple of (haplotype1, haplotype2) DNA sequence strings
        """
        # Get reference sequence
        sequence = str(self.reference_genome[chrom][start:end])
        width = end - start
                
        # Get variants in the region
        variants = []
        if chrom in self.variants_cache:
            for variant in self.variants_cache[chrom]:
                pos = variant['pos']
                if start < pos < end:
                    variants.append(variant)
        
        # Initialize haplotypes
        first_hap_seq = sequence
        second_hap_seq = sequence
        
        # Apply variants
        for variant in variants:
            pos = variant['pos']
            ref = variant['ref']
            alt = variant['alt']
            genotype = variant['genotype']
            
            # Skip INDELs for now
            if len(ref) > 1 or len(alt) > 1:
                continue
            
            first_hap, second_hap = genotype[0], genotype[1]
            
            # Adjust position relative to our region (0-based within region)
            rel_pos = pos - start
            assert rel_pos >= 0, f"Relative position {rel_pos} should be >= 0"
            assert rel_pos < width, f"Relative position {rel_pos} should be < {width}"
            
            # Verify reference allele matches
            region_ref = sequence[rel_pos:rel_pos + len(ref)]
            assert region_ref.upper() == ref.upper(), f"Reference mismatch: {region_ref} != {ref}"
            
            # Apply variant to first haplotype
            if first_hap == 1:
                first_hap_seq = first_hap_seq[:rel_pos] + alt + first_hap_seq[rel_pos + len(ref):]
            
            # Apply variant to second haplotype
            if second_hap == 1:
                second_hap_seq = second_hap_seq[:rel_pos] + alt + second_hap_seq[rel_pos + len(ref):]
            
            # Verify sequence lengths remain correct
            assert len(first_hap_seq) == width, f"First haplotype length {len(first_hap_seq)} != {width}"
            assert len(second_hap_seq) == width, f"Second haplotype length {len(second_hap_seq)} != {width}"
        
        return first_hap_seq.upper(), second_hap_seq.upper()
    
    def close(self):
        """Close the reference genome."""
        self.reference_genome.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def dna_to_one_hot(sequences: List[str]) -> np.ndarray:
    """
    Convert DNA sequences to one-hot encoding.
    
    Args:
        sequences: List of DNA sequence strings
    
    Returns:
        N x L x 4 numpy array of one-hot encodings
    """
    n_seqs = len(sequences)
    seq_len = len(sequences[0])
    
    # Initialize one-hot array
    one_hot = np.zeros((n_seqs, seq_len, 4), dtype=np.float32)
    
    # Mapping from nucleotide to index
    nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    for i, seq in enumerate(sequences):
        for j, nuc in enumerate(seq.upper()):
            if nuc in nuc_to_idx:
                one_hot[i, j, nuc_to_idx[nuc]] = 1.0
    
    return one_hot


def get_personalized_sequences(peaks_df: pd.DataFrame, personalized_genome: PersonalizedGenome, 
                             width: int) -> np.ndarray:
    """
    Get personalized sequences for peak regions, averaging haplotypes.
    
    Args:
        peaks_df: DataFrame with chr, start, summit columns
        personalized_genome: PersonalizedGenome instance
        width: Width of sequence to extract
    
    Returns:
        N x L x 4 numpy array of averaged one-hot encodings
    """
    first_vals = []
    second_vals = []
    
    for p_i, r in tqdm(peaks_df.iterrows(), total=len(peaks_df), desc="Processing personalized sequences"):
        chromo = r['chr']
        summit = r['start'] + r['summit']
        start = summit - width // 2
        end = summit + width // 2
        
        # Get both haplotypes
        first_hap, second_hap = personalized_genome.get_sequence_with_haplotypes(chromo, start, end)
        
        first_vals.append(first_hap.upper())
        second_vals.append(second_hap.upper())
    
    # Convert to one-hot and average
    first_vals = dna_to_one_hot(first_vals)
    second_vals = dna_to_one_hot(second_vals)
    vals = np.add(first_vals, second_vals) / 2
    
    return vals
