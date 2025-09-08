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


class PersonalizedGenome:
    """
    A class to handle personalized genomes by applying variants from VCF to reference genome.
    """
    
    def __init__(self, reference_genome: str, vcf_file: Optional[str] = None, sample_id: Optional[str] = None):
        """
        Initialize personalized genome.
        
        Args:
            reference_genome: Path to reference genome fasta file
            vcf_file: Path to VCF/BCF file containing variants
            sample_id: Sample ID to extract genotypes from VCF
        """
        self.reference_genome_path = reference_genome
        self.reference_genome = pyfaidx.Fasta(reference_genome)
        self.vcf_file = vcf_file
        self.sample_id = sample_id
        self.variants_cache = {}
        
        if vcf_file and sample_id:
            self._load_variants()
    
    def _load_variants(self):
        """Load variants from VCF file for the specified sample."""
        print(f"Loading variants from {self.vcf_file} for sample {self.sample_id}")
        
        with pysam.VariantFile(self.vcf_file, "rb") as vcf:
            if self.sample_id not in vcf.header.samples:
                raise ValueError(f"Sample {self.sample_id} not found in VCF file")
            
            for record in vcf.fetch():
                # Get genotype for the sample
                sample_gt = record.samples[self.sample_id]['GT']
                if sample_gt is None or sample_gt == (None, None):
                    continue
                
                # Check if ANY allele is alternate (not just the first)
                has_alt = any(gt is not None and gt > 0 for gt in sample_gt)
                if not has_alt:
                    continue
                    
                chrom = record.chrom
                pos = record.pos - 1  # Convert to 0-based
                ref_allele = record.ref
                alt_alleles = record.alts
                
                # Get all alternate alleles for this sample
                alt_indices = [gt for gt in sample_gt if gt is not None and gt > 0]
                
                # For heterozygous variants, we need to handle both alleles
                if len(alt_indices) == 1 and len(set(sample_gt)) == 2:
                    # Heterozygous: 0/1 or 1/0
                    alt_allele = alt_alleles[alt_indices[0] - 1]
                    is_heterozygous = True
                elif len(alt_indices) == 2:
                    # Homozygous alternate: 1/1
                    alt_allele = alt_alleles[alt_indices[0] - 1]  # Use first alt allele
                    is_heterozygous = False
                else:
                    continue
                
                # Store variant information
                if chrom not in self.variants_cache:
                    self.variants_cache[chrom] = []
                self.variants_cache[chrom].append({
                    'pos': pos,
                    'ref': ref_allele,
                    'alt': alt_allele,
                    'is_heterozygous': is_heterozygous,
                    'genotype': sample_gt
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
        ref_seq = str(self.reference_genome[chrom][start:end])
        
        if not self.vcf_file or chrom not in self.variants_cache:
            return ref_seq, ref_seq
        
        # Create two haplotypes
        first_hap_seq = list(ref_seq)
        second_hap_seq = list(ref_seq)
        
        for variant in self.variants_cache[chrom]:
            var_pos = variant['pos']
            if start <= var_pos < end:
                # Adjust position relative to our region
                rel_pos = var_pos - start
                
                # Apply the variant
                ref_allele = variant['ref']
                alt_allele = variant['alt']
                genotype = variant['genotype']
                
                # Check if reference allele matches
                if rel_pos + len(ref_allele) <= len(ref_seq):
                    region_ref = ref_seq[rel_pos:rel_pos + len(ref_allele)]
                    if region_ref == ref_allele:
                        # Apply to first haplotype
                        if genotype[0] == 1:
                            if len(alt_allele) == len(ref_allele):
                                for i, base in enumerate(alt_allele):
                                    first_hap_seq[rel_pos + i] = base
                        
                        # Apply to second haplotype
                        if genotype[1] == 1:
                            if len(alt_allele) == len(ref_allele):
                                for i, base in enumerate(alt_allele):
                                    second_hap_seq[rel_pos + i] = base
        
        return ''.join(first_hap_seq), ''.join(second_hap_seq)
    
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


class MultiSamplePersonalizedGenome:
    """
    A class to handle multiple personalized genomes from the same VCF file.
    """
    
    def __init__(self, reference_genome: str, vcf_file: str, sample_ids: List[str]):
        """
        Initialize multi-sample personalized genome.
        
        Args:
            reference_genome: Path to reference genome fasta file
            vcf_file: Path to VCF/BCF file containing variants
            sample_ids: List of sample IDs to extract genotypes from VCF
        """
        self.reference_genome_path = reference_genome
        self.vcf_file = vcf_file
        self.sample_ids = sample_ids
        self.personalized_genomes = {}
        
        # Create PersonalizedGenome instance for each sample
        for sample_id in sample_ids:
            self.personalized_genomes[sample_id] = PersonalizedGenome(
                reference_genome, vcf_file, sample_id
            )
    
    def get_sequence(self, sample_id: str, chrom: str, start: int, end: int) -> str:
        """
        Get personalized sequence for a specific sample and genomic region.
        """
        return self.personalized_genomes[sample_id].get_sequence(chrom, start, end)
    
    def get_sequence_with_haplotypes(self, sample_id: str, chrom: str, start: int, end: int) -> Tuple[str, str]:
        """
        Get personalized sequence for a specific sample and genomic region, returning both haplotypes.
        """
        return self.personalized_genomes[sample_id].get_sequence_with_haplotypes(chrom, start, end)
    
    def get_all_samples_sequences(self, chrom: str, start: int, end: int) -> List[Tuple[str, str]]:
        """
        Get personalized sequences for all samples for a genomic region.
        
        Returns:
            List of (haplotype1, haplotype2) tuples for each sample
        """
        return [
            self.get_sequence_with_haplotypes(sample_id, chrom, start, end)
            for sample_id in self.sample_ids
        ]
    
    def close(self):
        """Close all PersonalizedGenome instances."""
        for genome in self.personalized_genomes.values():
            genome.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


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


def get_multi_sample_personalized_sequences(peaks_df: pd.DataFrame, 
                                          multi_genome: MultiSamplePersonalizedGenome, 
                                          width: int) -> Dict[str, np.ndarray]:
    """
    Get personalized sequences for peak regions from multiple samples, averaging haplotypes.
    
    Args:
        peaks_df: DataFrame with chr, start, summit columns
        multi_genome: MultiSamplePersonalizedGenome instance
        width: Width of sequence to extract
    
    Returns:
        Dictionary mapping sample_id to N x L x 4 numpy arrays of averaged one-hot encodings
    """
    sample_sequences = {}
    
    for sample_id in multi_genome.sample_ids:
        first_vals = []
        second_vals = []
        
        for p_i, r in tqdm(peaks_df.iterrows(), total=len(peaks_df), 
                          desc=f"Processing personalized sequences for {sample_id}"):
            chromo = r['chr']
            summit = r['start'] + r['summit']
            start = summit - width // 2
            end = summit + width // 2
            
            # Get both haplotypes for this sample
            first_hap, second_hap = multi_genome.get_sequence_with_haplotypes(sample_id, chromo, start, end)
            
            first_vals.append(first_hap.upper())
            second_vals.append(second_hap.upper())
        
        # Convert to one-hot and average
        first_vals = dna_to_one_hot(first_vals)
        second_vals = dna_to_one_hot(second_vals)
        vals = np.add(first_vals, second_vals) / 2
        
        sample_sequences[sample_id] = vals
    
    return sample_sequences
