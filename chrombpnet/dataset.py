# Author: Lei Xiong <jsxlei@gmail.com>


"""
Data loading and processing module for genomic data.

This module provides classes for loading and processing genomic data for training and evaluation.
It handles the loading of genomic regions, their corresponding sequences, and various data augmentation techniques.
"""

from functools import cached_property
from time import time

# Third-party imports
import torch
import numpy as np
import pandas as pd
import lightning as L


from .data_utils import load_region_df, load_data, random_crop, crop_revcomp_augment, get_cts


class DataModule(L.LightningDataModule):
    """DataModule for loading and processing genomic data for training and evaluation.
    
    This module handles the loading of genomic regions, their corresponding sequences,
    and various data augmentation techniques. It supports different data types:
    - Profile data: For single-region analysis
    - Long-range data: For analyzing interactions between regions
    
    The module implements different sampling strategies for training, validation and testing:
    - Train: peaks + negative_sampling_ratio (0.1) of negatives, sampled each epoch
    - Val: peaks + negatives_sampling_ratio (1) of negatives, sampled once and fixed
    - Test: peaks + negatives, no sampling
    
    Attributes:
        config: Configuration object containing data loading parameters
        dataset_class: The dataset class to use (ChromBPNetBatchGenerator or LongRangeDataset)
        peaks: DataFrame containing peak regions
        negatives: DataFrame containing negative regions
        data: Combined DataFrame of peaks and negatives
        train_chroms: List of chromosomes used for training
        val_chroms: List of chromosomes used for validation
        test_chroms: List of chromosomes used for testing
    """
    
    def __init__(self, config):
        """Initialize the DataModule.
        
        Args:
            config: Configuration object containing data loading parameters
        """
        super().__init__()
        self.config = config

        # Set dataset class based on data type and training mode
        if self.config.data_type == 'profile':
            if self.config.training_mode == 'multi_sample_sequential':
                self.dataset_class = MultiSampleSequentialDataset
            elif self.config.training_mode == 'multi_sample_extended_loss':
                self.dataset_class = MultiSampleExtendedLossDataset
            else:
                self.dataset_class = ChromBPNetDataset
        else:
            raise ValueError(f'Invalid data type: {self.config.data_type}')

        # Load and process data
        self._load_regions()
        self._setup_chromosomes()
        self._split_data()

    def _load_regions(self):
        """Load peak and negative regions from files."""
        self.peaks = load_region_df(
            self.config.peaks, 
            chrom_sizes=self.config.chrom_sizes,
            in_window=self.config.in_window,
            shift=self.config.shift,
            is_peak=True
        )
        
        if self.config.negatives is not None:
            self.negatives = load_region_df(
                self.config.negatives,
                chrom_sizes=self.config.chrom_sizes,
                in_window=self.config.in_window,
                shift=self.config.shift,
                is_peak=False
            )
            self.data = pd.concat([self.peaks, self.negatives], ignore_index=True)
        else:
            self.negatives = None
            self.data = self.peaks



    def _setup_chromosomes(self):
        """Setup chromosome lists for training, validation and testing."""
        self.train_chroms = [i for i in self.config.training_chroms if i not in self.config.exclude_chroms]
        self.val_chroms = [i for i in self.config.validation_chroms if i not in self.config.exclude_chroms]
        self.test_chroms = [i for i in self.config.test_chroms if i not in self.config.exclude_chroms]
        self.chroms = self.train_chroms + self.val_chroms + self.test_chroms

    def _split_data(self):
        """Split data into training, validation and testing sets."""
        self.train_val = self.data[self.data.iloc[:, 0].isin(self.val_chroms+self.train_chroms)].reset_index(drop=True)
        self.train_data = self.data[self.data.iloc[:, 0].isin(self.train_chroms)].reset_index(drop=True)

        self.val_data = self.data[self.data.iloc[:, 0].isin(self.val_chroms)].reset_index(drop=True)
        self.test_data = self.data[self.data.iloc[:, 0].isin(self.test_chroms)].reset_index(drop=True)

    def setup(self, stage='fit'):
        print('Setting up data...'); t0 = time()

        config = self.config

        if stage == 'fit':
            train_peaks, train_nonpeaks = split_peak_and_nonpeak(self.train_data)
            val_peaks, val_nonpeaks = split_peak_and_nonpeak(self.val_data)

            self.train_dataset = self.dataset_class(
                peak_regions=train_peaks,
                nonpeak_regions=train_nonpeaks,
                genome_fasta=config.fasta,
                inputlen=config.in_window,                                        
                outputlen=config.out_window,
                max_jitter=config.shift,
                negative_sampling_ratio=config.negative_sampling_ratio,
                cts_bw_file=config.bigwig,
                add_revcomp=True,
                return_coords=False, #return_coords,
                shuffle_at_epoch_start=False, #shuffle_at_epoch_start
                vcf_file=config.vcf_file,
                sample_id=config.sample_id,
                sample_ids=config.sample_ids,
                training_mode=config.training_mode,
            )
            self.val_dataset = self.dataset_class(
                peak_regions=val_peaks,
                nonpeak_regions=val_nonpeaks,
                genome_fasta=config.fasta,
                inputlen=config.in_window,                                        
                outputlen=config.out_window,
                max_jitter=0,
                negative_sampling_ratio=config.negative_sampling_ratio,
                cts_bw_file=config.bigwig,
                add_revcomp=False,
                return_coords=False,
                shuffle_at_epoch_start=False,
                vcf_file=config.vcf_file,
                sample_id=config.sample_id,
                sample_ids=config.sample_ids,
                training_mode=config.training_mode,
            )
        elif stage == 'test':
            test_peaks, test_nonpeaks = split_peak_and_nonpeak(self.test_data)
            self.test_dataset = self.dataset_class(
                peak_regions=test_peaks,
                nonpeak_regions=test_nonpeaks,  
                genome_fasta=config.fasta,
                inputlen=config.in_window,                                        
                outputlen=config.out_window,
                max_jitter=0,
                negative_sampling_ratio=-1,
                cts_bw_file=config.bigwig,
                add_revcomp=False,
                return_coords=False,
                shuffle_at_epoch_start=False,
                vcf_file=config.vcf_file,
                sample_id=config.sample_id,
                sample_ids=config.sample_ids,
                training_mode=config.training_mode,
            )

        print(f'Data setup complete in {time() - t0:.2f} seconds')

    @cached_property
    def median_count(self):
        import pyBigWig
        ## Calculate median count to get weight of count loss
        self.train_val_subsampled = concat_peaks_and_subsampled_negatives(self.train_val, negative_sampling_ratio=self.config.negative_sampling_ratio)
        counts_subsampled = get_cts(self.train_val_subsampled, pyBigWig.open(self.config.bigwig), self.config.out_window).sum(-1)
        # counts_subsampled = extract_loci(self.train_val_subsampled, self.config.bigwig, width=self.config.out_windo, w, out='bigwig', shift=0, pool_size=64).sum(-1)
        return np.median(counts_subsampled)



    def train_dataloader(self):
        # Handle sequential training sample switching
        if isinstance(self.train_dataset, MultiSampleSequentialDataset):
            # For sequential training, set up the dataset for the next sample
            if hasattr(self, '_epoch_count'):
                self._epoch_count += 1
            else:
                self._epoch_count = 0
            
            if self.config.sample_ids:
                sample_idx = self._epoch_count % len(self.config.sample_ids)
                self.train_dataset.set_current_sample(sample_idx)
        else:
            self.train_dataset.crop_revcomp_data()
        
        return torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True, 
            drop_last=False,
            num_workers=self.config.num_workers, 
            # pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=self.config.num_workers, 
            # pin_memory=True
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=self.config.num_workers, 
        )

    def negative_dataloader(self):
        self.negative_dataset = self.dataset_class(
            peak_regions=self.negatives,
            nonpeak_regions=None,
            genome_fasta=self.config.fasta,
            batch_size=self.config.batch_size,
            inputlen=self.config.in_window,
            outputlen=self.config.out_window,
            max_jitter=0,
            negative_sampling_ratio=-1,
            cts_bw_file=self.config.bigwig,
            add_revcomp=False,
            return_coords=False,
            shuffle_at_epoch_start=False,
            vcf_file=self.config.vcf_file,
            sample_id=self.config.sample_id,
            sample_ids=self.config.sample_ids,
            training_mode=self.config.training_mode,
        )
        return torch.utils.data.DataLoader(
            self.negative_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=self.config.num_workers, 
        )
    
    def chrom_dataloader(self, chrom='chr1', negative_sampling_ratio=-1):

        dataset = self.chrom_dataset(chrom=chrom, negative_sampling_ratio=negative_sampling_ratio)

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        ), dataset


    def chrom_dataset(self, chrom='chr1', negative_sampling_ratio=-1):
        if isinstance(chrom, str):
            if chrom in ['train', 'val', 'test']:
                chrom = getattr(self, f'{chrom}_chroms')
                
            elif chrom == 'all':
                chrom = self.chroms
            else:
                chrom = [chrom]

        regions = self.data[self.data.iloc[:, 0].isin(chrom)].reset_index(drop=True)
        peaks, nonpeaks = split_peak_and_nonpeak(regions)

        dataset = self.dataset_class(
            peak_regions=peaks,
            nonpeak_regions=nonpeaks,
            genome_fasta=self.config.fasta,
            inputlen=self.config.in_window,
            outputlen=self.config.out_window,
            max_jitter=0,
            negative_sampling_ratio=-1,
            cts_bw_file=self.config.bigwig,
            add_revcomp=False,
            return_coords=False,
            shuffle_at_epoch_start=False,
            debug=self.config.debug,
            vcf_file=self.config.vcf_file,
            sample_id=self.config.sample_id,
            sample_ids=self.config.sample_ids,
            training_mode=self.config.training_mode,
        )
        return dataset





def split_peak_and_nonpeak(data):
    data['is_peak'] = data['is_peak'].astype(int).astype(bool)
    non_peaks = data[~data['is_peak']].copy()
    if not len(non_peaks) > 0:
        non_peaks = None
    peaks = data[data['is_peak']].copy()
    return peaks, non_peaks


def subsample_nonpeak_data(nonpeak_seqs, nonpeak_cts, nonpeak_coords, peak_data_size, negative_sampling_ratio):
    #Randomly samples a portion of the non-peak data to use in training
    num_nonpeak_samples = int(negative_sampling_ratio * peak_data_size)
    nonpeak_indices_to_keep = np.random.choice(len(nonpeak_seqs), size=min(num_nonpeak_samples, len(nonpeak_seqs)), replace=False)
    nonpeak_seqs = nonpeak_seqs[nonpeak_indices_to_keep]
    nonpeak_cts = nonpeak_cts[nonpeak_indices_to_keep]
    nonpeak_coords = nonpeak_coords[nonpeak_indices_to_keep]
    return nonpeak_seqs, nonpeak_cts, nonpeak_coords


def concat_peaks_and_subsampled_negatives(peaks, negatives=None, negative_sampling_ratio=0.1):
    if negatives is None:
        peaks, negatives = split_peak_and_nonpeak(peaks)
        # print(peaks.shape, negatives.shape)

    if negatives is not None and len(negatives) > len(peaks) * negative_sampling_ratio and negative_sampling_ratio > 0:
        negatives = negatives.sample(n=int(negative_sampling_ratio * len(peaks)), replace=False)
        
        data = pd.concat([peaks, negatives], ignore_index=True)
    else:
        data = peaks
    return data


def crop_revcomp_data(
    peak_seqs, peak_cts, peak_coords, 
    nonpeak_seqs=None, nonpeak_cts=None, nonpeak_coords=None, 
    inputlen=2114, outputlen=1000, add_revcomp=False, negative_sampling_ratio=0.1, shuffle=False):
    """Apply random cropping and reverse complement augmentation to the data.
        
        This method:
        1. Randomly crops peak data to inputlen and outputlen
        2. Samples negative examples according to negative_sampling_ratio
        3. Applies reverse complement augmentation if enabled
        4. Shuffles data if shuffle_at_epoch_start is True
    """
    if (peak_seqs is not None) and (nonpeak_seqs is not None):
        # Crop peak data
        cropped_peaks, cropped_cnts, cropped_coords = random_crop(
            peak_seqs, peak_cts, inputlen, outputlen, peak_coords
        )
        
        # Sample negative examples
        if negative_sampling_ratio > 0:
            sampled_nonpeak_seqs, sampled_nonpeak_cts, sampled_nonpeak_coords = subsample_nonpeak_data(
                nonpeak_seqs, nonpeak_cts, nonpeak_coords,
                len(peak_seqs), negative_sampling_ratio
            )
            seqs = np.vstack([cropped_peaks, sampled_nonpeak_seqs])
            cts = np.vstack([cropped_cnts, sampled_nonpeak_cts])
            coords = np.vstack([cropped_coords, sampled_nonpeak_coords])
        else:
            seqs = np.vstack([cropped_peaks, nonpeak_seqs])
            cts = np.vstack([cropped_cnts, nonpeak_cts])
            coords = np.vstack([cropped_coords, nonpeak_coords])

    elif peak_seqs is not None:
        # Only peak data
        cropped_peaks, cropped_cnts, cropped_coords = random_crop(
            peak_seqs, peak_cts, inputlen, outputlen, peak_coords
        )
        seqs = cropped_peaks
        cts = cropped_cnts
        coords = cropped_coords

    elif nonpeak_seqs is not None:
        # Only non-peak data
        seqs = nonpeak_seqs
        cts = nonpeak_cts
        coords = nonpeak_coords
    else:
        raise ValueError("Both peak and non-peak arrays are empty")

    # Apply augmentation
    seqs, cts, coords = crop_revcomp_augment(
        seqs, cts, coords, inputlen, outputlen,
        add_revcomp, shuffle=shuffle
    )
    # self.regions = pd.DataFrame(self.cur_coords, columns=['chrom', 'start', 'forward_or_reverse', 'is_peak'])
    # print('Regions', self.regions['is_peak'].value_counts())
    return seqs, cts, coords


def debug_subsample(peak_regions, chrom=None):
    if peak_regions is None:
        return None
    
    if chrom is None:
        chrom = peak_regions['chr'].unique()[0]


    peak_regions = peak_regions[peak_regions['chr'] == chrom]
    print('debugging on ', chrom, 'shape', peak_regions.shape)
    return peak_regions.reset_index(drop=True)

class ChromBPNetDataset(torch.utils.data.Dataset):
    """Generator for genomic sequence data with random cropping and reverse complement augmentation.
    
    This generator randomly crops (=jitter) and applies reverse complement augmentation to training examples
    for every epoch. It handles both peak and non-peak regions, with configurable sampling ratios.
    
    Attributes:
        peak_seqs: Array of peak sequences
        nonpeak_seqs: Array of non-peak sequences
        peak_cts: Array of peak counts
        nonpeak_cts: Array of non-peak counts
        peak_coords: Array of peak coordinates
        nonpeak_coords: Array of non-peak coordinates
        negative_sampling_ratio: Ratio of negative samples to use
        inputlen: Length of input sequences
        outputlen: Length of output sequences
        batch_size: Size of batches
        add_revcomp: Whether to add reverse complement augmentation
        return_coords: Whether to return coordinates
        shuffle_at_epoch_start: Whether to shuffle at epoch start
    """
    
    def __init__(
            self, 
            peak_regions, 
            nonpeak_regions, 
            genome_fasta, 
            inputlen=2114, 
            outputlen=1000, 
            max_jitter=0, 
            negative_sampling_ratio=0.1, 
            cts_bw_file=None, 
            add_revcomp=False, 
            return_coords=False,    
            shuffle_at_epoch_start=False, 
            debug=False,
            vcf_file=None,
            sample_id=None,
            sample_ids=None,
            training_mode='standard',
            **kwargs
    ):
        """Initialize the generator.
        
        Args:
            peak_regions: DataFrame containing peak regions
            nonpeak_regions: DataFrame containing non-peak regions
            genome_fasta: Path to genome FASTA file
            batch_size: Size of batches
            inputlen: Length of input sequences
            outputlen: Length of output sequences
            max_jitter: Maximum jitter for random cropping
            negative_sampling_ratio: Ratio of negative samples to use
            cts_bw_file: Path to bigwig file containing counts
            add_revcomp: Whether to add reverse complement augmentation
            return_coords: Whether to return coordinates
            shuffle_at_epoch_start: Whether to shuffle at epoch start
            **kwargs: Additional keyword arguments
        """
        if debug:
            peak_regions = debug_subsample(peak_regions)
            nonpeak_regions = debug_subsample(nonpeak_regions)

 
        # Load data
        loaded_data = load_data(
            peak_regions, nonpeak_regions, genome_fasta, cts_bw_file, inputlen, outputlen, max_jitter,
            vcf_file=vcf_file, sample_id=sample_id, sample_ids=sample_ids, training_mode=training_mode
        )
        
        # Handle multi-sample or single-sample data loading
        if isinstance(loaded_data, dict) and 'training_mode' in loaded_data:
            # Multi-sample case
            self.multi_sample_data = loaded_data
            peak_seqs = loaded_data['train_peaks_seqs']
            peak_cts = loaded_data['train_peaks_cts'] 
            peak_coords = loaded_data['train_peaks_coords']
            nonpeak_seqs = loaded_data['train_nonpeaks_seqs']
            nonpeak_cts = loaded_data['train_nonpeaks_cts']
            nonpeak_coords = loaded_data['train_nonpeaks_coords']
        else:
            # Single-sample case
            self.multi_sample_data = None
            peak_seqs, peak_cts, peak_coords, nonpeak_seqs, nonpeak_cts, nonpeak_coords = loaded_data

        # Store data
        self.peak_seqs, self.nonpeak_seqs = peak_seqs, nonpeak_seqs
        self.peak_cts, self.nonpeak_cts = peak_cts, nonpeak_cts
        self.peak_coords, self.nonpeak_coords = peak_coords, nonpeak_coords

        # Store parameters
        self.negative_sampling_ratio = negative_sampling_ratio
        self.inputlen = inputlen
        self.outputlen = outputlen
        self.add_revcomp = add_revcomp
        self.return_coords = return_coords
        self.shuffle_at_epoch_start = shuffle_at_epoch_start
        self.max_jitter = max_jitter
        self.genome_fasta = genome_fasta
        self.cts_bw_file = cts_bw_file
        self.training_mode = training_mode
        self.sample_ids = sample_ids

        if nonpeak_regions is not None:
            self.regions = pd.concat([peak_regions, nonpeak_regions], ignore_index=True)
        else:
            self.regions = peak_regions
        # Initialize data
        self.crop_revcomp_data()

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.cur_seqs)

    def crop_revcomp_data(self):
        """Apply random cropping and reverse complement augmentation to the data.
        
        This method:
        1. Randomly crops peak data to inputlen and outputlen
        2. Samples negative examples according to negative_sampling_ratio
        3. Applies reverse complement augmentation if enabled
        4. Shuffles data if shuffle_at_epoch_start is True
        """
        self.cur_seqs, self.cur_cts, self.cur_coords = crop_revcomp_data(
            self.peak_seqs, self.peak_cts, self.peak_coords,
            self.nonpeak_seqs, self.nonpeak_cts, self.nonpeak_coords,
            self.inputlen, self.outputlen, self.add_revcomp, self.negative_sampling_ratio, self.shuffle_at_epoch_start
        )

    def _get_adj(self):
        """Get adjacency matrix for the data."""
        pass


    def __getitem__(self, idx):
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to get
            
        Returns:
            Dictionary containing:
                - onehot_seq: One-hot encoded sequence
                - profile: Profile data
        """
        if self.training_mode == 'multi_sample_sequential':
            # For sequential training, we need to return data for specific sample in this epoch
            # This will be handled by the MultiSampleSequentialDataset
            return {
                'onehot_seq': self.cur_seqs[idx].astype(np.float32).transpose(),
                'profile': self.cur_cts[idx].astype(np.float32),
            }
        elif self.training_mode == 'multi_sample_extended_loss':
            # For extended loss training, we return all samples for this region
            return {
                'onehot_seq': self.cur_seqs[idx].astype(np.float32).transpose(),
                'profile': self.cur_cts[idx].astype(np.float32),
                'all_samples_seq': getattr(self, 'cur_all_samples_seqs', [self.cur_seqs[idx]])[idx],
                'all_samples_profile': getattr(self, 'cur_all_samples_cts', [self.cur_cts[idx]])[idx],
            }
        else:
            return {
                'onehot_seq': self.cur_seqs[idx].astype(np.float32).transpose(),
                'profile': self.cur_cts[idx].astype(np.float32),
            }


class MultiSampleSequentialDataset(ChromBPNetDataset):
    """
    Dataset for multi-sample sequential training where each epoch trains on one sample.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_sample_idx = 0
        self.current_epoch = 0
        
    def set_current_sample(self, sample_idx):
        """Set the current sample to use for this epoch."""
        self.current_sample_idx = sample_idx
        # Set the data for the current sample
        if self.multi_sample_data:
            sample_id = self.sample_ids[sample_idx]
            
            if isinstance(self.multi_sample_data['train_peaks_seqs'], dict):
                self.peak_seqs = self.multi_sample_data['train_peaks_seqs'][sample_id]
            if isinstance(self.multi_sample_data['train_nonpeaks_seqs'], dict):
                self.nonpeak_seqs = self.multi_sample_data['train_nonpeaks_seqs'][sample_id]
                
            # Re-apply cropping and augmentation
            self.crop_revcomp_data()
    
    def on_epoch_end(self):
        """Called at the end of each epoch to switch to next sample."""
        if self.sample_ids:
            self.current_sample_idx = (self.current_sample_idx + 1) % len(self.sample_ids)
            self.set_current_sample(self.current_sample_idx)


class MultiSampleExtendedLossDataset(ChromBPNetDataset):
    """
    Dataset for multi-sample training with extended loss that averages across samples.
    """
    
    def crop_revcomp_data(self):
        """Apply random cropping and reverse complement augmentation to the multi-sample data."""
        if self.multi_sample_data and isinstance(self.multi_sample_data['train_peaks_seqs'], dict):
            # For extended loss, we need to prepare data from all samples
            all_samples_seqs = []
            all_samples_cts = []
            
            sample_ids = self.sample_ids or []
            
            for sample_id in sample_ids:
                # Get data for this sample
                peak_seqs = self.multi_sample_data['train_peaks_seqs'].get(sample_id, self.peak_seqs)
                nonpeak_seqs = self.multi_sample_data['train_nonpeaks_seqs'].get(sample_id, self.nonpeak_seqs)
                
                # Apply cropping and augmentation for this sample
                sample_seqs, sample_cts, sample_coords = crop_revcomp_data(
                    peak_seqs, self.peak_cts, self.peak_coords,
                    nonpeak_seqs, self.nonpeak_cts, self.nonpeak_coords,
                    self.inputlen, self.outputlen, self.add_revcomp, 
                    self.negative_sampling_ratio, self.shuffle_at_epoch_start
                )
                
                all_samples_seqs.append(sample_seqs)
                all_samples_cts.append(sample_cts)
            
            # Store all samples data
            self.cur_all_samples_seqs = all_samples_seqs
            self.cur_all_samples_cts = all_samples_cts
            
            # Use first sample as the primary data (for compatibility)
            if all_samples_seqs:
                self.cur_seqs = all_samples_seqs[0]
                self.cur_cts = all_samples_cts[0]
                self.cur_coords = self.peak_coords  # Use original coords
        else:
            # Fall back to standard cropping
            super().crop_revcomp_data()
    
    def __getitem__(self, idx):
        """Get a sample that includes data from all samples for extended loss calculation."""
        result = {
            'onehot_seq': self.cur_seqs[idx].astype(np.float32).transpose(),
            'profile': self.cur_cts[idx].astype(np.float32),
        }
        
        # Add all samples data for extended loss
        if hasattr(self, 'cur_all_samples_seqs') and self.cur_all_samples_seqs:
            all_samples_seqs = []
            all_samples_profiles = []
            
            for sample_seqs, sample_cts in zip(self.cur_all_samples_seqs, self.cur_all_samples_cts):
                if idx < len(sample_seqs) and idx < len(sample_cts):
                    all_samples_seqs.append(sample_seqs[idx].astype(np.float32).transpose())
                    all_samples_profiles.append(sample_cts[idx].astype(np.float32))
            
            result['all_samples_seqs'] = np.stack(all_samples_seqs) if all_samples_seqs else result['onehot_seq'][np.newaxis]
            result['all_samples_profiles'] = np.stack(all_samples_profiles) if all_samples_profiles else result['profile'][np.newaxis]
        else:
            # Fallback if no multi-sample data
            result['all_samples_seqs'] = result['onehot_seq'][np.newaxis]
            result['all_samples_profiles'] = result['profile'][np.newaxis]
            
        return result

