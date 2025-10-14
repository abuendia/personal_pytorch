import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT
from typing import Dict, List, Optional, Tuple, Any
import pyBigWig
from tqdm import tqdm

from .chrombpnet import ChromBPNet
from .personalized_genome import PersonalizedGenome, dna_to_one_hot
from .model_wrappers import ModelWrapper
from .data_config import DataConfig
from .genome import hg38


class ContrastiveDataset(Dataset):
    
    def __init__(
        self,
        consensus_peaks_path: str,
        individual_bigwig_dir: str,
        individual_vcf_dir: str,
        individual_list: List[str],
        reference_genome: str,
        rename_map: Optional[str] = None,
        input_window: int = 2114,
        output_window: int = 1000,
        max_pairs_per_epoch: int = 10000,
        device: str = 'cuda'
    ):
        """
        Initialize contrastive dataset.
        
        Args:
            consensus_peaks_path: Path to consensus peaks BED file
            individual_bigwig_dir: Directory containing individual BigWig files
            individual_vcf_dir: Directory containing individual VCF files
            individual_list: List of individual IDs to sample from
            reference_genome: Path to reference genome FASTA
            rename_map: Optional rename mapping file
            input_window: Input sequence window size
            output_window: Output profile window size
            max_pairs_per_epoch: Maximum number of individual pairs per epoch
            device: Device
        """
        self.consensus_peaks_path = consensus_peaks_path
        self.individual_bigwig_dir = individual_bigwig_dir
        self.individual_vcf_dir = individual_vcf_dir
        self.individual_list = individual_list
        self.reference_genome = reference_genome
        self.input_window = input_window
        self.output_window = output_window
        self.max_pairs_per_epoch = max_pairs_per_epoch
        self.device = device
        
        # Load rename map if provided
        self.rename_map = {}
        if rename_map and os.path.exists(rename_map):
            with open(rename_map, 'r') as f:
                for line in f:
                    old_name, new_name = line.strip().split('\t')
                    self.rename_map[old_name] = new_name
        
        # Load consensus peaks
        self.peaks_df = self._load_consensus_peaks()
        
        # Cache for personalized genomes and bigwig files
        self.personalized_genomes = {}
        self.bigwig_files = {}
        
        # Calculate z-score statistics for each individual
        self.zscore_stats = self._calculate_zscore_stats()
        
        print(f"Loaded {len(self.peaks_df)} consensus peaks")
        print(f"Available individuals: {len(self.individual_list)}")
    
    def _load_consensus_peaks(self) -> pd.DataFrame:
        """Load and process consensus peaks."""
        peaks_df = pd.read_csv(self.consensus_peaks_path, sep='\t', header=None)
        peaks_df.columns = ['chr', 'start', 'end']
        
        # Add summit position (center of peak)
        peaks_df['summit'] = ((peaks_df['end'] - peaks_df['start']) // 2).astype(int)
        
        # Filter peaks to ensure they fit within input window
        peaks_df = peaks_df[peaks_df['end'] - peaks_df['start'] >= self.output_window]
        
        return peaks_df.reset_index(drop=True)
    
    def _calculate_zscore_stats(self) -> Dict[str, Dict[str, float]]:
        """Calculate mean and std for z-score normalization per individual."""
        zscore_stats = {}
        
        for individual in tqdm(self.individual_list, desc="Calculating z-score stats"):
            bigwig_path = os.path.join(self.individual_bigwig_dir, f"{individual}.bw")
            if not os.path.exists(bigwig_path):
                continue
                
            try:
                bw = pyBigWig.open(bigwig_path)
                counts = []
                
                # Sample counts from all peaks
                for _, peak in self.peaks_df.iterrows():
                    chrom = peak['chr']
                    start = peak['start']
                    end = peak['end']
                    
                    if chrom in bw.chroms():
                        try:
                            count_values = bw.values(chrom, start, end)
                            if count_values and not all(v is None for v in count_values):
                                total_count = sum(v for v in count_values if v is not None)
                                counts.append(total_count)
                        except:
                            continue
                
                bw.close()
                
                if len(counts) > 0:
                    zscore_stats[individual] = {
                        'mean': np.mean(counts),
                        'std': np.std(counts)
                    }
                else:
                    zscore_stats[individual] = {'mean': 0.0, 'std': 1.0}
                    
            except Exception as e:
                print(f"Warning: Could not process {individual}: {e}")
                zscore_stats[individual] = {'mean': 0.0, 'std': 1.0}
        
        return zscore_stats
    
    def _get_personalized_genome(self, individual: str) -> PersonalizedGenome:
        """Get or create PersonalizedGenome for individual."""
        if individual not in self.personalized_genomes:
            self.personalized_genomes[individual] = PersonalizedGenome(
                reference_genome=self.reference_genome,
                vcf_dir=self.individual_vcf_dir,
                sample_id=individual,
                rename_map=self.rename_map
            )
        return self.personalized_genomes[individual]
    
    def _get_bigwig_file(self, individual: str) -> Optional[pyBigWig.pyBigWig]:
        """Get or open BigWig file for individual."""
        if individual not in self.bigwig_files:
            bigwig_path = os.path.join(self.individual_bigwig_dir, f"{individual}.bw")
            if os.path.exists(bigwig_path):
                self.bigwig_files[individual] = pyBigWig.open(bigwig_path)
            else:
                self.bigwig_files[individual] = None
        return self.bigwig_files[individual]
    
    def _get_individual_counts_and_sequence(self, individual: str, peak_idx: int) -> Tuple[float, torch.Tensor]:
        """Get ATAC counts and personalized sequence for individual at peak."""
        peak = self.peaks_df.iloc[peak_idx]
        
        # Get ATAC counts
        bw = self._get_bigwig_file(individual)
        if bw is None:
            return 0.0, torch.zeros((4, self.input_window))
        
        try:
            # Get counts in peak region
            count_values = bw.values(peak['chr'], peak['start'], peak['end'])
            if count_values and not all(v is None for v in count_values):
                total_count = sum(v for v in count_values if v is not None)
            else:
                total_count = 0.0
        except:
            total_count = 0.0
        
        # Calculate z-score
        if individual in self.zscore_stats:
            stats = self.zscore_stats[individual]
            if stats['std'] > 0:
                z_score = (total_count - stats['mean']) / stats['std']
            else:
                z_score = 0.0
        else:
            z_score = 0.0
        
        # Get personalized sequence
        try:
            personalized_genome = self._get_personalized_genome(individual)
            
            # Calculate sequence coordinates centered on peak
            summit = peak['start'] + peak['summit']
            seq_start = summit - self.input_window // 2
            seq_end = summit + self.input_window // 2
            
            # Get both haplotypes and average them
            hap1, hap2 = personalized_genome.get_sequence_with_haplotypes(
                peak['chr'], seq_start, seq_end
            )
            
            # Convert to one-hot and average
            hap1_onehot = dna_to_one_hot([hap1])[0]  # Shape: (L, 4)
            hap2_onehot = dna_to_one_hot([hap2])[0]  # Shape: (L, 4)
            averaged_onehot = (hap1_onehot + hap2_onehot) / 2
            
            # Convert to tensor and transpose to (4, L)
            sequence = torch.tensor(averaged_onehot.T, dtype=torch.float32)
            
        except Exception as e:
            print(f"Warning: Could not get personalized sequence for {individual}: {e}")
            sequence = torch.zeros((4, self.input_window))
        
        return z_score, sequence
    
    def __len__(self) -> int:
        return self.max_pairs_per_epoch
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a contrastive training sample (pair of individuals at a peak)."""
        # Randomly sample two individuals
        individual1, individual2 = random.sample(self.individual_list, 2)
        
        # Randomly sample a peak
        peak_idx = random.randint(0, len(self.peaks_df) - 1)
        
        # Get data for both individuals
        z1, seq1 = self._get_individual_counts_and_sequence(individual1, peak_idx)
        z2, seq2 = self._get_individual_counts_and_sequence(individual2, peak_idx)
        
        # Calculate true contrastive target
        true_contrast = (z1 - z2) ** 2
        
        return {
            'individual1_sequence': seq1,
            'individual2_sequence': seq2,
            'true_contrast': torch.tensor(true_contrast, dtype=torch.float32),
            'individual1_zscore': torch.tensor(z1, dtype=torch.float32),
            'individual2_zscore': torch.tensor(z2, dtype=torch.float32),
            'individual1_id': individual1,
            'individual2_id': individual2,
            'peak_idx': peak_idx
        }
    
    def close(self):
        """Close all open files."""
        for bw in self.bigwig_files.values():
            if bw is not None:
                bw.close()
        for pg in self.personalized_genomes.values():
            pg.close()


class ContrastiveChromBPNetWrapper(ModelWrapper):
    """
    ChromBPNet wrapper for contrastive training.
    
    This wrapper extends the base ChromBPNet model to learn contrastive representations
    by predicting differences in ATAC-seq accessibility between individuals.
    """
    
    def __init__(self, args):
        """Initialize contrastive wrapper."""
        super().__init__(args)
        
        # Load the base ChromBPNet model
        from .model_config import ChromBPNetConfig
        config = ChromBPNetConfig.from_argparse_args(args)
        self.model = ChromBPNet(config)
        
        # Load pretrained weights if checkpoint is provided
        if hasattr(args, 'checkpoint') and args.checkpoint:
            self._load_pretrained_weights(args.checkpoint)
        
        # Contrastive loss weight
        self.contrastive_weight = getattr(args, 'contrastive_weight', 1.0)
    
    def _load_pretrained_weights(self, checkpoint_path: str):
        """Load pretrained ChromBPNet weights."""
        if checkpoint_path.endswith('.ckpt'):
            # Lightning checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.load_state_dict(checkpoint['state_dict'], strict=False)
        elif checkpoint_path.endswith('.pt'):
            # PyTorch state dict
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(state_dict, strict=False)
        else:
            print(f"Warning: Unknown checkpoint format: {checkpoint_path}")
    
    def _step(self, batch: Dict[str, torch.Tensor], batch_idx: int, mode: str = 'train') -> STEP_OUTPUT:
        """Training/validation step for contrastive learning."""
        seq1 = batch['individual1_sequence']  # (batch_size, 4, seq_len)
        seq2 = batch['individual2_sequence']  # (batch_size, 4, seq_len)
        true_contrast = batch['true_contrast']  # (batch_size,)
        
        # Get model predictions for both individuals
        # ChromBPNet returns (profile, counts), we only need counts
        _, pred_counts1 = self.model(seq1)  # (batch_size, 1)
        _, pred_counts2 = self.model(seq2)  # (batch_size, 1)
        
        # Squeeze counts to (batch_size,)
        pred_counts1 = pred_counts1.squeeze(-1)
        pred_counts2 = pred_counts2.squeeze(-1)
        
        # Calculate predicted contrastive difference
        pred_contrast = (pred_counts1 - pred_counts2) ** 2

        breakpoint()
        
        # Contrastive loss: MSE between true and predicted contrasts
        contrastive_loss = F.mse_loss(pred_contrast, true_contrast)
        
        # Store metrics for logging
        if mode in ['train', 'val']:
            self.metrics[mode]['preds'].append(pred_contrast.detach())
            self.metrics[mode]['targets'].append(true_contrast.detach())
        
        # Calculate correlation for monitoring
        with torch.no_grad():
            correlation = torch.corrcoef(torch.stack([pred_contrast, true_contrast]))[0, 1]
            if torch.isnan(correlation):
                correlation = torch.tensor(0.0)
        
        # Log metrics
        log_dict = {
            f'{mode}_contrastive_loss': contrastive_loss,
            f'{mode}_contrastive_correlation': correlation,
            f'{mode}_pred_contrast_mean': pred_contrast.mean(),
            f'{mode}_true_contrast_mean': true_contrast.mean(),
        }
        
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        if mode == 'predict':
            return {
                'pred_contrast': pred_contrast.detach().cpu().numpy(),
                'true_contrast': true_contrast.detach().cpu().numpy(),
                'pred_counts1': pred_counts1.detach().cpu().numpy(),
                'pred_counts2': pred_counts2.detach().cpu().numpy(),
                'true_zscore1': batch['individual1_zscore'].detach().cpu().numpy(),
                'true_zscore2': batch['individual2_zscore'].detach().cpu().numpy(),
            }
        
        return contrastive_loss
    
    def configure_optimizers(self):
        """Configure optimizer for contrastive training."""
        # Use a lower learning rate for fine-tuning
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, eps=1e-7)
        
        # Optional: Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_contrastive_loss'
            }
        }


class ContrastiveDataModule(L.LightningDataModule):
    """DataModule for contrastive training."""
    
    def __init__(
        self,
        consensus_peaks_path: str,
        individual_bigwig_dir: str,
        individual_vcf_dir: str,
        reference_genome: str,
        rename_map: Optional[str] = None,
        input_window: int = 2114,
        output_window: int = 1000,
        batch_size: int = 32,
        num_workers: int = 4,
        max_pairs_per_epoch: int = 10000,
        train_individuals: Optional[List[str]] = None,
        val_individuals: Optional[List[str]] = None
    ):
        """Initialize contrastive data module."""
        super().__init__()
        
        self.consensus_peaks_path = consensus_peaks_path
        self.individual_bigwig_dir = individual_bigwig_dir
        self.individual_vcf_dir = individual_vcf_dir
        self.reference_genome = reference_genome
        self.rename_map = rename_map
        self.input_window = input_window
        self.output_window = output_window
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_pairs_per_epoch = max_pairs_per_epoch
        self.individual_list = [item.split(".")[0] for item in os.listdir(self.individual_vcf_dir) if item.endswith('.csv')]
        
        # Split individuals for train/val
        if train_individuals is None:
            train_individuals = self.individual_list[:int(0.8 * len(self.individual_list))]
        if val_individuals is None:
            val_individuals = self.individual_list[int(0.8 * len(self.individual_list)):]
        
        self.train_individuals = train_individuals
        self.val_individuals = val_individuals
    
    def setup(self, stage: str = None):
        """Setup datasets for training and validation."""
        if stage == 'fit' or stage is None:
            self.train_dataset = ContrastiveDataset(
                consensus_peaks_path=self.consensus_peaks_path,
                individual_bigwig_dir=self.individual_bigwig_dir,
                individual_vcf_dir=self.individual_vcf_dir,
                individual_list=self.train_individuals,
                reference_genome=self.reference_genome,
                rename_map=self.rename_map,
                input_window=self.input_window,
                output_window=self.output_window,
                max_pairs_per_epoch=self.max_pairs_per_epoch
            )
            
            self.val_dataset = ContrastiveDataset(
                consensus_peaks_path=self.consensus_peaks_path,
                individual_bigwig_dir=self.individual_bigwig_dir,
                individual_vcf_dir=self.individual_vcf_dir,
                individual_list=self.val_individuals,
                reference_genome=self.reference_genome,
                rename_map=self.rename_map,
                input_window=self.input_window,
                output_window=self.output_window,
                max_pairs_per_epoch=self.max_pairs_per_epoch // 4  # Fewer pairs for validation
            )
    
    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
    
    def teardown(self, stage: str = None):
        """Clean up resources."""
        if hasattr(self, 'train_dataset'):
            self.train_dataset.close()
        if hasattr(self, 'val_dataset'):
            self.val_dataset.close()


def run_contrastive_finetuning(
    base_model_checkpoint: str,
    consensus_peaks_path: str,
    individual_bigwig_dir: str,
    individual_vcf_dir: str,
    output_dir: str,
    reference_genome: str = hg38.fasta,
    rename_map: Optional[str] = None,
    max_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    contrastive_weight: float = 1.0,
    max_pairs_per_epoch: int = 10000,
    gpu_ids: List[int] = [0],
    fast_dev_run: bool = False
):
    """
    Run contrastive fine-tuning of ChromBPNet.
    
    Args:
        base_model_checkpoint: Path to pretrained ChromBPNet checkpoint
        consensus_peaks_path: Path to consensus peaks BED file
        individual_bigwig_dir: Directory containing individual BigWig files
        individual_vcf_dir: Directory containing individual VCF files
        output_dir: Output directory for saving model and logs
        reference_genome: Path to reference genome FASTA
        rename_map: Optional rename mapping file
        max_epochs: Maximum number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimization
        contrastive_weight: Weight for contrastive loss
        max_pairs_per_epoch: Maximum number of individual pairs per epoch
        gpu_ids: GPU device IDs to use
        fast_dev_run: Whether to run a quick development test
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create data module
    datamodule = ContrastiveDataModule(
        consensus_peaks_path=consensus_peaks_path,
        individual_bigwig_dir=individual_bigwig_dir,
        individual_vcf_dir=individual_vcf_dir,
        reference_genome=reference_genome,
        rename_map=rename_map,
        batch_size=batch_size,
        max_pairs_per_epoch=max_pairs_per_epoch
    )
    
    # Create model wrapper
    class Args:
        def __init__(self):
            # Required attributes for ModelWrapper
            self.alpha = 1.0  # Weight for count loss
            self.beta = 1.0   # Weight for profile loss
            self.verbose = False
            
            # Contrastive training specific attributes
            self.checkpoint = None
            self.contrastive_weight = 1.0
            self.learning_rate = 1e-4
            
            # Model config attributes (from ChromBPNetConfig defaults)
            self.out_dim = 1000
            self.n_filters = 512
            self.n_layers = 8
            self.conv1_kernel_size = 21
            self.profile_kernel_size = 75
            self.n_outputs = 1
            self.n_control_tracks = 0
            self.profile_output_bias = True
            self.count_output_bias = True
            self.bias_scaled = None
            self.chrombpnet_wo_bias = None
    
    args = Args()
    model = ContrastiveChromBPNetWrapper(args)
    
    # Create logger
    logger = L.pytorch.loggers.CSVLogger(output_dir, name='contrastive_finetuning')
    
    # Create trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu',
        devices=gpu_ids,
        logger=logger,
        callbacks=[
            L.pytorch.callbacks.EarlyStopping(
                monitor='val_contrastive_loss',
                patience=10,
                mode='min'
            ),
            L.pytorch.callbacks.ModelCheckpoint(
                monitor='val_contrastive_loss',
                save_top_k=1,
                mode='min',
                filename='best_contrastive_model',
                save_last=True
            )
        ],
        fast_dev_run=fast_dev_run,
        precision=32
    )
    
    # Train the model
    print("Starting contrastive fine-tuning...")
    trainer.fit(model, datamodule)
    
    print(f"Training complete! Model saved to {output_dir}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'contrastive_model_final.pt')
    torch.save(model.model.state_dict(), final_model_path)
    
    return model, trainer


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Contrastive fine-tuning for ChromBPNet')
    
    # Required arguments
    parser.add_argument('--base_model_checkpoint', type=str, required=True,
                       help='Path to pretrained ChromBPNet checkpoint')
    parser.add_argument('--consensus_peaks_path', type=str, required=True,
                       help='Path to consensus peaks BED file')
    parser.add_argument('--individual_bigwig_dir', type=str, required=True,
                       help='Directory containing individual BigWig files')
    parser.add_argument('--individual_vcf_dir', type=str, required=True,
                       help='Directory containing individual VCF files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for saving model and logs')
    
    # Optional arguments
    parser.add_argument('--reference_genome', type=str, default=hg38.fasta,
                       help='Path to reference genome FASTA')
    parser.add_argument('--rename_map', type=str, default=None,
                       help='Optional rename mapping file')
    parser.add_argument('--max_epochs', type=int, default=50,
                       help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate for optimization')
    parser.add_argument('--contrastive_weight', type=float, default=1.0,
                       help='Weight for contrastive loss')
    parser.add_argument('--max_pairs_per_epoch', type=int, default=10000,
                       help='Maximum number of individual pairs per epoch')
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0],
                       help='GPU device IDs to use')
    parser.add_argument('--fast_dev_run', type=str, default='False',
                       help='Whether to run a quick development test')
    
    args = parser.parse_args()
    
    # Convert string to boolean for fast_dev_run
    fast_dev_run = args.fast_dev_run.lower() in ['true', '1', 'yes']
    
    # Run contrastive fine-tuning
    model, trainer = run_contrastive_finetuning(
        base_model_checkpoint=args.base_model_checkpoint,
        consensus_peaks_path=args.consensus_peaks_path,
        individual_bigwig_dir=args.individual_bigwig_dir,
        individual_vcf_dir=args.individual_vcf_dir,
        output_dir=args.output_dir,
        reference_genome=args.reference_genome,
        rename_map=args.rename_map,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        contrastive_weight=args.contrastive_weight,
        max_pairs_per_epoch=args.max_pairs_per_epoch,
        gpu_ids=args.gpu_ids,
        fast_dev_run=fast_dev_run
    )
