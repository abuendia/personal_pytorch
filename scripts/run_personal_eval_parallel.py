#!/usr/bin/env python3
"""
Truly optimized personal evaluation script that:
1. Loads BigWig and variants ONCE per sample
2. Runs all 5 fold predictions using the same loaded data
3. Uses GPU pooling for maximum efficiency
"""

import os
import sys
import tempfile
import shutil
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from pathlib import Path

# Add the project to Python path
sys.path.append('/oak/stanford/groups/akundaje/abuen/personal_genome/personal_pytorch')

import torch
import pandas as pd
from chrombpnet.data_config import DataConfig
from chrombpnet.dataset import DataModule
from chrombpnet.model_wrappers import create_model_wrapper
import lightning as L

class GPUManager:
    """Manages GPU allocation across parallel jobs"""
    
    def __init__(self, gpu_list):
        self.gpu_queue = queue.Queue()
        for gpu in gpu_list:
            self.gpu_queue.put(gpu)
        self.lock = threading.Lock()
    
    def get_gpu(self):
        """Get an available GPU"""
        return self.gpu_queue.get()
    
    def return_gpu(self, gpu):
        """Return a GPU to the pool"""
        self.gpu_queue.put(gpu)

def run_fold_prediction_optimized(sample_id, fold, gpu_manager, output_dir, datamodule, log_dir):
    """Run prediction for a single fold using pre-loaded data"""
    gpu = gpu_manager.get_gpu()
    
    try:
        print(f"[{time.strftime('%T')}] Starting fold {fold} for {sample_id} on GPU {gpu}")
        
        # Create log file for this fold
        log_file = os.path.join(log_dir, f"{sample_id}_fold_{fold}_{time.strftime('%Y%m%d_%H%M%S')}.log")
        
        # Create output directory for this fold
        fold_output_dir = os.path.join(output_dir, sample_id, f"fold_{fold}")
        os.makedirs(fold_output_dir, exist_ok=True)
        
        # Load model for this fold
        checkpoint_path = f"/oak/stanford/groups/akundaje/abuen/personal_genome/personal_pytorch/models/ENCFF142IOR/fold_{fold}/model.chrombpnet_nobias.fold_{fold}.ENCSR637XSC.h5"
        
        if not os.path.exists(checkpoint_path):
            error_msg = f"Checkpoint not found: {checkpoint_path}"
            print(f"[{time.strftime('%T')}] {error_msg}")
            return (sample_id, fold, False, error_msg)
        
        # Create model wrapper
        class Args:
            def __init__(self):
                self.checkpoint = checkpoint_path
                self.model_type = "chrombpnet"
                self.fold = fold
                self.out_dir = output_dir
                self.name = sample_id
                self.gpu = [gpu]
                self.fast_dev_run = False
                self.adjust_bias = True
                self.bias_scaled = f"/oak/stanford/groups/akundaje/abuen/personal_genome/personal_pytorch/models/ENCFF142IOR/fold_{fold}/model.bias_scaled.fold_{fold}.ENCSR637XSC.h5"
        
        args = Args()
        
        # Load model
        model_wrapper = create_model_wrapper(args)
        
        # Adjust bias if needed
        if args.adjust_bias and os.path.exists(args.bias_scaled):
            from chrombpnet.model_wrappers import adjust_bias_model_logcounts
            adjust_bias_model_logcounts(model_wrapper.model.bias, datamodule.negative_dataloader())
        
        # Create trainer
        trainer = L.Trainer(logger=False, devices=args.gpu, val_check_interval=None)
        
        # Run prediction
        with open(log_file, 'w') as f:
            # Redirect stdout/stderr to log file
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = f
            sys.stderr = f
            
            try:
                # Get dataloader for test chromosomes
                dataloader, dataset = datamodule.chrom_dataloader('test')
                
                # Run prediction
                output = trainer.predict(model_wrapper, dataloader)
                
                # Save results
                from chrombpnet.metrics import load_output_to_regions, save_predictions, compare_with_observed
                
                out_dir = os.path.join(fold_output_dir, 'predict')
                os.makedirs(out_dir, exist_ok=True)
                
                regions, parsed_output = load_output_to_regions(
                    output, 
                    dataset.regions, 
                    os.path.join(out_dir, 'test')
                )
                
                model_metrics = compare_with_observed(
                    regions, 
                    parsed_output, 
                    os.path.join(out_dir, 'test')
                )
                
                save_predictions(
                    output, 
                    regions, 
                    datamodule.config.chrom_sizes, 
                    os.path.join(out_dir, 'test')
                )
                
            finally:
                # Restore stdout/stderr
                sys.stdout = old_stdout
                sys.stderr = old_stderr
        
        print(f"[{time.strftime('%T')}] Completed fold {fold} for {sample_id} on GPU {gpu}")
        return (sample_id, fold, True, None)
        
    except Exception as e:
        error_msg = f"Error in fold {fold} for {sample_id}: {str(e)}"
        print(f"[{time.strftime('%T')}] {error_msg}")
        return (sample_id, fold, False, error_msg)
    finally:
        gpu_manager.return_gpu(gpu)

def process_sample_optimized(sample_id, gpu_manager, output_dir, folds, log_dir, max_workers=5):
    """Process a single sample with all folds using shared data loading"""
    print(f"[{time.strftime('%T')}] Starting optimized processing for {sample_id}")
    
    try:
        # Create data config for this sample
        data_config = DataConfig(
            data_dir="/oak/stanford/groups/akundaje/abuen/personal_genome/personal_pytorch",
            peaks="/oak/stanford/groups/akundaje/abuen/personal_genome/personal_pytorch/data/consensus_peaks/overlap.optimal_peak.narrowPeak.merged.bed",
            bigwig=f"/oak/stanford/groups/akundaje/ziwei75/african-omics/data/processed/{sample_id}/{sample_id}_unstranded.bw",
            vcf_dir="/oak/stanford/groups/akundaje/ziwei75/african-omics/data/genotype/individual_vcf",
            sample_id=sample_id,
            rename_map="/oak/stanford/groups/akundaje/ziwei75/african-omics/data/genotype/individual_vcf/rename.txt",
            fold=0  # We'll override this for each fold
        )
        
        print(f"[{time.strftime('%T')}] Loading data for {sample_id} (this may take a while for variant processing)...")
        
        # Create datamodule - this loads BigWig, variants, and peaks ONCE
        datamodule = DataModule(data_config)
        
        print(f"[{time.strftime('%T')}] Data loaded for {sample_id}, starting fold predictions...")
        
        # Create arguments for each fold
        fold_args = [(sample_id, fold, gpu_manager, output_dir, datamodule, log_dir) for fold in folds]
        
        # Run all folds in parallel with thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_fold_prediction_optimized, *args) for args in fold_args]
            
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
        
        # Analyze results
        successful = sum(1 for r in results if r[2])
        failed = len(results) - successful
        
        print(f"[{time.strftime('%T')}] Completed {sample_id}: {successful} successful, {failed} failed")
        
        if failed > 0:
            print(f"Failed folds for {sample_id}:")
            for sample_id_result, fold, success, error in results:
                if not success:
                    print(f"  Fold {fold}: {error}")
        
        return successful == len(folds)
        
    except Exception as e:
        error_msg = f"Error processing sample {sample_id}: {str(e)}"
        print(f"[{time.strftime('%T')}] {error_msg}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Truly optimized personal evaluation with shared data loading")
    parser.add_argument("--samples", nargs="+", help="Sample IDs to process")
    parser.add_argument("--output-dir", default="/oak/stanford/groups/akundaje/abuen/personal_genome/personal_pytorch/output/personal_eval_ENCFF142IOR")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-workers", type=int, default=5, help="Max workers per sample")
    parser.add_argument("--auto-samples", action="store_true", help="Auto-detect samples from directory")
    parser.add_argument("--log-dir", default="./logs", help="Directory for log files")
    
    args = parser.parse_args()
    
    # Get samples
    if args.auto_samples:
        samples_dir = "/oak/stanford/groups/akundaje/ziwei75/african-omics/data/processed"
        samples = [d for d in os.listdir(samples_dir) if d.startswith(('G', 'H'))]
    else:
        samples = args.samples
    
    if not samples:
        print("No samples found!")
        return 1
    
    print(f"Processing {len(samples)} samples: {samples}")
    print(f"Folds: {args.folds}")
    print(f"GPUs: {args.gpus}")
    print(f"Max workers per sample: {args.max_workers}")
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Initialize GPU manager
    gpu_manager = GPUManager(args.gpus)
    
    # Process samples sequentially (to avoid memory conflicts)
    # but run all folds for each sample in parallel using shared data
    results = []
    for i, sample_id in enumerate(samples):
        print(f"\n[{time.strftime('%T')}] Processing sample {i+1}/{len(samples)}: {sample_id}")
        
        success = process_sample_optimized(
            sample_id, 
            gpu_manager, 
            args.output_dir, 
            args.folds, 
            args.log_dir, 
            args.max_workers
        )
        results.append((sample_id, success))
    
    # Summary
    print(f"\n[{time.strftime('%T')}] All samples completed!")
    successful_samples = sum(1 for _, success in results if success)
    print(f"Successful: {successful_samples}/{len(samples)}")
    
    if successful_samples < len(samples):
        print("Failed samples:")
        for sample_id, success in results:
            if not success:
                print(f"  {sample_id}")
    
    return 0 if successful_samples == len(samples) else 1

if __name__ == "__main__":
    sys.exit(main())
