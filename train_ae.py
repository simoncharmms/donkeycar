#!/usr/bin/env python3
"""
Train autoencoder on collected Donkey Car images.

Usage:
    python train_ae.py --train-dirs /path/to/train/images/ /path/to/more/images/ \
                       --val-dirs /path/to/val/images/ \
                       --z-size 64 \
                       --epochs 100
"""
import argparse
import os
import sys

from rl_pipeline.config import AEConfig, PathConfig
from rl_pipeline.ae_trainer import AETrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train VAE on Donkey Car images"
    )
    
    # Data paths
    parser.add_argument(
        '--train-dirs',
        type=str,
        nargs='+',
        required=True,
        help="One or more directories containing training images"
    )
    parser.add_argument(
        '--val-dirs',
        type=str,
        nargs='+',
        required=True,
        help="One or more directories containing validation images"
    )
    
    # Model config
    parser.add_argument('--z-size', type=int, default=64, help="Latent dimension")
    parser.add_argument('--c-hid', type=int, default=64, help="Hidden channel size")
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--noise-factor', type=float, default=0.3)
    parser.add_argument('--num-threads', type=int, default=-1)
    parser.add_argument('--n-samples', type=int, default=-1)
    
    # Output
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./ae_outputs',
        help="Directory for model checkpoints and logs"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    for dir_path in args.train_dirs + args.val_dirs:
        if not os.path.isdir(dir_path):
            print(f"✗ Directory not found: {dir_path}")
            sys.exit(1)
        
        images = [f for f in os.listdir(dir_path) if f.endswith('.jpg')]
        print(f"  {dir_path}: {len(images)} images")
    
    # Create config
    ae_config = AEConfig(
        z_size=args.z_size,
        c_hid=args.c_hid,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        noise_factor=args.noise_factor,
        num_threads=args.num_threads,
        n_samples=args.n_samples,
    )
    
    # Create trainer
    print(f"\n{'='*60}")
    print("Autoencoder Training")
    print(f"{'='*60}")
    print(f"Model config:")
    print(f"  Latent dimension: {ae_config.z_size}")
    print(f"  Hidden channels: {ae_config.c_hid}")
    print(f"  Batch size: {ae_config.batch_size}")
    print(f"  Epochs: {ae_config.epochs}")
    print(f"  Learning rate: {ae_config.learning_rate}")
    print(f"  Noise factor: {ae_config.noise_factor}")
    
    trainer = AETrainer(ae_config, output_dir=args.output_dir)
    
    # Train
    best_model_path = trainer.train(args.train_dirs, args.val_dirs)
    
    print(f"\n✓ Best model: {best_model_path}")


if __name__ == "__main__":
    main()
