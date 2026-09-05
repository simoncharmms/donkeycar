#!/usr/bin/env python3
"""
Train RL agent (SAC or PPO) on Donkey Car.

Usage:
    # Train SAC
    python train_rl.py --algorithm SAC \
                       --env donkey-simulator-v0 \
                       --total-timesteps 50000 \
                       --batch-size 64

    # Train PPO
    python train_rl.py --algorithm PPO \
                       --env donkey-simulator-v0 \
                       --total-timesteps 100000

    # Train with AE encoder
    python train_rl.py --algorithm SAC \
                       --ae-path ./ae_outputs/vae_64_1234567890_best.pt \
                       --z-size 64
"""
import argparse
import os
import sys

import torch

from rl_pipeline.config import RLConfig, SimulatorConfig, PathConfig
from rl_pipeline.rl_trainer import RLTrainer
from rl_pipeline.ae_trainer import VAE


def main():
    parser = argparse.ArgumentParser(
        description="Train RL agent on Donkey Car"
    )
    
    # Algorithm
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['SAC', 'PPO'],
        default='SAC',
        help="RL algorithm"
    )
    
    # Environment
    parser.add_argument(
        '--env',
        type=str,
        default='donkey-simulator-v0',
        help="Gym environment ID"
    )
    parser.add_argument(
        '--exe-path',
        type=str,
        default='',
        help="Path to Donkey simulator executable (leave empty for gym-donkeycar)"
    )
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help="Simulator host"
    )
    parser.add_argument(
        '--port',
        type=int,
        default=9091,
        help="Simulator port"
    )
    parser.add_argument(
        '--max-cte',
        type=float,
        default=4.0,
        help="Maximum cross-track error"
    )
    
    # Training config
    parser.add_argument('--total-timesteps', type=int, default=50000)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--buffer-size', type=int, default=30000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.02)
    parser.add_argument('--gradient-steps', type=int, default=600)
    parser.add_argument('--eval-freq', type=int, default=50)
    parser.add_argument('--n-eval-episodes', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    
    # AE (optional)
    parser.add_argument(
        '--ae-path',
        type=str,
        default=None,
        help="Path to trained AE model"
    )
    parser.add_argument('--z-size', type=int, default=64)
    
    # Output
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./rl_outputs',
        help="Output directory for models and logs"
    )
    
    args = parser.parse_args()
    
    # Validate AE if provided
    ae_encoder = None
    if args.ae_path:
        if not os.path.exists(args.ae_path):
            print(f"✗ AE model not found: {args.ae_path}")
            sys.exit(1)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ae_encoder = VAE(z_size=args.z_size)
        ae_encoder.load(args.ae_path, device=device)
        ae_encoder.to(device)
        print(f"✓ Loaded AE from {args.ae_path}")
    
    # Create configs
    sim_config = SimulatorConfig(
        exe_path=args.exe_path,
        host=args.host,
        port=args.port,
        max_cte=args.max_cte,
    )
    
    rl_config = RLConfig(
        algorithm=args.algorithm,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        gamma=args.gamma,
        tau=args.tau,
        gradient_steps=args.gradient_steps,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        seed=args.seed,
    )
    
    path_config = PathConfig(base_dir=args.output_dir)
    
    # Create trainer
    print(f"\n{'='*60}")
    print(f"RL Training - {args.algorithm}")
    print(f"{'='*60}")
    print(f"Environment: {args.env}")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    if ae_encoder:
        print(f"Using AE: latent dim={args.z_size}")
    
    trainer = RLTrainer(
        rl_config=rl_config,
        sim_config=sim_config,
        path_config=path_config,
        env_id=args.env,
        ae_encoder=ae_encoder,
    )
    
    # Train
    try:
        trainer.train()
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
