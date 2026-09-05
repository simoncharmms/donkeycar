# Donkey Car RL Pipeline

Complete pipeline for training a reinforcement learning agent on a Donkey Car using both simulator and real-world data.

## Overview

```
Manual Data Collection
    ↓
[Simulator]  [Real Car]
    ↓              ↓
    └── Image Dataset ──→ Train AE/VAE
                             ↓
                    Trained Encoder (z_size=64)
                             ↓
    ┌────────────────────────┴────────────────────────┐
    ↓                                                  ↓
Train RL (SAC/PPO)               Train RL (SAC/PPO)
[Simulator]                       [Real Car]
    ↓                                ↓
Trained Policy                   Trained Policy
```

## Quick Start

### 1. Collect Training Data

**Option A: Manual driving in simulator**
```python
from rl_pipeline.data_collector import ManualDrivingSession

session = ManualDrivingSession(
    output_dir='./data/simulator_run_1',
    env_kwargs={
        'exe_path': '/path/to/donkey_sim',
        'host': '127.0.0.1',
        'port': 9091,
    }
)
session.create_env('donkey-simulator-v0')
session.run_interactive(duration_seconds=600)  # 10 minutes
```

**Option B: Collect from real car**
```python
session = ManualDrivingSession(
    output_dir='./data/real_car_run_1',
    env_kwargs={
        'host': '192.168.1.100',  # Pi IP
        'port': 8887,
    }
)
# Real car will save images automatically
```

### 2. Train Autoencoder

```bash
python train_ae.py \
    --train-dirs ./data/simulator_run_1/images ./data/real_car_run_1/images \
    --val-dirs ./data/test_run/images \
    --z-size 64 \
    --epochs 100 \
    --batch-size 32
```

This produces: `./ae_outputs/vae_64_<timestamp>_best.pt`

### 3. Train RL Agent

```bash
# Option A: Train with SAC in simulator
python train_rl.py \
    --algorithm SAC \
    --env donkey-simulator-v0 \
    --exe-path /path/to/donkey_sim \
    --total-timesteps 50000 \
    --ae-path ./ae_outputs/vae_64_1234567890_best.pt \
    --z-size 64

# Option B: Train with PPO
python train_rl.py \
    --algorithm PPO \
    --total-timesteps 100000

# Option C: Train on real car
python train_rl.py \
    --algorithm SAC \
    --host 192.168.1.100 \
    --port 8887 \
    --ae-path ./ae_outputs/vae_64_1234567890_best.pt
```

### 4. Deploy to Pi

```python
from stable_baselines3 import SAC
import torch
from rl_pipeline.ae_trainer import VAE

# Load models
ae = VAE(z_size=64)
ae.load('./ae_outputs/vae_64_1234567890_best.pt')

policy = SAC.load('./rl_outputs/rl_models/sac_1234567890')

# Run inference loop on Pi
while True:
    image = camera.capture()
    latent = ae.encode_raw_image(image)
    action, _ = policy.predict(latent)
    car.drive(action)
```

## Directory Structure

```
donkeycar/
├── rl_pipeline/
│   ├── __init__.py
│   ├── config.py           # Configuration dataclasses
│   ├── ae_trainer.py       # VAE training
│   ├── rl_trainer.py       # RL training (SAC/PPO)
│   └── data_collector.py   # Data collection tools
├── train_ae.py             # AE training script
├── train_rl.py             # RL training script
├── RL_PIPELINE.md          # This file
└── config_example.yaml     # Example configuration

rl_outputs/
├── data/                   # Raw collected images
├── ae_models/              # Trained AE checkpoints
├── rl_models/              # Trained RL policies
├── logs/                   # Training logs and videos
├── tensorboard/            # TensorBoard logs
└── training_images/        # AE reconstruction visualizations
```

## Configuration

### Simulator Config
```python
SimulatorConfig(
    exe_path="/path/to/donkey_sim.x86_64",
    host="127.0.0.1",
    port=9091,
    car_name="training",
    max_cte=4.0,  # Cross-track error threshold
)
```

### AE Config
- `z_size`: Latent dimension (default 64)
- `c_hid`: Hidden channel size (default 64)
- `learning_rate`: (default 1e-4)
- `batch_size`: (default 32)
- `epochs`: Number of training epochs (default 100)
- `noise_factor`: Gaussian noise std (default 0.3)

### RL Config
- `algorithm`: "SAC" or "PPO"
- `total_timesteps`: Total training steps
- `learning_rate`: (default 3e-4)
- `batch_size`: (default 64)
- `gamma`: Discount factor (default 0.99)
- `tau`: Soft update coefficient (default 0.02)
- `eval_freq`: Evaluation frequency (default 50)

## Key Components

### VAE Training

The `AETrainer` class handles:
- Image loading and preprocessing
- Gaussian noise augmentation (robustness)
- VAE training with KL + reconstruction loss
- Validation and checkpointing
- TensorBoard logging

**Why VAE?**
- Compresses images to 64-D latent space
- Faster RL training (64 inputs vs 19,200)
- Better generalization
- Learned features capture relevant driving dynamics

### RL Training

The `RLTrainer` class supports:
- **SAC (Soft Actor-Critic)**: More stable, recommended for this task
- **PPO (Proximal Policy Optimization)**: Good convergence, more exploratory

**Key hyperparameters:**
- `batch_size=64`: Balance between stability and speed
- `buffer_size=30000`: Experience replay buffer
- `gradient_steps=600`: Updates per episode
- `use_sde=True`: Stochastic dynamics exploration

## Reward Functions

The environment provides:
1. **Speed reward**: Driving fast
2. **CTE penalty**: Cross-track error (stay centered)
3. **Crash penalty**: Negative reward for going out of bounds
4. **Smooth driving bonus**: Optional, for smooth steering/throttle

Customize in your wrapper or environment setup.

## Known Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Car learns to spin | CTE threshold too large | Reduce `max_cte` (try 2.0) |
| AE sees noise in latent | Underexplored track areas | Collect more diverse data |
| Slow training on real Pi | Inference latency | Pre-record experience on laptop, transfer model |
| Reward not improving | Bad reward function | Check CTE calc, add smooth driving bonus |
| Unstable PPO training | High LR or large batch | Reduce LR to 1e-4, batch to 32 |

## Tips

### Data Collection
- Collect 2000+ images for robust AE training
- Drive diverse lines (center, left, right)
- Include recoveries from poor positions
- Record on both simulator and real track if possible

### AE Training
- Run for ~100 epochs
- Watch TensorBoard: `tensorboard --logdir ./ae_outputs/runs`
- Reconstruction quality should improve steadily
- Val loss should plateau, not diverge

### RL Training
- Start with SAC (more stable than PPO)
- Eval every 50 steps, save best model
- TensorBoard: `tensorboard --logdir ./rl_outputs/tensorboard`
- Training should show reward improvement over time

### Deployment
- Convert both models to TorchScript for Pi inference
- Run AE on Pi GPU (if available) or CPU
- RL policy is small (few KB), very fast
- Expect 30-50 FPS on Raspberry Pi 4

## API Reference

### AETrainer
```python
trainer = AETrainer(config, output_dir='./outputs')
best_model_path = trainer.train(train_dirs, val_dirs)
```

### RLTrainer
```python
trainer = RLTrainer(rl_config, sim_config, path_config)
trainer.create_model()
trainer.train()
trainer.save_model()
trainer.evaluate(n_episodes=10)
```

### DataCollector
```python
collector = DataCollector('./data/run1')
collector.record_frame(obs, steering, throttle)
collector.close()
```

## References

- [Donkey Car Docs](https://docs.donkeycar.com/)
- [Donkey Car RL (ian0)](https://github.com/ian0/donkeycar-rl)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [World Models](https://worldmodels.github.io/)

## Next Steps

1. **Collect data**: Drive car manually in simulator + real world
2. **Train AE**: `python train_ae.py --train-dirs ...`
3. **Train RL**: `python train_rl.py --algorithm SAC --ae-path ...`
4. **Deploy**: Copy models to Pi, run inference
5. **Iterate**: Collect more data, improve reward function

Good luck! 🚗
