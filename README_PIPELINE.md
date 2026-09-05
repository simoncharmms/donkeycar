# Donkey Car RL Pipeline - Quick Reference

## 📁 Structure

```
donkeycar/
├── rl_pipeline/                 # Core library
│   ├── __init__.py
│   ├── config.py               # Configuration dataclasses
│   ├── ae_trainer.py           # VAE training
│   ├── rl_trainer.py           # RL agent training (SAC/PPO)
│   └── data_collector.py       # Manual driving data collection
│
├── train_ae.py                 # Train autoencoder (CLI)
├── train_rl.py                 # Train RL agent (CLI)
├── example_full_workflow.py    # End-to-end example
│
├── RL_PIPELINE.md              # Full documentation
├── SETUP.md                    # Installation guide
└── config_example.yaml         # Example config
```

## 🚀 Quick Start (3 steps)

### 1️⃣ Train Autoencoder
```bash
python train_ae.py \
    --train-dirs data/train_images \
    --val-dirs data/val_images \
    --z-size 64 \
    --epochs 100
```
**Output:** `ae_outputs/vae_64_<timestamp>_best.pt`

### 2️⃣ Train RL Agent
```bash
python train_rl.py \
    --algorithm SAC \
    --total-timesteps 50000 \
    --ae-path ae_outputs/vae_64_<timestamp>_best.pt
```
**Output:** `rl_outputs/rl_models/sac_<timestamp>`

### 3️⃣ Deploy to Pi
```bash
scp ae_outputs/vae_64_*_best.pt pi@<pi_ip>:~/models/
scp rl_outputs/rl_models/sac_* pi@<pi_ip>:~/models/
```

## 📊 Configuration

All configs in `rl_pipeline/config.py`:

| Component | Key Params | Defaults |
|-----------|-----------|----------|
| **Simulator** | exe_path, host, port, max_cte | 127.0.0.1:9091, cte=4.0 |
| **AE** | z_size, epochs, batch_size, noise_factor | 64, 100, 32, 0.3 |
| **RL** | algorithm, total_timesteps, learning_rate | SAC, 50k, 3e-4 |

## 🎯 Key Components

### AETrainer
```python
from rl_pipeline.ae_trainer import AETrainer
from rl_pipeline.config import AEConfig

config = AEConfig(z_size=64, epochs=100)
trainer = AETrainer(config, output_dir='./ae_outputs')
best_model = trainer.train(train_dirs, val_dirs)
```

### RLTrainer
```python
from rl_pipeline.rl_trainer import RLTrainer
from rl_pipeline.config import RLConfig, SimulatorConfig

rl_cfg = RLConfig(algorithm='SAC', total_timesteps=50000)
sim_cfg = SimulatorConfig(exe_path='/path/to/sim')
trainer = RLTrainer(rl_cfg, sim_cfg, ...)
trainer.train()
trainer.save_model()
```

### DataCollector
```python
from rl_pipeline.data_collector import DataCollector

collector = DataCollector('./data/run1')
collector.record_frame(image, steering, throttle)
collector.close()
```

## 📈 Monitoring

```bash
# During training
tensorboard --logdir ./rl_outputs/tensorboard --port 6006
# Open http://localhost:6006
```

Tracks:
- Loss curves (AE reconstruction, KL divergence)
- Reward over episodes
- Action distributions
- Model checkpoints

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Simulator won't connect | Start simulator manually: `/path/to/donkey_sim --port 9091` |
| Out of memory | Reduce batch_size (8), buffer_size (10k) |
| Slow training | Check GPU: `python -c "import torch; print(torch.cuda.is_available())"` |
| Car learns to spin | Reduce max_cte (try 2.0) |
| AE sees noise | Collect more diverse data (2000+ images) |

## 📚 Documentation

- **RL_PIPELINE.md** — Full pipeline guide, tips, algorithm details
- **SETUP.md** — Installation, dependencies, Pi setup
- **example_full_workflow.py** — End-to-end working example

## 🎓 Learning Path

1. **Understand the pipeline** → Read RL_PIPELINE.md
2. **Install dependencies** → Follow SETUP.md
3. **Run example** → `python example_full_workflow.py`
4. **Collect real data** → Manual drive in simulator/real car
5. **Train AE** → `python train_ae.py --train-dirs ...`
6. **Train RL** → `python train_rl.py --algorithm SAC --ae-path ...`
7. **Deploy** → Copy to Pi and run inference

## 💡 Tips

### Data Collection
- Collect 2000+ images for robust AE
- Diverse driving lines (center, left, right)
- Include recoveries from bad positions

### AE Training
- Monitor TensorBoard for reconstruction quality
- Val loss should plateau (not diverge)
- ~100 epochs typically enough

### RL Training
- SAC is more stable than PPO for this task
- Eval every 50 steps
- Watch rewards — should improve monotonically
- If plateau: adjust reward function, not hyperparams

### Deployment
- AE inference: 10-30 FPS on Pi
- RL policy: <1ms per step
- Total: 20-50 FPS realistic target

## 🔗 References

- [Donkey Car Docs](https://docs.donkeycar.com/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [World Models](https://worldmodels.github.io/)
- [Ian's Donkey Car RL](https://github.com/ian0/donkeycar-rl)

---

**Need help?** Check RL_PIPELINE.md for detailed guide. Good luck! 🚗
