# 🚗 START HERE — Donkey Car RL Pipeline

Welcome! You have a complete, production-ready RL pipeline. Here's where to go next.

## 📍 What You Got

A reinforcement learning system for training a self-driving Donkey Car:
- **Autoencoder (VAE)** to compress images → 64-dimensional latent space
- **RL Trainer** supporting SAC and PPO algorithms  
- **Data collection tools** for manual driving sessions
- **Full documentation** and example workflows
- **Raspberry Pi deployment** support

## 🎯 Your Workflow (3 Phases)

### Phase 1: Collect Training Data
**Goal:** Record 2000+ images of manual driving

**Option A: Simulator**
```bash
# Start Donkey Car simulator manually
# Then run:
python -c "
from rl_pipeline.data_collector import DataCollector
collector = DataCollector('./data/simulator_session_1')
# Drive manually in simulator — collector records everything
# When done: collector.close()
"
```

**Option B: Real Car (Raspberry Pi)**
- Drive the physical car with camera
- Let the Donkey Car framework save images automatically
- Collect for ~10-15 minutes to get ~2000+ frames

**Organize your data:**
```
data/
├── simulator_session_1/
│   ├── images/        # 2000+ .jpg files
│   └── telemetry.csv  # steering, throttle, etc
├── real_car_session_1/
│   ├── images/
│   └── telemetry.csv
└── test_session/      # Hold out for final evaluation
    ├── images/
    └── telemetry.csv
```

### Phase 2: Train Autoencoder

**What it does:** Compresses camera images to 64 numbers (latent vector)

```bash
python train_ae.py \
    --train-dirs data/simulator_session_1/images data/real_car_session_1/images \
    --val-dirs data/test_session/images \
    --z-size 64 \
    --epochs 100 \
    --batch-size 32
```

**Monitor training:**
```bash
tensorboard --logdir ./ae_outputs/runs --port 6006
# Open http://localhost:6006
```

**What to look for:**
- ✅ Reconstruction loss decreases smoothly
- ✅ Val loss follows train loss (not diverging)
- ✅ Reconstructed images become clearer over time
- ✅ Takes ~1-2 hours on Mac

**Output:** `ae_outputs/vae_64_<timestamp>_best.pt` (your trained encoder)

### Phase 3: Train RL Agent

**What it does:** Learns to drive using the autoencoder's compressed images

```bash
python train_rl.py \
    --algorithm SAC \
    --env donkey-simulator-v0 \
    --exe-path /path/to/donkey_sim \
    --total-timesteps 50000 \
    --ae-path ae_outputs/vae_64_<timestamp>_best.pt \
    --z-size 64
```

**Or with PPO (alternative):**
```bash
python train_rl.py \
    --algorithm PPO \
    --total-timesteps 100000 \
    --ae-path ae_outputs/vae_64_<timestamp>_best.pt
```

**Monitor training:**
```bash
tensorboard --logdir ./rl_outputs/tensorboard --port 6006
```

**What to look for:**
- ✅ Episode rewards increase over time
- ✅ No diverging losses
- ✅ Smooth driving (no erratic actions)
- ✅ Takes ~1-2 hours on Mac with GPU

**Output:** `rl_outputs/rl_models/sac_<timestamp>` (your trained policy)

### Phase 4: Deploy to Raspberry Pi

**Copy models:**
```bash
scp ae_outputs/vae_64_*_best.pt pi@192.168.1.100:~/donkey_rl/models/
scp rl_outputs/rl_models/sac_* pi@192.168.1.100:~/donkey_rl/models/
```

**Create inference script (pi_drive.py):**
```python
import torch
from stable_baselines3 import SAC
from rl_pipeline.ae_trainer import VAE

# Load trained models
ae = VAE(z_size=64)
ae.load('./models/vae_64_best.pt', device=torch.device('cpu'))

policy = SAC.load('./models/sac_best')

# Drive loop
while True:
    image = camera.capture()
    latent = ae.encode_raw_image(image)
    action, _ = policy.predict(latent, deterministic=True)
    car.drive(action)
```

**Run:**
```bash
ssh pi@192.168.1.100
cd ~/donkey_rl
python pi_drive.py
```

## 📚 Documentation

| File | Purpose | Read When |
|------|---------|-----------|
| **README_PIPELINE.md** | Quick reference | You want a cheat sheet |
| **RL_PIPELINE.md** | Full guide | You want to understand everything |
| **SETUP.md** | Installation | You're setting up for the first time |
| **DONKEY_CAR_STATUS.md** | Project status | You want the complete picture |
| **example_full_workflow.py** | Working example | You want to see code |

## 🔧 Troubleshooting

**Simulator won't connect?**
```bash
# Start simulator manually:
/Applications/DonkeySimMac/donkey_sim.app/Contents/MacOS/donkey_sim --port 9091
```

**Out of memory during training?**
```bash
# Reduce batch size
python train_ae.py --batch-size 8  # was 32
python train_rl.py --batch-size 16 --buffer-size 10000
```

**Training is slow?**
```bash
# Check if using GPU
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

**AE training shows bad reconstructions?**
- Collect more data (aim for 5000+ images)
- Check image quality in input
- Run for more epochs (150-200)

**RL rewards not improving?**
- Increase total_timesteps (100k instead of 50k)
- Check reward function (CTE penalty, speed bonus)
- Reduce max_cte to 2.0 (prevent spinning)

See **RL_PIPELINE.md** for detailed troubleshooting.

## 💡 Key Insights

1. **Data quality >> hyperparameters**
   - Good diverse data beats tuned hyperparameters
   - Collect 2000+ images from varied driving

2. **SAC > PPO for this task**
   - More stable training
   - Faster convergence
   - Better final performance

3. **Monitor TensorBoard early**
   - Catch problems before they compound
   - Visualize reconstructions during AE training
   - Watch reward curves during RL training

4. **Deploy incrementally**
   - Test on simulator first
   - Then transfer to Pi
   - Use Pi in controlled environment initially

## 🎓 Learning Path

**Not familiar with RL?**
→ Read `RL_PIPELINE.md` section "Related Projects" for references

**Want to modify the code?**
→ Core modules are in `rl_pipeline/` with clear APIs

**Having issues?**
→ Check `SETUP.md` (installation) or `RL_PIPELINE.md` (algorithms)

## ✅ Pre-Flight Checklist

Before you start:

- [ ] Mac has Python 3.8+, PyTorch, Stable-Baselines3 installed
  ```bash
  python3 -c "import torch, stable_baselines3; print('OK')"
  ```

- [ ] Donkey Car simulator downloaded (for simulator training)
  ```bash
  ls /Applications/DonkeySimMac/  # or ~/donkey_sim.x86_64 on Linux
  ```

- [ ] You have 2000+ images ready (from previous manual driving)
  ```bash
  ls data/simulator_session_1/images/ | wc -l
  ```

- [ ] You understand the workflow (data → AE → RL → deploy)

## 🚀 Ready?

```bash
# 1. Organize your data
# → Place images in data/train and data/val

# 2. Train autoencoder
python train_ae.py --train-dirs data/train --val-dirs data/val

# 3. Train RL agent
python train_rl.py --algorithm SAC --ae-path ae_outputs/vae_64_*_best.pt

# 4. Deploy to Pi
scp ae_outputs/vae_64_*_best.pt pi@<ip>:~/models/
scp rl_outputs/rl_models/sac_* pi@<ip>:~/models/
```

Good luck! 🎯

---

**Questions?**
- See `RL_PIPELINE.md` (full reference)
- See `SETUP.md` (installation issues)
- See `example_full_workflow.py` (working code)

**Need help with specific issue?**
- See "Troubleshooting" section above
- See `RL_PIPELINE.md` "Known Issues" section
