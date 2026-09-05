# RL Pipeline Setup Guide

## Prerequisites

### Mac (Development)
- Python 3.8+
- Git
- Homebrew (optional, for system deps)

### Raspberry Pi (Deployment)
- Raspberry Pi 4B+ recommended (4GB+ RAM)
- Raspbian OS
- Python 3.9+
- Camera module + motor controllers

## Installation

### 1. Clone & Setup Workspace

```bash
cd /path/to/donkeycar
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Core dependencies
pip install --upgrade pip
pip install numpy torch torchvision torchaudio
pip install stable-baselines3 gym
pip install pillow tqdm pyyaml loguru

# Optional: GPU support (install CUDA before torch for performance)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Donkey Car (if not already installed)
pip install donkeycar
```

### 3. Download Simulator (Mac/Linux)

**macOS:**
```bash
# Visit https://docs.donkeycar.com/guide/simulator/ and download DonkeySimMac
unzip DonkeySimMac.zip
mv DonkeySimMac ~/Applications/
```

**Linux:**
```bash
# Visit https://docs.donkeycar.com/guide/simulator/ and download Linux binary
chmod +x donkey_sim.x86_64
export SIM_PATH="$(pwd)/donkey_sim.x86_64"
```

### 4. Verify Installation

```bash
python3 -c "import torch; print(torch.__version__); print(f'CUDA: {torch.cuda.is_available()}')"
python3 -c "import stable_baselines3; print(stable_baselines3.__version__)"
python3 -c "import gym; print(gym.__version__)"
```

## Project Setup

### 1. Create Data Directories

```bash
mkdir -p data/{simulator,real_car}
mkdir -p ae_outputs rl_outputs
```

### 2. Configure Paths

Edit `rl_pipeline/config.py` or create a config file:

```yaml
# config.yaml
simulator:
  exe_path: "/Applications/DonkeySimMac/donkey_sim.app/Contents/MacOS/donkey_sim"  # Mac
  # exe_path: "/home/user/donkey_sim.x86_64"  # Linux
  host: "127.0.0.1"
  port: 9091
  max_cte: 4.0

ae:
  z_size: 64
  epochs: 100
  batch_size: 32
  learning_rate: 0.0001

rl:
  algorithm: "SAC"
  total_timesteps: 50000
  learning_rate: 0.0003
```

## Quick Test

### Test AE Training

```bash
# Create dummy data
mkdir -p test_data/images
# Add some .jpg images to test_data/images/

# Run short training
python train_ae.py \
    --train-dirs test_data/images \
    --val-dirs test_data/images \
    --epochs 2 \
    --batch-size 4
```

Expected output:
```
Training on X images, validating on Y images
Device: cpu (or cuda if available)
Epoch 1/2 | Train Loss: 1234.56 | Val Loss: 1200.34
✓ Best model saved: ./ae_outputs/vae_64_1234567890_best.pt
```

### Test RL Training

```bash
# Make sure simulator is NOT running for this test
python train_rl.py \
    --algorithm SAC \
    --total-timesteps 1000 \
    --batch-size 32 \
    2>&1 | head -50
```

## Raspberry Pi Setup

### 1. Initial Setup

```bash
# SSH into Pi
ssh pi@<pi_ip>

# Install system deps
sudo apt update
sudo apt install python3-dev python3-pip python3-venv
sudo apt install libjpeg-dev zlib1g-dev  # For PIL

# Create project
mkdir -p ~/donkey_car_rl
cd ~/donkey_car_rl
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Minimal Runtime

```bash
# Only need inference, not training
pip install --upgrade pip
pip install numpy torch torchvision
pip install stable-baselines3
pip install pillow

# Donkey Car
pip install donkeycar
```

### 3. Copy Models

```bash
# On Mac, after training
scp ae_outputs/vae_64_*_best.pt pi@<pi_ip>:~/donkey_car_rl/models/
scp rl_outputs/rl_models/sac_* pi@<pi_ip>:~/donkey_car_rl/models/
```

### 4. Inference Script

Create `pi_drive.py`:

```python
import torch
from stable_baselines3 import SAC
from rl_pipeline.ae_trainer import VAE
from donkeycar.vehicle import Vehicle

# Load models
ae = VAE(z_size=64)
ae.load('./models/vae_64_best.pt', device=torch.device('cpu'))

policy = SAC.load('./models/sac_best')

# Setup vehicle
v = Vehicle()
# ... add camera, motors, etc

# Drive loop
while True:
    image = v.camera.capture()
    latent = ae.encode_raw_image(image)
    action, _ = policy.predict(latent, deterministic=True)
    
    # Action = [steering, throttle]
    v.steering.set(action[0])
    v.throttle.set(action[1])
```

## TensorBoard Monitoring

```bash
# During training
tensorboard --logdir ./rl_outputs/tensorboard --port 6006
# Open http://localhost:6006
```

## Troubleshooting

### CUDA Issues
```bash
# Check GPU
python3 -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall with CPU-only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Simulator Connection
```bash
# Start simulator manually
/path/to/donkey_sim --port 9091

# Then in another terminal, test connection
python3 -c "import gym; env = gym.make('donkey-simulator-v0'); print(env.reset())"
```

### Out of Memory
- Reduce `batch_size` (8 or 16)
- Reduce `buffer_size` (10000)
- Use CPU instead of GPU

### Slow Training
- Check device: `print(torch.cuda.is_available())`
- Reduce image resolution in preprocessing
- Use smaller AE model (z_size=32)

## Next Steps

1. ✓ Install dependencies
2. ✓ Download simulator
3. ✓ Run quick tests
4. → Collect training data
5. → Train AE
6. → Train RL agent
7. → Deploy to Pi

See [RL_PIPELINE.md](RL_PIPELINE.md) for full workflow.
