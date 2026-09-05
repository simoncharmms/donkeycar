#!/usr/bin/env python3
"""
Example: Full workflow from data collection to RL training.
Shows how all components work together.
"""
import os
from pathlib import Path

from rl_pipeline.config import (
    PipelineConfig, SimulatorConfig, AEConfig, RLConfig, PathConfig
)
from rl_pipeline.ae_trainer import AETrainer
from rl_pipeline.rl_trainer import RLTrainer
from rl_pipeline.data_collector import DataCollector


def example_config():
    """Create example configuration."""
    config = PipelineConfig(
        simulator=SimulatorConfig(
            exe_path="/Applications/DonkeySimMac/donkey_sim.app/Contents/MacOS/donkey_sim",
            host="127.0.0.1",
            port=9091,
            max_cte=4.0,
        ),
        ae=AEConfig(
            z_size=64,
            c_hid=64,
            learning_rate=1e-4,
            batch_size=32,
            epochs=100,
            noise_factor=0.3,
        ),
        rl=RLConfig(
            algorithm="SAC",
            total_timesteps=50000,
            learning_rate=3e-4,
            batch_size=64,
            eval_freq=50,
        ),
        paths=PathConfig(base_dir="./rl_outputs")
    )
    
    return config


def workflow_step_1_collect_data():
    """Step 1: Collect training data from manual driving."""
    print("\n" + "="*60)
    print("STEP 1: Collect Training Data")
    print("="*60)
    
    # In real usage, you would manually drive the car in the simulator
    # or real world. This example just creates dummy data.
    
    data_dir = "./data/example_run"
    os.makedirs(data_dir, exist_ok=True)
    
    collector = DataCollector(data_dir)
    
    # Simulate collecting 100 frames
    import numpy as np
    for i in range(100):
        # Dummy observation (would be real camera image)
        obs = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
        
        # Dummy telemetry
        steering = np.random.uniform(-1, 1)
        throttle = np.random.uniform(-1, 1)
        
        collector.record_frame(obs, steering, throttle, cte=0.0, speed=1.0)
        
        if (i + 1) % 25 == 0:
            print(f"  Collected {i+1}/100 frames")
    
    collector.close()
    print(f"\n✓ Data saved to {data_dir}")
    return data_dir


def workflow_step_2_train_ae(config: PipelineConfig, data_dir: str):
    """Step 2: Train autoencoder on collected images."""
    print("\n" + "="*60)
    print("STEP 2: Train Autoencoder")
    print("="*60)
    
    images_dir = os.path.join(data_dir, 'images')
    
    # For real training, you'd split into train/val
    train_dirs = [images_dir]
    val_dirs = [images_dir]  # In practice, use separate data
    
    trainer = AETrainer(config.ae, output_dir=config.paths.ae_models_dir)
    
    print(f"Training AE on {images_dir}")
    print(f"Config: z_size={config.ae.z_size}, epochs={config.ae.epochs}")
    
    # In real usage:
    # best_model_path = trainer.train(train_dirs, val_dirs)
    
    # For this example, just print what would happen
    print("\nAE Training would:")
    print("  1. Load images from train_dirs")
    print("  2. Add Gaussian noise for robustness")
    print("  3. Train VAE with reconstruction + KL loss")
    print("  4. Validate on val_dirs")
    print("  5. Save best model on checkpoint")
    print("  6. Log to TensorBoard")
    
    # Dummy path for next step
    best_model_path = f"{config.paths.ae_models_dir}/vae_64_1234567890_best.pt"
    
    print(f"\n✓ AE training complete (would save to {best_model_path})")
    return best_model_path


def workflow_step_3_train_rl(config: PipelineConfig, ae_model_path: str):
    """Step 3: Train RL agent using trained AE."""
    print("\n" + "="*60)
    print("STEP 3: Train RL Agent")
    print("="*60)
    
    print(f"Config:")
    print(f"  Algorithm: {config.rl.algorithm}")
    print(f"  Total timesteps: {config.rl.total_timesteps}")
    print(f"  Learning rate: {config.rl.learning_rate}")
    print(f"  Batch size: {config.rl.batch_size}")
    print(f"  AE path: {ae_model_path}")
    
    # In real usage:
    # trainer = RLTrainer(
    #     rl_config=config.rl,
    #     sim_config=config.simulator,
    #     path_config=config.paths,
    #     env_id='donkey-simulator-v0',
    #     ae_encoder=loaded_ae,  # Use trained AE
    # )
    # trainer.train()
    # model_path = trainer.save_model()
    
    print("\nRL Training would:")
    print("  1. Load AE encoder")
    print("  2. Create Donkey Car gym environment")
    print("  3. Wrap env with AE encoder")
    print("  4. Create SAC/PPO agent")
    print("  5. Train with callbacks (checkpoints, eval, tensorboard)")
    print("  6. Save best model")
    
    model_path = f"{config.paths.rl_models_dir}/sac_1234567890"
    print(f"\n✓ RL training complete (would save to {model_path})")
    
    return model_path


def workflow_step_4_deploy():
    """Step 4: Deploy to Raspberry Pi."""
    print("\n" + "="*60)
    print("STEP 4: Deploy to Raspberry Pi")
    print("="*60)
    
    print("""
Deployment Steps:
  1. Copy AE model to Pi:
     $ scp ae_outputs/vae_64_*_best.pt pi@<pi_ip>:~/models/
  
  2. Copy RL model to Pi:
     $ scp rl_outputs/rl_models/sac_* pi@<pi_ip>:~/models/
  
  3. Create inference script (pi_drive.py):
     - Load AE + RL models
     - Connect to camera
     - Run inference loop
     - Send control commands to motors
  
  4. Run on Pi:
     $ ssh pi@<pi_ip>
     $ python pi_drive.py
    """)


def main():
    """Run full workflow example."""
    print("\n" + "="*60)
    print("Donkey Car RL Pipeline - Full Workflow Example")
    print("="*60)
    
    # Step 0: Create configuration
    config = example_config()
    print("\n✓ Configuration created")
    print(f"  Output dir: {config.paths.base_dir}")
    print(f"  AE latent dim: {config.ae.z_size}")
    print(f"  RL algorithm: {config.rl.algorithm}")
    
    # Step 1: Collect data
    data_dir = workflow_step_1_collect_data()
    
    # Step 2: Train AE
    ae_model_path = workflow_step_2_train_ae(config, data_dir)
    
    # Step 3: Train RL
    rl_model_path = workflow_step_3_train_rl(config, ae_model_path)
    
    # Step 4: Deploy
    workflow_step_4_deploy()
    
    # Summary
    print("\n" + "="*60)
    print("WORKFLOW COMPLETE")
    print("="*60)
    print("""
Files created:
  ✓ Training data: ./data/example_run/
  ✓ AE model: ./rl_outputs/ae_models/vae_64_*_best.pt
  ✓ RL model: ./rl_outputs/rl_models/sac_*
  
Logs:
  ✓ TensorBoard: ./rl_outputs/tensorboard/
  ✓ Monitor: ./rl_outputs/logs/
  
Next:
  1. Review TensorBoard logs
  2. Test trained model on Pi
  3. Fine-tune hyperparameters if needed
  4. Collect more diverse data
  5. Retrain with improved dataset
    """)


if __name__ == "__main__":
    main()
