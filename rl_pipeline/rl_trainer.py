"""
RL training harness for Donkey Car.
Supports SAC and PPO algorithms with evaluation, callbacks, and monitoring.
"""
import os
import time
from typing import Optional, Dict, Any

import numpy as np
import torch
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback, CallbackList, EvalCallback, BaseCallback
)
from stable_baselines3.sac import MlpPolicy as SACPolicy
from stable_baselines3.ppo import MlpPolicy as PPOPolicy
import gym
from torch.utils.tensorboard import SummaryWriter

from config import RLConfig, SimulatorConfig, PathConfig


class TensorboardCallback(BaseCallback):
    """Custom callback for additional tensorboard logging."""
    
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            mean_length = np.mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
            
            self.logger.record("episode/mean_reward", mean_reward)
            self.logger.record("episode/mean_length", mean_length)
        
        return True


class RLTrainer:
    """Trainer for RL agents on Donkey Car."""
    
    def __init__(self, 
                 rl_config: RLConfig,
                 sim_config: SimulatorConfig,
                 path_config: PathConfig,
                 env_id: str = "donkey-simulator-v0",
                 ae_encoder = None):
        """
        Args:
            rl_config: RL training configuration
            sim_config: Simulator configuration
            path_config: Path configuration
            env_id: OpenAI Gym environment ID
            ae_encoder: Optional autoencoder for state compression
        """
        self.rl_config = rl_config
        self.sim_config = sim_config
        self.path_config = path_config
        self.env_id = env_id
        self.ae_encoder = ae_encoder
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.env = None
        self.run_id = int(time.time())
    
    def create_environment(self, monitor: bool = True) -> gym.Env:
        """Create and wrap the training environment."""
        # Create base environment
        env_kwargs = {
            'exe_path': self.sim_config.exe_path,
            'host': self.sim_config.host,
            'port': self.sim_config.port,
            'car_name': self.sim_config.car_name,
            'max_cte': self.sim_config.max_cte,
        }
        
        try:
            env = gym.make(self.env_id, **env_kwargs)
        except Exception as e:
            print(f"Warning: Could not create environment with ID {self.env_id}: {e}")
            print("Make sure the Donkey Car gym is installed and registered.")
            raise
        
        # Apply wrappers if using AE
        if self.ae_encoder is not None:
            env = self._apply_ae_wrapper(env)
        
        # Monitor
        if monitor:
            monitor_dir = os.path.join(self.path_config.logs_dir, f"monitor_{self.run_id}")
            os.makedirs(monitor_dir, exist_ok=True)
            env = gym.wrappers.Monitor(
                env,
                monitor_dir,
                force=True,
                video_callable=lambda ep: ep % 10 == 0,
            )
        
        return env
    
    def _apply_ae_wrapper(self, env: gym.Env) -> gym.Env:
        """Apply autoencoder wrapper to environment."""
        # This is a placeholder - implement based on your AE wrapper
        # You may want to create a custom wrapper class
        class AEWrapper(gym.Wrapper):
            def __init__(self, env, encoder):
                super().__init__(env)
                self.encoder = encoder
                # Update observation space to match latent dimension
                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(self.encoder.z_size,),
                    dtype=np.float32
                )
            
            def _encode_observation(self, obs):
                with torch.no_grad():
                    if len(obs.shape) == 3:
                        obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
                    else:
                        obs_tensor = torch.from_numpy(obs).float()
                    mu, _ = self.encoder.encode(obs_tensor)
                    return mu.squeeze(0).cpu().numpy()
            
            def step(self, action):
                obs, reward, done, info = self.env.step(action)
                encoded_obs = self._encode_observation(obs)
                return encoded_obs, reward, done, info
            
            def reset(self):
                obs = self.env.reset()
                return self._encode_observation(obs)
        
        return AEWrapper(env, self.ae_encoder)
    
    def setup_callbacks(self) -> CallbackList:
        """Setup training callbacks."""
        callbacks = []
        
        # Checkpoint callback
        checkpoint_dir = os.path.join(
            self.path_config.logs_dir,
            f"{self.rl_config.algorithm.lower()}_{self.run_id}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_callback = CheckpointCallback(
            save_freq=1000,
            save_path=checkpoint_dir,
            name_prefix=f"{self.rl_config.algorithm.lower()}_model"
        )
        callbacks.append(checkpoint_callback)
        
        # Evaluation callback (optional)
        eval_dir = os.path.join(
            self.path_config.logs_dir,
            f"eval_{self.run_id}"
        )
        os.makedirs(eval_dir, exist_ok=True)
        
        eval_callback = EvalCallback(
            self.env,
            best_model_save_path=eval_dir,
            log_path=eval_dir,
            eval_freq=self.rl_config.eval_freq,
            n_eval_episodes=self.rl_config.n_eval_episodes,
            deterministic=False,
        )
        callbacks.append(eval_callback)
        
        # Tensorboard callback
        tb_callback = TensorboardCallback()
        callbacks.append(tb_callback)
        
        return CallbackList(callbacks)
    
    def create_model(self):
        """Create the RL model (SAC or PPO)."""
        if self.env is None:
            self.env = self.create_environment()
        
        # Policy kwargs
        policy_kwargs = {
            'net_arch': [64, 64],
            'activation_fn': torch.nn.ReLU,
        }
        
        tensorboard_log = os.path.join(
            self.path_config.tensorboard_dir,
            f"{self.rl_config.algorithm.lower()}_{self.run_id}"
        )
        os.makedirs(tensorboard_log, exist_ok=True)
        
        if self.rl_config.algorithm.upper() == "SAC":
            policy_kwargs['use_sde'] = self.rl_config.use_sde
            
            self.model = SAC(
                SACPolicy,
                self.env,
                policy_kwargs=policy_kwargs,
                learning_rate=self.rl_config.learning_rate,
                batch_size=self.rl_config.batch_size,
                buffer_size=self.rl_config.buffer_size,
                learning_starts=self.rl_config.learning_starts,
                gradient_steps=self.rl_config.gradient_steps,
                train_freq=self.rl_config.train_freq,
                ent_coef=self.rl_config.ent_coef,
                gamma=self.rl_config.gamma,
                tau=self.rl_config.tau,
                use_sde_at_warmup=True,
                sde_sample_freq=self.rl_config.sde_sample_freq,
                tensorboard_log=tensorboard_log,
                verbose=1,
                device=self.device,
                seed=self.rl_config.seed,
            )
        
        elif self.rl_config.algorithm.upper() == "PPO":
            self.model = PPO(
                PPOPolicy,
                self.env,
                policy_kwargs=policy_kwargs,
                learning_rate=self.rl_config.learning_rate,
                n_steps=self.rl_config.n_steps,
                batch_size=self.rl_config.batch_size,
                n_epochs=self.rl_config.n_epochs,
                gamma=self.rl_config.gamma,
                gae_lambda=self.rl_config.gae_lambda,
                clip_range=self.rl_config.clip_range,
                tensorboard_log=tensorboard_log,
                verbose=1,
                device=self.device,
                seed=self.rl_config.seed,
            )
        
        else:
            raise ValueError(f"Unknown algorithm: {self.rl_config.algorithm}")
        
        print(f"✓ Created {self.rl_config.algorithm} model on {self.device}")
    
    def train(self, total_timesteps: Optional[int] = None):
        """Train the model."""
        if self.model is None:
            self.create_model()
        
        if total_timesteps is None:
            total_timesteps = self.rl_config.total_timesteps
        
        callbacks = self.setup_callbacks()
        
        print(f"\n{'='*60}")
        print(f"Starting {self.rl_config.algorithm} training")
        print(f"Total timesteps: {total_timesteps}")
        print(f"Environment: {self.env_id}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                log_interval=10,
            )
            print(f"\n✓ Training complete!")
        
        except KeyboardInterrupt:
            print(f"\n⚠ Training interrupted by user")
        
        finally:
            self.save_model()
    
    def save_model(self, name: Optional[str] = None):
        """Save the trained model."""
        if self.model is None:
            print("⚠ No model to save")
            return
        
        if name is None:
            name = f"{self.rl_config.algorithm.lower()}_{self.run_id}"
        
        model_path = os.path.join(self.path_config.rl_models_dir, name)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        self.model.save(model_path)
        print(f"✓ Model saved to {model_path}")
        
        # Save metadata
        metadata = {
            'algorithm': self.rl_config.algorithm,
            'total_timesteps': self.rl_config.total_timesteps,
            'timestamp': self.run_id,
            'env_id': self.env_id,
        }
        
        import json
        metadata_path = f"{model_path}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        return model_path
    
    def load_model(self, model_path: str):
        """Load a trained model."""
        if self.rl_config.algorithm.upper() == "SAC":
            self.model = SAC.load(model_path, device=self.device)
        elif self.rl_config.algorithm.upper() == "PPO":
            self.model = PPO.load(model_path, device=self.device)
        else:
            raise ValueError(f"Unknown algorithm: {self.rl_config.algorithm}")
        
        print(f"✓ Model loaded from {model_path}")
    
    def evaluate(self, n_episodes: int = 10, deterministic: bool = True):
        """Evaluate the trained model."""
        if self.model is None:
            raise ValueError("No model to evaluate. Train or load a model first.")
        
        if self.env is None:
            self.env = self.create_environment(monitor=False)
        
        episode_rewards = []
        episode_lengths = []
        
        for ep in range(n_episodes):
            obs = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            print(f"Episode {ep+1}/{n_episodes} | Reward: {episode_reward:.2f} | Length: {episode_length}")
        
        print(f"\nMean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Mean length: {np.mean(episode_lengths):.2f}")
        
        return episode_rewards, episode_lengths
    
    def close(self):
        """Close the environment."""
        if self.env is not None:
            self.env.close()
