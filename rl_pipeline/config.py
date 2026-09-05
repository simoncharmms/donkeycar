"""
Configuration management for RL pipeline.
Handles simulator config, training params, paths, etc.
"""
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from pathlib import Path
import json
import yaml


@dataclass
class SimulatorConfig:
    """Donkey Car Simulator configuration."""
    exe_path: str = ""  # Path to donkey_sim executable
    host: str = "127.0.0.1"
    port: int = 9091
    car_name: str = "training"
    max_cte: float = 4.0  # Cross-track error threshold
    max_throttle: float = 1.0
    min_throttle: float = -1.0
    max_steering: float = 1.0
    min_steering: float = -1.0


@dataclass
class AEConfig:
    """Autoencoder training configuration."""
    z_size: int = 64  # Latent dimension
    c_hid: int = 64  # Hidden channel size
    num_image_channels: int = 3
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 100
    num_threads: int = -1
    n_samples: int = -1  # -1 = use all
    noise_factor: float = 0.3


@dataclass
class RLConfig:
    """RL training configuration."""
    algorithm: str = "SAC"  # "SAC" or "PPO"
    total_timesteps: int = 50000
    learning_rate: float = 3e-4
    batch_size: int = 64
    buffer_size: int = 30000
    gamma: float = 0.99
    tau: float = 0.02
    learning_starts: int = 0
    gradient_steps: int = 600
    train_freq: tuple = field(default_factory=lambda: (1, "episode"))
    ent_coef: str = "auto_0.1"
    eval_freq: int = 50
    n_eval_episodes: int = 5
    use_sde: bool = True
    sde_sample_freq: int = 64
    seed: int = 42
    
    # PPO specific
    n_steps: int = 2048
    n_epochs: int = 20
    gae_lambda: float = 0.95
    clip_range: float = 0.2


@dataclass
class PathConfig:
    """Directory structure for outputs."""
    base_dir: str = "./rl_outputs"
    data_dir: str = "data"
    ae_models_dir: str = "ae_models"
    rl_models_dir: str = "rl_models"
    logs_dir: str = "logs"
    training_images_dir: str = "training_images"
    tensorboard_dir: str = "tensorboard"
    
    def __post_init__(self):
        # Create all directories
        for attr in ['data_dir', 'ae_models_dir', 'rl_models_dir', 'logs_dir', 
                     'training_images_dir', 'tensorboard_dir']:
            path = getattr(self, attr)
            if not os.path.isabs(path):
                path = os.path.join(self.base_dir, path)
            os.makedirs(path, exist_ok=True)
            setattr(self, attr, path)


@dataclass
class PipelineConfig:
    """Master configuration combining all components."""
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    ae: AEConfig = field(default_factory=AEConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'PipelineConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        sim_cfg = SimulatorConfig(**data.get('simulator', {}))
        ae_cfg = AEConfig(**data.get('ae', {}))
        rl_cfg = RLConfig(**data.get('rl', {}))
        paths_cfg = PathConfig(**data.get('paths', {}))
        
        return cls(
            simulator=sim_cfg,
            ae=ae_cfg,
            rl=rl_cfg,
            paths=paths_cfg
        )
    
    @classmethod
    def from_json(cls, json_path: str) -> 'PipelineConfig':
        """Load configuration from JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        sim_cfg = SimulatorConfig(**data.get('simulator', {}))
        ae_cfg = AEConfig(**data.get('ae', {}))
        rl_cfg = RLConfig(**data.get('rl', {}))
        paths_cfg = PathConfig(**data.get('paths', {}))
        
        return cls(
            simulator=sim_cfg,
            ae=ae_cfg,
            rl=rl_cfg,
            paths=paths_cfg
        )
    
    def to_yaml(self, output_path: str):
        """Save configuration to YAML."""
        data = {
            'simulator': asdict(self.simulator),
            'ae': asdict(self.ae),
            'rl': asdict(self.rl),
            'paths': asdict(self.paths),
        }
        with open(output_path, 'w') as f:
            yaml.dump(data, f)
    
    def to_json(self, output_path: str):
        """Save configuration to JSON."""
        data = {
            'simulator': asdict(self.simulator),
            'ae': asdict(self.ae),
            'rl': asdict(self.rl),
            'paths': asdict(self.paths),
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)


# Default configs for different scenarios
def get_simulator_config(env_type: str = "simulator") -> SimulatorConfig:
    """Get simulator config for different environments."""
    if env_type == "simulator":
        return SimulatorConfig(
            exe_path="/path/to/donkey_sim.x86_64",  # Set by user
            host="127.0.0.1",
            port=9091,
            car_name="training"
        )
    elif env_type == "real_car":
        return SimulatorConfig(
            host="192.168.1.100",  # Pi IP
            port=8887,
            car_name="donkey_car"
        )
    else:
        raise ValueError(f"Unknown environment type: {env_type}")
