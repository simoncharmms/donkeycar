"""
Data collection helpers for manual driving in simulator or real car.
Saves images + telemetry for later AE training.
"""
import os
import csv
import time
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import gym
from PIL import Image


class DataCollector:
    """Collect driving data (images + telemetry) from manual driving sessions."""
    
    def __init__(self, output_dir: str, env_id: str = "donkey-simulator-v0"):
        """
        Args:
            output_dir: Directory to save collected data
            env_id: Gym environment ID
        """
        self.output_dir = output_dir
        self.env_id = env_id
        
        # Create subdirectories
        self.images_dir = os.path.join(output_dir, 'images')
        self.metadata_path = os.path.join(output_dir, 'telemetry.csv')
        
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Initialize CSV
        self.csv_file = open(self.metadata_path, 'w', newline='')
        self.csv_writer = csv.DictWriter(
            self.csv_file,
            fieldnames=['timestamp', 'image_path', 'steering', 'throttle', 'cte', 'speed']
        )
        self.csv_writer.writeheader()
        
        self.frame_count = 0
    
    def record_frame(self, 
                    observation: np.ndarray,
                    steering: float,
                    throttle: float,
                    cte: float = 0.0,
                    speed: float = 0.0):
        """Record a single frame of data."""
        timestamp = time.time()
        
        # Save image
        image_filename = f"frame_{self.frame_count:06d}.jpg"
        image_path = os.path.join(self.images_dir, image_filename)
        
        # Convert observation to image if needed
        if isinstance(observation, np.ndarray):
            if observation.dtype == np.float32 or observation.dtype == np.float64:
                observation = (observation * 255).astype(np.uint8)
            image = Image.fromarray(observation)
        else:
            image = observation
        
        image.save(image_path)
        
        # Save metadata
        self.csv_writer.writerow({
            'timestamp': timestamp,
            'image_path': image_filename,
            'steering': steering,
            'throttle': throttle,
            'cte': cte,
            'speed': speed,
        })
        self.csv_file.flush()
        
        self.frame_count += 1
    
    def close(self):
        """Close the CSV file."""
        self.csv_file.close()
        print(f"✓ Collected {self.frame_count} frames")
        print(f"  Images: {self.images_dir}")
        print(f"  Telemetry: {self.metadata_path}")


class ManualDrivingSession:
    """Manual driving session for data collection."""
    
    def __init__(self, 
                 output_dir: str,
                 env_kwargs: Dict = None):
        """
        Args:
            output_dir: Directory to save data
            env_kwargs: Arguments for environment creation
        """
        self.output_dir = output_dir
        self.env_kwargs = env_kwargs or {}
        
        self.collector = DataCollector(output_dir)
        self.env = None
    
    def create_env(self, env_id: str = "donkey-simulator-v0"):
        """Create the gym environment."""
        try:
            self.env = gym.make(env_id, **self.env_kwargs)
            print(f"✓ Environment created: {env_id}")
        except Exception as e:
            print(f"✗ Failed to create environment: {e}")
            raise
    
    def run_interactive(self, duration_seconds: Optional[int] = None):
        """
        Run interactive manual driving session.
        
        Note: This requires a human to manually control the car using
        a joystick/gamepad connected to the simulator or real car.
        """
        if self.env is None:
            self.create_env()
        
        obs = self.env.reset()
        start_time = time.time()
        
        try:
            while True:
                # Check duration
                if duration_seconds and (time.time() - start_time) > duration_seconds:
                    break
                
                # Get action from environment (manual control)
                # This is handled by the Donkey Car framework
                action = self.env.action_space.sample()  # Placeholder
                
                # Step
                obs, reward, done, info = self.env.step(action)
                
                # Extract telemetry from info if available
                steering = info.get('steering', 0.0)
                throttle = info.get('throttle', 0.0)
                cte = info.get('cte', 0.0)
                speed = info.get('speed', 0.0)
                
                # Record
                self.collector.record_frame(obs, steering, throttle, cte, speed)
                
                if done:
                    obs = self.env.reset()
                
                print(f"Frame {self.collector.frame_count}: steer={steering:.2f}, throttle={throttle:.2f}")
        
        except KeyboardInterrupt:
            print("\n⚠ Session interrupted")
        
        finally:
            self.collector.close()
            self.env.close()


def split_dataset(data_dir: str, train_ratio: float = 0.8) -> tuple:
    """
    Split collected data into train/val sets.
    
    Args:
        data_dir: Directory with collected images
        train_ratio: Fraction for training (rest goes to val)
    
    Returns:
        (train_images, val_images)
    """
    images_dir = os.path.join(data_dir, 'images')
    all_images = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
    
    n_train = int(len(all_images) * train_ratio)
    train_images = all_images[:n_train]
    val_images = all_images[n_train:]
    
    return train_images, val_images
