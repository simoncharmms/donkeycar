"""
Autoencoder (VAE) training pipeline for Donkey Car.
Handles image loading, training, validation, and checkpointing.
"""
import os
import time
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from config import AEConfig


class VAE(nn.Module):
    """Variational Autoencoder for image compression."""
    
    def __init__(self, z_size: int = 64, c_hid: int = 64, 
                 num_image_channels: int = 3, learning_rate: float = 1e-4):
        super().__init__()
        self.z_size = z_size
        self.c_hid = c_hid
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(num_image_channels, c_hid, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(c_hid, c_hid * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(c_hid * 2, c_hid * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(c_hid * 4, c_hid * 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # Latent space
        self.fc_mu = nn.Linear(c_hid * 8 * 5 * 10, z_size)
        self.fc_logvar = nn.Linear(c_hid * 8 * 5 * 10, z_size)
        
        # Decoder
        self.fc_decode = nn.Linear(z_size, c_hid * 8 * 5 * 10)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c_hid * 8, c_hid * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(c_hid * 4, c_hid * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(c_hid * 2, c_hid, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(c_hid, num_image_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode image to latent distribution."""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to image."""
        h = self.fc_decode(z)
        h = h.view(-1, self.c_hid * 8, 5, 10)
        return self.decoder(h)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: encode -> reparameterize -> decode."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    def encode_raw_image(self, image: Image.Image) -> torch.Tensor:
        """Encode a PIL image directly."""
        transform = transforms.Compose([
            transforms.Resize((80, 160)),
            transforms.ToTensor(),
        ])
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            mu, logvar = self.encode(image_tensor)
            z = self.reparameterize(mu, logvar)
        return z
    
    def decode_forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector."""
        with torch.no_grad():
            return self.decode(z)
    
    @staticmethod
    def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, 
                 logvar: torch.Tensor) -> torch.Tensor:
        """VAE loss: reconstruction + KL divergence."""
        bce = nn.BCELoss(reduction='sum')(recon_x, x)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return bce + kld
    
    def save(self, path: str):
        """Save model weights."""
        torch.save(self.state_dict(), path)
    
    def load(self, path: str, device: torch.device = None):
        """Load model weights."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_state_dict(torch.load(path, map_location=device, weights_only=True))


class ImageDataset(Dataset):
    """Custom dataset for Donkey Car images."""
    
    def __init__(self, image_dirs: List[str], max_samples: int = -1):
        """
        Args:
            image_dirs: List of directories containing .jpg images
            max_samples: Max number of images to load (-1 for all)
        """
        self.images = []
        for img_dir in image_dirs:
            if not os.path.isdir(img_dir):
                continue
            images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                     if f.endswith('.jpg')]
            self.images.extend(images)
        
        if max_samples > 0:
            self.images = self.images[:max_samples]
        
        self.transform = transforms.Compose([
            transforms.Resize((80, 160)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx: int):
        image = Image.open(self.images[idx]).convert('RGB')
        return self.transform(image)


class GaussianNoiseTransform:
    """Add Gaussian noise to images for robustness."""
    
    def __init__(self, mean: float = 0.05, std: float = 0.05):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn_like(tensor) * self.std + self.mean


def add_noise(images: torch.Tensor, noise_factor: float = 0.3) -> torch.Tensor:
    """Add Gaussian noise to a batch of images."""
    gaussian = GaussianNoiseTransform(0.05, 0.05)
    noisy = images + torch.randn_like(images) * noise_factor
    return torch.clamp(noisy, 0.0, 1.0)


class AETrainer:
    """Autoencoder trainer with evaluation and checkpointing."""
    
    def __init__(self, config: AEConfig, output_dir: str = "./ae_outputs"):
        self.config = config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = VAE(
            z_size=config.z_size,
            c_hid=config.c_hid,
            num_image_channels=config.num_image_channels,
            learning_rate=config.learning_rate
        ).to(self.device)
        
        self.ae_id = int(time.time())
        self.writer = SummaryWriter(
            log_dir=os.path.join(output_dir, f'runs/vae_{self.ae_id}')
        )
        
        self.best_loss = np.inf
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, dataloader: TorchDataLoader, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for images in tqdm(dataloader, desc=f"Epoch {epoch+1} Train"):
            # Add noise for robustness
            noisy_images = add_noise(images, self.config.noise_factor)
            
            images = images.to(self.device)
            noisy_images = noisy_images.to(self.device)
            
            # Forward pass
            recon_images, mu, logvar = self.model(noisy_images)
            loss = VAE.vae_loss(recon_images, images, mu, logvar)
            
            # Backward pass
            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader.dataset)
        return avg_loss
    
    def validate(self, dataloader: TorchDataLoader, epoch: int) -> Tuple[float, torch.Tensor]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        sample_recon = None
        
        with torch.no_grad():
            for i, images in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1} Val")):
                noisy_images = add_noise(images, self.config.noise_factor)
                
                images = images.to(self.device)
                noisy_images = noisy_images.to(self.device)
                
                recon_images, mu, logvar = self.model(noisy_images)
                loss = VAE.vae_loss(recon_images, images, mu, logvar)
                total_loss += loss.item()
                
                # Save last batch
                if i == len(dataloader) - 1:
                    sample_recon = recon_images.detach().cpu()
        
        avg_loss = total_loss / len(dataloader.dataset)
        return avg_loss, sample_recon
    
    def train(self, train_dirs: List[str], val_dirs: List[str]):
        """Full training loop."""
        # Load datasets
        train_dataset = ImageDataset(
            train_dirs, 
            max_samples=self.config.n_samples
        )
        val_dataset = ImageDataset(
            val_dirs,
            max_samples=self.config.n_samples
        )
        
        train_loader = TorchDataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_threads if self.config.num_threads > 0 else 0
        )
        val_loader = TorchDataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_threads if self.config.num_threads > 0 else 0
        )
        
        print(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)}")
        print(f"Device: {self.device}")
        
        for epoch in range(self.config.epochs):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, sample_recon = self.validate(val_loader, epoch)
            self.val_losses.append(val_loss)
            
            # Log to tensorboard
            self.writer.add_scalar('loss/train', train_loss, epoch)
            self.writer.add_scalar('loss/val', val_loss, epoch)
            
            if sample_recon is not None:
                grid = make_grid(sample_recon[:8])
                self.writer.add_image('reconstructions', grid, epoch)
            
            print(f"Epoch {epoch+1}/{self.config.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                best_path = os.path.join(
                    self.output_dir,
                    f"vae_{self.config.z_size}_{self.ae_id}_best.pt"
                )
                self.model.save(best_path)
                print(f"✓ Best model saved: {best_path}")
        
        # Final save
        final_path = os.path.join(
            self.output_dir,
            f"vae_{self.config.z_size}_{self.ae_id}_final.pt"
        )
        self.model.save(final_path)
        self.writer.flush()
        self.writer.close()
        
        print(f"\n✓ Training complete. Model saved to {final_path}")
        return best_path
