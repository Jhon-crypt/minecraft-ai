import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import os
from texture_generator import MinecraftTextureGenerator

class MinecraftDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.textures_path = self.dataset_path / 'textures'
        
        if not self.textures_path.exists():
            raise ValueError(f"Textures directory not found at {self.textures_path}")
            
        self.image_files = []
        for ext in ['*.png', '*.jpg']:
            self.image_files.extend(list(self.textures_path.rglob(ext)))
            
        print(f"Looking for textures in: {self.textures_path}")
        print(f"Found {len(self.image_files)} texture files")
        
        if len(self.image_files) == 0:
            raise ValueError(f"No texture files found in {self.textures_path}")
        
        self.transform = transforms.Compose([
            transforms.Resize((16, 16)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return torch.zeros(3, 16, 16)

class TextureTrainer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.texture_generator = MinecraftTextureGenerator().to(device)
        self.optimizer = optim.Adam(self.texture_generator.parameters(), lr=0.0002)
        self.criterion = nn.MSELoss()

    def train(self, dataset, num_epochs=100, batch_size=64):
        """Train the texture generator"""
        if len(dataset) == 0:
            raise ValueError("Dataset is empty!")
            
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print(f"\nStarting training with:")
        print(f"- {len(dataset)} texture files")
        print(f"- {num_epochs} epochs")
        print(f"- Batch size: {batch_size}")
        print(f"- Device: {self.device}")
        
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_idx, textures in enumerate(dataloader):
                textures = textures.to(self.device)
                batch_size = textures.size(0)

                z = torch.randn(batch_size, self.texture_generator.latent_dim).to(self.device)

                self.optimizer.zero_grad()
                generated = self.texture_generator(z)
                loss = self.criterion(generated, textures)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                if batch_idx % 10 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(dataloader)}] '
                          f'Loss: {loss.item():.4f}')

            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1)

            avg_loss = total_loss / len(dataloader)
            print(f'Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}')

    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        checkpoint_dir = 'checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f'texture_generator_epoch_{epoch:03d}.pth')
        torch.save(self.texture_generator.state_dict(), checkpoint_path)
        print(f'Checkpoint saved: {checkpoint_path}')