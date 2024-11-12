import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import numpy as np
from PIL import Image
from torchvision import transforms

class MinecraftDataset(Dataset):
    def __init__(self, base_path, transform=None):
        self.base_path = Path(base_path)
        self.transform = transform or transforms.Compose([
            transforms.Resize((16, 16)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Get texture files
        self.texture_files = list((self.base_path / "textures").glob("*.png"))
        if len(self.texture_files) == 0:
            raise ValueError(f"No texture files found in {self.base_path}/textures")
        
        print(f"Found {len(self.texture_files)} texture files")
        
        # Extract labels from filenames
        self.labels = [self._extract_label(f.stem) for f in self.texture_files]
    
    def _extract_label(self, filename):
        # Extract material type from filename (e.g., "wood_planks" -> "wood")
        for material in ["wood", "stone", "dirt", "metal", "gold", "diamond"]:
            if material in filename.lower():
                return material
        return "other"
    
    def __len__(self):
        return len(self.texture_files)
    
    def __getitem__(self, idx):
        texture_path = self.texture_files[idx]
        label = self.labels[idx]
        
        # Load and transform texture
        texture = Image.open(texture_path).convert('RGB')
        if self.transform:
            texture = self.transform(texture)
        
        return {
            'texture': texture,
            'label': label,
            'path': str(texture_path)
        }

class ModelTrainer:
    def __init__(self, texture_generator, device='cpu'):
        self.device = torch.device(device)
        self.texture_generator = texture_generator.to(self.device)
        
        # Loss functions
        self.criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.texture_generator.parameters(), 
            lr=0.0002, 
            betas=(0.5, 0.999)
        )
        
        print(f"Initialized trainer on device: {self.device}")
    
    def train(self, dataset, num_epochs=100, batch_size=32):
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0  # Important for macOS
        )
        
        print(f"Starting training for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                textures = batch['texture'].to(self.device)
                
                # Train step
                self.optimizer.zero_grad()
                generated = self.texture_generator(textures.size(0))
                loss = self.criterion(generated, textures)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f"Epoch [{epoch}/{num_epochs}] "
                          f"Batch [{batch_idx}/{len(dataloader)}] "
                          f"Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, avg_loss)
    
    def save_checkpoint(self, epoch, loss):
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save only the model state dict
        checkpoint_path = checkpoint_dir / f"texture_generator_epoch_{epoch}.pth"
        torch.save(
            self.texture_generator.state_dict(),
            checkpoint_path
        )
        
        # Save metadata separately if needed
        metadata = {
            'epoch': epoch,
            'loss': loss,
        }
        metadata_path = checkpoint_dir / f"texture_generator_epoch_{epoch}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)