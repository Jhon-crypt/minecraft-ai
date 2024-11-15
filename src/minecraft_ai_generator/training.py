import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
import json
import numpy as np

class MinecraftDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.texture_files = list(self.dataset_path.glob('**/*.png'))
        print(f"Found {len(self.texture_files)} texture files")
        
        # Define transform to ensure 16x16 size
        self.transform = transforms.Compose([
            transforms.Resize((16, 16), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.texture_files)

    def __getitem__(self, idx):
        img_path = self.texture_files[idx]
        image = Image.open(img_path).convert('RGB')
        return self.transform(image)

class ModelTrainer:
    def __init__(self, texture_generator, device='cpu'):
        self.device = device
        self.texture_generator = texture_generator.to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.texture_generator.parameters(), lr=0.0002)
        print(f"Initialized trainer on device: {device}")

    def train(self, dataset, num_epochs=100, batch_size=32):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print(f"Starting training for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            for batch_idx, textures in enumerate(dataloader):
                textures = textures.to(self.device)
                
                self.optimizer.zero_grad()
                
                generated = self.texture_generator(textures.size(0))
                
                loss = self.criterion(generated, textures)
                
                loss.backward()
                self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
                self._save_checkpoint(epoch)

    def _save_checkpoint(self, epoch):
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"texture_generator_epoch_{epoch+1}.pth"
        torch.save(self.texture_generator.state_dict(), checkpoint_path)