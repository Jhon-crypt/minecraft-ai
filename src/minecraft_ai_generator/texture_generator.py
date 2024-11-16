import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from typing import Tuple
import colorsys


class MinecraftTextureGenerator(nn.Module):
    def __init__(self, latent_dim=100):
        super(MinecraftTextureGenerator, self).__init__()
        self.latent_dim = latent_dim
        
        # Modified architecture to work with batch size 1
        self.initial_layer = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.LeakyReLU(0.2)
        )
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.InstanceNorm2d(128),  # Using InstanceNorm instead of BatchNorm
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.InstanceNorm2d(64),   # Using InstanceNorm instead of BatchNorm
                nn.LeakyReLU(0.2)
            )
        ])
        
        self.final_layer = nn.Sequential(
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh()
        )

        # Color handling
        self.color_map = {
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "purple": (128, 0, 128),
            "orange": (255, 165, 0),
            "pink": (255, 192, 203),
            "brown": (165, 42, 42),
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "gray": (128, 128, 128),
            "cyan": (0, 255, 255),
            "magenta": (255, 0, 255),
            "lime": (50, 255, 50),
            "maroon": (128, 0, 0),
            "navy": (0, 0, 128),
            "olive": (128, 128, 0),
            "teal": (0, 128, 128)
        }

    def forward(self, z):
        # Generate base texture
        x = self.initial_layer(z)
        x = x.view(-1, 256, 4, 4)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.final_layer(x)
        return x

    def generate_untrained_texture(self, prompt: str) -> Image.Image:
        """Generate a basic texture without training"""
        # Create a simple colored noise pattern
        size = (16, 16)
        array = np.random.rand(size[0], size[1], 3) * 255
        
        # Find requested color
        color = None
        prompt = prompt.lower()
        for color_name, color_value in self.color_map.items():
            if color_name in prompt:
                color = color_value
                break
        
        if color is not None:
            # Apply color tint
            color_array = np.array(color) / 255.0
            array = array * 0.3 + np.array(color) * 0.7
            
        # Ensure values are in valid range
        array = np.clip(array, 0, 255).astype(np.uint8)
        
        # Create image
        image = Image.fromarray(array.astype('uint8'))
        
        # Add some texture
        image = self._add_noise(image, 0.2)
        
        return image

    def _tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        """Convert a tensor to a PIL Image"""
        # Denormalize from [-1, 1] to [0, 1]
        tensor = (tensor + 1) / 2.0
        # Clamp values to valid range
        tensor = torch.clamp(tensor, 0, 1)
        # Convert to PIL Image
        array = (tensor.cpu().numpy() * 255).astype('uint8')
        array = array.transpose(1, 2, 0)  # CHW to HWC
        return Image.fromarray(array)

    def _add_noise(self, image: Image.Image, amount: float = 0.1) -> Image.Image:
        """Add noise to the texture"""
        array = np.array(image)
        noise = np.random.normal(0, amount * 255, array.shape)
        noisy_array = np.clip(array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_array)

    # ... (rest of the methods remain the same) ...