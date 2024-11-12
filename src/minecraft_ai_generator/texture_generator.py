import torch
import torch.nn as nn
from PIL import Image
import numpy as np

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)

class MinecraftTextureGenerator(nn.Module):
    def __init__(self, latent_dim=100):
        super(MinecraftTextureGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.texture_size = 16
        
        # Split the generator into separate components
        self.initial_layer = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256 * 4 * 4)
        )
        
        self.reshape_layer = Reshape((-1, 256, 4, 4))
        
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh()
        )
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, batch_size):
        # Generate random latent vectors
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        
        # Pass through layers
        x = self.initial_layer(z)
        x = self.reshape_layer(x)
        x = self.conv_layers(x)
        
        return x
    
    def generate_texture(self, prompt, model_json=None):
        """Generate a texture based on the prompt"""
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            # Generate single texture
            z = torch.randn(1, self.latent_dim, device=self.device)
            
            # Pass through generator layers
            x = self.initial_layer(z)
            x = self.reshape_layer(x)
            texture = self.conv_layers(x)
            
            # Convert to image
            texture = self._to_image(texture[0])
            texture = self.ensure_minecraft_compatibility(texture)
            
            return texture
    
    def _to_image(self, tensor):
        """Convert tensor to PIL Image"""
        # Denormalize
        tensor = (tensor + 1) / 2
        tensor = tensor.clamp(0, 1)
        
        # Convert to numpy array
        array = tensor.cpu().numpy()
        array = (array * 255).astype(np.uint8)
        array = np.transpose(array, (1, 2, 0))
        
        return Image.fromarray(array)
    
    def ensure_minecraft_compatibility(self, texture):
        """Ensure texture meets Minecraft requirements"""
        if texture.size != (self.texture_size, self.texture_size):
            texture = texture.resize((self.texture_size, self.texture_size), Image.NEAREST)
        
        if texture.mode != 'RGB':
            texture = texture.convert('RGB')
        
        return texture