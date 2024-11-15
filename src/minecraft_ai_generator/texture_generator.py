import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import numpy as np
import colorsys
import webcolors
import re
from typing import List, Tuple
from transformers import pipeline


class MinecraftTextureGenerator(nn.Module):
    def __init__(self, latent_dim=100, initial_size=4, device='cpu'):
        super().__init__()
        self.latent_dim = latent_dim
        self.initial_size = initial_size
        self.device = device

        # Calculate initial dense layer size
        initial_channels = 256
        dense_size = initial_channels * initial_size * initial_size
        
        self.initial_layer = nn.Sequential(
            nn.Linear(latent_dim, dense_size),
            nn.BatchNorm1d(dense_size),
            nn.ReLU()
        )

        # Two upsampling blocks to go from 4x4 to 16x16
        self.blocks = nn.ModuleList([
            self._make_block(256, 128),  # 4x4 -> 8x8
            self._make_block(128, 64)    # 8x8 -> 16x16
        ])

        self.final_layer = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def _make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, batch_size):
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        x = self.initial_layer(z)
        x = x.view(batch_size, 256, self.initial_size, self.initial_size)
        
        for block in self.blocks:
            x = block(x)
            
        return self.final_layer(x)

    def generate_texture(self, prompt):
        self.eval()
        with torch.no_grad():
            # Generate a single texture
            z = torch.randn(1, self.latent_dim, device=self.device)
            x = self.initial_layer(z)
            x = x.view(1, 256, self.initial_size, self.initial_size)
            
            for block in self.blocks:
                x = block(x)
                
            x = self.final_layer(x)
            
            # Convert to image
            x = (x + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
            x = x.squeeze(0).permute(1, 2, 0)
            x = (x * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)
            return Image.fromarray(x)
    
    def _create_stone_texture(self, colors, weathered=False):
        """Create stone texture"""
        img = Image.new('RGB', (self.texture_size, self.texture_size), colors[0])
        draw = ImageDraw.Draw(img)
        
        # Base noise
        for x in range(self.texture_size):
            for y in range(self.texture_size):
                if np.random.random() < 0.3:
                    size = np.random.randint(1, 3)
                    color = colors[np.random.randint(0, len(colors))]
                    draw.rectangle([x, y, x+size, y+size], fill=color)
        
        # Add cracks
        for _ in range(5):
            x1 = np.random.randint(0, self.texture_size)
            y1 = np.random.randint(0, self.texture_size)
            x2 = x1 + np.random.randint(-3, 4)
            y2 = y1 + np.random.randint(-3, 4)
            draw.line([(x1, y1), (x2, y2)], fill=colors[2], width=1)
        
        return img
    
    def _create_marble_texture(self, colors):
        """Create marble texture"""
        img = Image.new('RGB', (self.texture_size, self.texture_size), colors[0])
        draw = ImageDraw.Draw(img)
        
        # Create veins
        for _ in range(3):
            x = np.random.randint(0, self.texture_size)
            y = np.random.randint(0, self.texture_size)
            for i in range(4):
                x2 = x + np.random.randint(-3, 4)
                y2 = y + np.random.randint(-3, 4)
                draw.line([(x, y), (x2, y2)], fill=colors[1], width=1)
                x, y = x2, y2
        
        return img
    
    def _create_wood_texture(self, colors):
        """Create wood texture"""
        img = Image.new('RGB', (self.texture_size, self.texture_size), colors[0])
        draw = ImageDraw.Draw(img)
        
        # Create wood grain
        for y in range(0, self.texture_size, 2):
            color = colors[1] if y % 4 == 0 else colors[2]
            draw.line([(0, y), (self.texture_size-1, y)], fill=color)
        
        return img
    
    def _create_default_texture(self, colors):
        """Create default texture"""
        img = Image.new('RGB', (self.texture_size, self.texture_size), colors[0])
        draw = ImageDraw.Draw(img)
        
        for _ in range(20):
            x = np.random.randint(0, self.texture_size)
            y = np.random.randint(0, self.texture_size)
            size = np.random.randint(1, 4)
            color = colors[np.random.randint(0, len(colors))]
            draw.rectangle([x, y, x+size, y+size], fill=color)
        
        return img
    
    def _apply_weathering(self, img):
        """Apply weathering effect"""
        draw = ImageDraw.Draw(img)
        for _ in range(30):
            x = np.random.randint(0, self.texture_size)
            y = np.random.randint(0, self.texture_size)
            size = np.random.randint(1, 3)
            color = tuple(max(0, c - 20) for c in img.getpixel((x, y)))
            draw.ellipse([x, y, x+size, y+size], fill=color)
        
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(0.9)
    
    def _apply_polish(self, img):
        """Apply polished effect"""
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.1)
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(1.2)
    
    def _parse_color_from_prompt(self, prompt: str) -> List[Tuple[int, int, int]]:
        """Enhanced color parsing from prompt"""
        prompt = prompt.lower()
        
        # Try to find RGB values
        rgb_match = re.search(r'rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', prompt)
        if rgb_match:
            r, g, b = map(int, rgb_match.groups())
            return self._generate_color_variations((r, g, b))
        
        # Try to find hex color
        hex_match = re.search(r'#([0-9a-fA-F]{6})', prompt)
        if hex_match:
            hex_color = hex_match.group(1)
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return self._generate_color_variations((r, g, b))
        
        # Look for color names
        for color_name, color_value in self.color_names.items():
            if color_name in prompt:
                return self._generate_color_variations(color_value)
            
        # Try to parse any CSS color name
        try:
            for word in prompt.split():
                try:
                    color = webcolors.name_to_rgb(word)
                    return self._generate_color_variations((color.red, color.green, color.blue))
                except ValueError:
                    continue
        except:
            pass
        
        # Default color based on material
        if "stone" in prompt:
            return self._generate_color_variations(self.color_names["stone"])
        elif "wood" in prompt:
            return self._generate_color_variations(self.color_names["oak"])
        
        # Fallback to gray if no color is found
        return self._generate_color_variations(self.color_names["gray"])
    
    def _generate_color_variations(self, base_color: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Generate variations of a base color"""
        # Convert to HSV for better color manipulation
        h, s, v = colorsys.rgb_to_hsv(base_color[0]/255, base_color[1]/255, base_color[2]/255)
        
        variations = []
        # Base color
        variations.append(base_color)
        
        # Slightly darker
        v_darker = max(0, v * 0.85)
        r, g, b = colorsys.hsv_to_rgb(h, s, v_darker)
        variations.append((int(r*255), int(g*255), int(b*255)))
        
        # Even darker
        v_darkest = max(0, v * 0.7)
        r, g, b = colorsys.hsv_to_rgb(h, s, v_darkest)
        variations.append((int(r*255), int(g*255), int(b*255)))
        
        return variations
    
    def ensure_minecraft_compatibility(self, texture):
        """Ensure texture meets Minecraft requirements"""
        if texture.size != (self.texture_size, self.texture_size):
            texture = texture.resize((self.texture_size, self.texture_size), Image.NEAREST)
        if texture.mode != 'RGB':
            texture = texture.convert('RGB')
        return texture