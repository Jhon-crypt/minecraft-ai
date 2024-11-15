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
    def __init__(self, latent_dim=100):
        super(MinecraftTextureGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.texture_size = 16
        
        # Neural network layers (matching the saved model)
        self.initial_layer = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256 * 4 * 4)
        )
        
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
        
        # Extended color dictionary
        self.color_names = {
            # Basic colors
            "red": (255, 0, 0),
            "dark_red": (139, 0, 0),
            "green": (0, 255, 0),
            "dark_green": (0, 100, 0),
            "blue": (0, 0, 255),
            "dark_blue": (0, 0, 139),
            "yellow": (255, 255, 0),
            "purple": (128, 0, 128),
            "orange": (255, 165, 0),
            "brown": (165, 42, 42),
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "gray": (128, 128, 128),
            "pink": (255, 192, 203),
            "cyan": (0, 255, 255),
            
            # Minecraft-specific colors
            "oak": (199, 159, 99),
            "spruce": (114, 84, 48),
            "birch": (216, 201, 158),
            "jungle": (150, 111, 51),
            "acacia": (168, 90, 50),
            "dark_oak": (66, 43, 21),
            
            # Stone variants
            "stone": (125, 125, 125),
            "granite": (153, 114, 99),
            "diorite": (225, 225, 225),
            "andesite": (136, 136, 136),
            
            # Metals
            "iron": (216, 216, 216),
            "gold": (249, 236, 79),
            "copper": (180, 113, 77),
            "netherite": (68, 68, 68),
            
            # Gems
            "diamond": (108, 236, 238),
            "emerald": (68, 218, 123),
            "amethyst": (153, 92, 219),
        }
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def forward(self, z):
        x = self.initial_layer(z)
        x = x.view(-1, 256, 4, 4)
        x = self.conv_layers(x)
        return x
    
    def generate_texture(self, prompt):
        """Generate texture based on prompt"""
        self.eval()
        prompt = prompt.lower()
        
        # Get base colors
        colors = self._parse_color_from_prompt(prompt)
        
        # Create texture based on prompt
        if "stone" in prompt:
            img = self._create_stone_texture(colors, "weathered" in prompt)
        elif "marble" in prompt:
            img = self._create_marble_texture(colors)
        elif "wood" in prompt:
            img = self._create_wood_texture(colors)
        else:
            img = self._create_default_texture(colors)
        
        # Apply effects
        if "weathered" in prompt:
            img = self._apply_weathering(img)
        if "polished" in prompt or "smooth" in prompt:
            img = self._apply_polish(img)
        
        return self.ensure_minecraft_compatibility(img)
    
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
