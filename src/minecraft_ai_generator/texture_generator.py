import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import numpy as np
import colorsys
import re
from typing import List, Tuple, Optional
from transformers import pipeline

class MinecraftTextureGenerator(nn.Module):
    def __init__(self, texture_size: int = 16):
        super(MinecraftTextureGenerator, self).__init__()
        self.texture_size = texture_size
        
        # Initialize text understanding pipeline
        self.text_analyzer = pipeline('text-classification', 
                                    model='distilbert-base-uncased',
                                    return_all_scores=True)
        
        # Extended texture attributes
        self.attributes = {
            "material": ["wood", "stone", "metal", "crystal", "dirt", "sand"],
            "pattern": ["smooth", "rough", "striped", "checkered", "spotted"],
            "style": ["natural", "polished", "weathered", "ornate", "simple"],
            "variant": ["light", "dark", "aged", "new", "worn"],
        }
        
        # Expanded color palettes with variations
        self.color_map = {
            # Woods
            "oak": [(199, 159, 99), (188, 152, 98), (167, 133, 79)],
            "dark_oak": [(66, 43, 21), (59, 39, 19), (52, 34, 16)],
            "spruce": [(114, 84, 48), (103, 77, 46), (92, 69, 38)],
            "birch": [(216, 201, 158), (206, 192, 151), (195, 179, 138)],
            "acacia": [(168, 90, 50), (160, 86, 48), (152, 82, 46)],
            
            # Stones
            "stone": [(125, 125, 125), (115, 115, 115), (105, 105, 105)],
            "granite": [(153, 114, 99), (146, 109, 95), (139, 103, 90)],
            "marble": [(255, 250, 250), (245, 240, 240), (235, 230, 230)],
            
            # Metals
            "iron": [(216, 216, 216), (206, 206, 206), (196, 196, 196)],
            "gold": [(249, 236, 79), (242, 229, 75), (235, 222, 71)],
            "copper": [(180, 113, 77), (171, 108, 73), (162, 102, 69)],
            
            # Crystals
            "diamond": [(108, 236, 238), (102, 225, 227), (96, 214, 216)],
            "emerald": [(68, 218, 123), (65, 207, 117), (61, 196, 111)],
            "amethyst": [(153, 92, 219), (146, 88, 208), (138, 83, 197)]
        }
    
    def _analyze_prompt(self, prompt: str) -> dict:
        """Advanced prompt analysis using NLP"""
        # Normalize prompt
        prompt = prompt.lower().strip()
        
        # Extract key information
        analysis = {
            "material": None,
            "color": None,
            "pattern": None,
            "style": None,
            "variant": None,
            "modifiers": []
        }
        
        # Use transformer model to understand intent
        classifications = self.text_analyzer(prompt)
        
        # Extract material
        for material in self.attributes["material"]:
            if material in prompt:
                analysis["material"] = material
                break
        
        # Extract color (including RGB and HEX)
        color = self._parse_color_from_prompt(prompt)
        if color:
            analysis["color"] = color
        
        # Extract pattern
        for pattern in self.attributes["pattern"]:
            if pattern in prompt:
                analysis["pattern"] = pattern
                break
        
        # Extract style
        for style in self.attributes["style"]:
            if style in prompt:
                analysis["style"] = style
                break
        
        # Extract variant
        for variant in self.attributes["variant"]:
            if variant in prompt:
                analysis["variant"] = variant
                break
        
        # Extract modifiers (adjectives)
        words = prompt.split()
        modifiers = ["shiny", "rough", "smooth", "glossy", "matte", "weathered"]
        analysis["modifiers"] = [word for word in words if word in modifiers]
        
        return analysis
    
    def generate_texture(self, prompt: str) -> Image:
        """Generate texture based on analyzed prompt"""
        # Analyze prompt
        attributes = self._analyze_prompt(prompt)
        print(f"Understood attributes: {attributes}")  # Debug info
        
        # Create base image
        base_color = self._get_base_color(attributes)
        img = Image.new('RGB', (self.texture_size, self.texture_size), base_color)
        
        # Apply base pattern
        img = self._apply_base_pattern(img, attributes)
        
        # Apply style modifications
        img = self._apply_style(img, attributes)
        
        # Apply final effects
        img = self._apply_effects(img, attributes)
        
        return self.ensure_minecraft_compatibility(img)
    
    def _get_base_color(self, attributes: dict) -> Tuple[int, int, int]:
        """Get base color from attributes"""
        if attributes["color"]:
            return attributes["color"][0]
        elif attributes["material"] and attributes["material"] in self.color_map:
            return self.color_map[attributes["material"]][0]
        return self.color_map["oak"][0]  # default
    
    def _apply_base_pattern(self, img: Image, attributes: dict) -> Image:
        """Apply the base pattern based on attributes"""
        draw = ImageDraw.Draw(img)
        
        if attributes["material"] == "wood":
            return self._generate_wood_pattern(img, attributes)
        elif attributes["material"] == "stone":
            return self._generate_stone_pattern(img, attributes)
        elif attributes["material"] == "metal":
            return self._generate_metal_pattern(img, attributes)
        elif attributes["material"] == "crystal":
            return self._generate_crystal_pattern(img, attributes)
        
        return img
    
    def _apply_style(self, img: Image, attributes: dict) -> Image:
        """Apply style modifications"""
        if "weathered" in attributes["modifiers"]:
            img = self._add_weathering(img)
        if "shiny" in attributes["modifiers"]:
            img = self._add_shine(img)
        if "rough" in attributes["modifiers"]:
            img = self._add_roughness(img)
        
        return img
    
    def _apply_effects(self, img: Image, attributes: dict) -> Image:
        """Apply final effects and adjustments"""
        # Enhance contrast based on material
        contrast = 1.0
        if attributes["material"] in ["metal", "crystal"]:
            contrast = 1.2
        elif attributes["material"] in ["stone", "dirt"]:
            contrast = 0.9
            
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)
        
        return img
    
    def _parse_color_from_prompt(self, prompt: str) -> Optional[List[Tuple[int, int, int]]]:
        """Parse color information from prompt with high accuracy"""
        prompt = prompt.lower()
        
        # Check for exact color names
        for color_name, colors in self.color_map.items():
            if color_name in prompt:
                return colors
        
        # Parse RGB values with validation
        rgb_match = re.search(r'rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', prompt)
        if rgb_match:
            try:
                r, g, b = map(int, rgb_match.groups())
                if all(0 <= c <= 255 for c in (r, g, b)):
                    return self._generate_color_variations((r, g, b))
            except ValueError:
                pass
        
        # Parse hex colors
        hex_match = re.search(r'#([0-9a-fA-F]{6})', prompt)
        if hex_match:
            try:
                hex_color = hex_match.group(1)
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                return self._generate_color_variations((r, g, b))
            except ValueError:
                pass
        
        return None
    
    def _generate_color_variations(self, base_color: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Generate harmonious color variations"""
        r, g, b = base_color
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        
        variations = []
        # Base color
        variations.append(base_color)
        # Slightly darker
        variations.append(tuple(int(c * 0.95) for c in base_color))
        # Even darker
        variations.append(tuple(int(c * 0.90) for c in base_color))
        
        return variations
    
    def _generate_planks_pattern(self, img: Image, colors: List[Tuple[int, int, int]]) -> Image:
        """Generate realistic wood plank pattern"""
        draw = ImageDraw.Draw(img)
        plank_height = self.texture_size // 4
        
        for i in range(4):
            y = i * plank_height
            # Main plank color
            draw.rectangle([0, y, self.texture_size-1, y+plank_height-1], fill=colors[0])
            
            # Wood grain
            for grain in range(3):
                y_grain = y + grain + np.random.randint(0, plank_height-2)
                draw.line([(0, y_grain), (self.texture_size-1, y_grain)], 
                         fill=colors[1], width=1)
            
            # Plank separation
            draw.line([(0, y+plank_height-1), (self.texture_size-1, y+plank_height-1)],
                     fill=colors[2], width=1)
        
        return img
    
    def _generate_stone_pattern(self, img: Image, colors: List[Tuple[int, int, int]]) -> Image:
        """Generate realistic stone texture"""
        draw = ImageDraw.Draw(img)
        
        # Base texture
        for x in range(self.texture_size):
            for y in range(self.texture_size):
                if np.random.random() < 0.3:
                    size = np.random.randint(1, 3)
                    color_idx = np.random.randint(0, len(colors))
                    draw.rectangle([x, y, x+size, y+size], fill=colors[color_idx])
        
        # Apply noise and cracks
        for _ in range(5):
            start_x = np.random.randint(0, self.texture_size)
            start_y = np.random.randint(0, self.texture_size)
            end_x = start_x + np.random.randint(-4, 4)
            end_y = start_y + np.random.randint(-4, 4)
            draw.line([(start_x, start_y), (end_x, end_y)], fill=colors[2], width=1)
        
        return img
    
    def ensure_minecraft_compatibility(self, texture: Image) -> Image:
        """Ensure texture meets Minecraft requirements"""
        if texture.size != (self.texture_size, self.texture_size):
            texture = texture.resize((self.texture_size, self.texture_size), Image.NEAREST)
        
        if texture.mode != 'RGB':
            texture = texture.convert('RGB')
        
        return texture