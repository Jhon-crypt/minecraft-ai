import os
import sys

print("Current working directory:", os.getcwd())
print("Python path:", sys.path)

try:
    from model_generator import MinecraftModelGenerator
    print("Successfully imported MinecraftModelGenerator")
except ImportError as e:
    print(f"Failed to import MinecraftModelGenerator: {e}")

try:
    from texture_generator import MinecraftTextureGenerator
    print("Successfully imported MinecraftTextureGenerator")
except ImportError as e:
    print(f"Failed to import MinecraftTextureGenerator: {e}")

try:
    from training import MinecraftDataset, ModelTrainer
    print("Successfully imported training modules")
except ImportError as e:
    print(f"Failed to import training modules: {e}") 