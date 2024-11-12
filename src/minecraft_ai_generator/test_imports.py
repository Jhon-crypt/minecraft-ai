import os
import sys

print("Current working directory:", os.getcwd())
print("Python path:", sys.path)

try:
    from model_generator import MinecraftModelGenerator
    print("Successfully imported MinecraftModelGenerator")
except ImportError as e:
    print(f"Failed to import MinecraftModelGenerator: {e}") 