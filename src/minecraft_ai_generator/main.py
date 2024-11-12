import torch
from pathlib import Path
from texture_generator import MinecraftTextureGenerator
from training import MinecraftDataset, ModelTrainer
import argparse
import json

def load_trained_model():
    # Initialize model
    model = MinecraftTextureGenerator()
    
    # Load the latest checkpoint
    checkpoint_dir = Path("checkpoints")
    checkpoints = list(checkpoint_dir.glob("texture_generator_epoch_*.pth"))
    if not checkpoints:
        raise ValueError("No trained model checkpoints found! Please train the model first.")
    
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
    print(f"Loading checkpoint: {latest_checkpoint}")
    
    # Load the state dict with weights_only=True for safety
    checkpoint = torch.load(latest_checkpoint, weights_only=True, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    
    return model

def generate_from_prompt(prompt):
    try:
        print(f"\nGenerating texture from prompt: '{prompt}'")
        
        # Load trained model
        model = load_trained_model()
        
        # Generate texture
        texture = model.generate_texture(prompt)
        
        # Save the generated texture
        output_dir = Path("output/generated_assets")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename from prompt
        filename = prompt.lower().replace(' ', '_').replace('-', '_')
        texture_path = output_dir / f"{filename}.png"
        
        # Save texture
        texture.save(texture_path)
        print(f"Saved texture to: {texture_path}")
        
        return texture_path
        
    except Exception as e:
        print(f"Error generating texture: {str(e)}")
        raise

def train_model(dataset_path, num_epochs=100):
    print("Setting up training...")
    
    # Initialize generator
    texture_gen = MinecraftTextureGenerator()
    
    # Setup dataset
    dataset = MinecraftDataset(dataset_path)
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Setup trainer
    trainer = ModelTrainer(texture_gen)
    
    # Train the model
    trainer.train(dataset, num_epochs=num_epochs)
    
    print("Training completed!")

def main():
    parser = argparse.ArgumentParser(description='Minecraft Texture Generator')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--generate', action='store_true', help='Generate textures')
    parser.add_argument('--dataset', type=str, default='dataset', 
                        help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    
    args = parser.parse_args()
    
    if args.train:
        train_model(args.dataset, args.epochs)
    elif args.generate:
        while True:
            prompt = input("\nEnter your prompt (or 'quit' to exit): ")
            if prompt.lower() == 'quit':
                break
            generate_from_prompt(prompt)
    else:
        print("Please specify --train or --generate")

if __name__ == "__main__":
    main() 