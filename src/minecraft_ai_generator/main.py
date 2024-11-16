import argparse
import torch
from pathlib import Path
from training import MinecraftDataset, TextureTrainer
from texture_generator import MinecraftTextureGenerator

def load_trained_model():
    """Load the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MinecraftTextureGenerator().to(device)
    
    try:
        checkpoint_path = 'checkpoints/texture_generator_epoch_200.pth'
        if not Path(checkpoint_path).exists():
            print(f"Warning: Checkpoint file not found at {checkpoint_path}")
            print("Using untrained model...")
            return model
            
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        model.eval()
        print(f"Model loaded from {checkpoint_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using untrained model...")
    
    return model

def train_model(dataset_path: str, num_epochs: int):
    """Train the model"""
    print("Setting up training...")
    
    # Load dataset
    dataset = MinecraftDataset(dataset_path)
    print(f"Found {len(dataset)} texture files")
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Initialize trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = TextureTrainer(device)
    print(f"Initialized trainer on device: {device}")
    
    # Start training
    print(f"Starting training for {num_epochs} epochs...")
    trainer.train(dataset, num_epochs=num_epochs)

def generate_from_prompt(prompt: str):
    """Generate texture from prompt"""
    print(f"Generating texture from prompt: '{prompt}'")
    try:
        model = load_trained_model()
        
        # If no checkpoint found, use untrained generation
        if not Path('checkpoints/texture_generator_epoch_200.pth').exists():
            texture = model.generate_untrained_texture(prompt)
        else:
            # Generate using trained model
            device = next(model.parameters()).device
            with torch.no_grad():
                z = torch.randn(1, model.latent_dim, device=device)
                generated = model(z)
                texture = model._tensor_to_image(generated[0])
                texture = model._apply_color(texture, prompt)
        
        # Save the generated texture
        output_dir = Path('generated')
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"texture_{len(list(output_dir.glob('*.png')))}.png"
        texture.save(output_path)
        print(f"Saved generated texture to: {output_path}")
        
    except Exception as e:
        print(f"Error generating texture: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Minecraft AI Texture Generator')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--generate', action='store_true', help='Generate textures')
    parser.add_argument('--dataset', type=str, help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    
    args = parser.parse_args()
    
    if args.train:
        if not args.dataset:
            print("Error: --dataset path is required for training")
            return
        train_model(args.dataset, args.epochs)
    elif args.generate:
        while True:
            prompt = input("\nEnter your prompt (or 'quit' to exit): ")
            if prompt.lower() == 'quit':
                break
            generate_from_prompt(prompt)
    else:
        print("Please specify either --train or --generate")

if __name__ == '__main__':
    main() 