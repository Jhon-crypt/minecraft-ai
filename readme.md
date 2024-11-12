# Minecraft AI Texture & Model Generator

An advanced AI-powered tool that generates Minecraft textures and models using deep learning. This project uses PyTorch to train and run generative models that can create custom Minecraft-compatible textures and block models from text prompts.

## 🌟 Features

- **Texture Generation**: Create 16x16 Minecraft-compatible textures from text descriptions
- **Model Generation**: Generate JSON models for Minecraft blocks and items
- **Training Pipeline**: Custom training system for fine-tuning the generators
- **Minecraft Compatibility**: All outputs are automatically formatted to work with Minecraft's resource pack system

## 🔧 Installation

1. Clone the repository:

``
git clone https://github.com/yourusername/minecraft-ai-generator.git
cd minecraft-ai-generator
``

2. Install dependencies:

``
pip install torch torchvision pillow numpy

``

## 🚀 Usage

``
python src/minecraft_ai_generator/main.py --generate
``

### Generating Textures

Then follow the prompts to enter your texture descriptions. For example:
- "A weathered oak plank texture"
- "Smooth polished granite"
- "Glowing ancient debris"

### Training the Model

1. Prepare your dataset in the following structure:

dataset/
├── textures/
│ ├── wood_planks.png
│ ├── stone_brick.png
│ └── ...

2. Start training:
``
python src/minecraft_ai_generator/main.py --train --dataset path/to/dataset --epochs 100
``

## 🏗️ Project Structure

- `src/minecraft_ai_generator/`
  - `main.py` - Main entry point and CLI interface
  - `texture_generator.py` - Texture generation model
  - `model_generator.py` - 3D model generation
  - `training.py` - Training pipeline and dataset handling

## 🔍 Technical Details

### Texture Generator
- Uses a GAN-based architecture
- 16x16 RGB output
- Conditional generation based on text prompts
- Built-in Minecraft compatibility checks

### Model Generator
- Transformer-based architecture
- Generates Minecraft-compatible JSON models
- Includes validation for Minecraft's model format
- Supports custom display settings

### Training System
- Custom dataset loader for Minecraft textures
- Automatic labeling based on texture names
- Checkpoint system for model saving
- Progress tracking and logging

## 📊 Model Architecture

### Texture Generator
- Latent dimension: 100
- Multiple deconvolutional layers
- BatchNorm and LeakyReLU activations
- Final Tanh activation for [-1, 1] output range

### Training Parameters
- Batch size: 32
- Learning rate: 0.0002
- Adam optimizer with β1=0.5, β2=0.999
- Checkpoints saved every 10 epochs

## 🔐 Requirements

- Python 3.8+
- PyTorch 1.8+
- PIL (Pillow)
- NumPy
- CUDA-capable GPU (optional, but recommended for training)

## 📝 License

[Your chosen license]

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🐛 Known Issues

- Training requires significant GPU resources
- Generation time can vary based on hardware
- Limited to 16x16 textures currently

