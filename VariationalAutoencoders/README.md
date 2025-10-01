# Variational Autoencoders (VAE) Implementation

A comprehensive implementation of Variational Autoencoders using PyTorch, featuring detailed training, visualization, and analysis capabilities.

## 🎯 Overview

This repository contains a complete implementation of Variational Autoencoders (VAEs) trained on the MNIST dataset. The implementation includes:

- **Complete VAE Architecture**: Encoder, decoder, and reparameterization trick
- **Comprehensive Training System**: With logging, model checkpointing, and metrics tracking
- **Rich Visualizations**: Latent space plots, reconstructions, interpolations, and generated samples
- **Detailed Documentation**: Both in code and this README

## 🏗️ Architecture

The VAE implementation features:

- **Input Dimension**: 784 (28×28 MNIST images)
- **Latent Dimension**: 2 (for easy visualization)
- **Hidden Dimensions**: 256 → 128 in encoder, 128 → 256 in decoder
- **Regularization**: Batch normalization and dropout layers
- **Loss Function**: Binary cross-entropy reconstruction loss + KL divergence

### Model Components

1. **Encoder**: Maps input images to latent distribution parameters (μ, σ²)
2. **Reparameterization**: Samples latent variables using the reparameterization trick
3. **Decoder**: Reconstructs images from latent variables

## 📊 Results

The trained model achieves:

- **Reconstruction Quality**: Clear digit reconstructions
- **Latent Space Organization**: Well-separated digit clusters in 2D latent space
- **Generation Capability**: Smooth interpolations between different digits
- **Training Stability**: Balanced reconstruction and KL losses

### Generated Samples

![Generated Samples](vae_logs_latent2_beta1.0/generated_samples.png)

### Latent Space Visualization

![Latent Space](vae_logs_latent2_beta1.0/latent_space_visualization.png)

### Reconstruction Comparison

![Reconstructions](vae_logs_latent2_beta1.0/reconstruction_comparison.png)

### Training Curves

![Training Metrics](vae_logs_latent2_beta1.0/comprehensive_training_curves.png)

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch torchvision matplotlib pandas numpy seaborn
```

### Running the Code

1. Clone this repository:
```bash
git clone https://github.com/GruheshKurra/VariationalAutoencoders.git
cd VariationalAutoencoders
```

2. Open and run the Jupyter notebook:
```bash
jupyter notebook Untitled.ipynb
```

The notebook will automatically:
- Download and preprocess the MNIST dataset
- Train the VAE model
- Generate visualizations and save results
- Create comprehensive training logs

## 📁 Repository Structure

```
├── Untitled.ipynb                 # Main implementation notebook
├── vae_logs_latent2_beta1.0/      # Training results and visualizations
│   ├── best_vae_model.pth         # Trained model weights
│   ├── training_metrics.csv       # Training metrics
│   ├── generated_samples.png      # Generated digit samples
│   ├── latent_space_visualization.png  # 2D latent space plot
│   ├── reconstruction_comparison.png   # Original vs reconstructed
│   ├── latent_interpolation.png   # Interpolation between digits
│   └── comprehensive_training_curves.png  # Training curves
├── pytorch_vae_logs/              # Additional training logs
├── data/                          # MNIST dataset (auto-downloaded)
├── grok.md                        # Detailed VAE explanation
└── README.md                      # This file
```

## 🧠 Key Features

### 1. Comprehensive VAE Implementation
- Clean, well-documented PyTorch implementation
- Modular design with separate classes for model, trainer, logger, and visualizer
- Support for different latent dimensions and beta values

### 2. Advanced Training System
- Automatic model checkpointing
- Comprehensive metrics logging
- Learning rate scheduling
- Early stopping capabilities

### 3. Rich Visualizations
- **Latent Space Plots**: Visualize learned representations
- **Reconstructions**: Compare original and reconstructed images
- **Generated Samples**: New digits sampled from the latent space
- **Interpolations**: Smooth transitions between different digits
- **Training Curves**: Monitor loss components over time

### 4. Detailed Analysis
- Training metrics tracking (reconstruction loss, KL loss, total loss)
- Model performance evaluation
- Latent space analysis and interpretation

## 🔧 Customization

The implementation supports easy customization:

```python
# Different latent dimensions
latent_dim = 10  # Higher dimensional latent space

# Different beta values (β-VAE)
beta = 0.5  # Lower beta for better reconstructions
beta = 4.0  # Higher beta for better disentanglement

# Different architectures
hidden_dim = 512  # Larger hidden layers
```

## 📚 Educational Value

This repository is designed for learning and includes:

- **Detailed Comments**: Every line of code is explained
- **Mathematical Background**: Complete loss function derivations
- **Visualization Examples**: Understanding what VAEs learn
- **Training Analysis**: How to monitor and improve VAE training

## 🎯 Use Cases

This VAE implementation can be adapted for:

- **Image Generation**: Generate new images similar to training data
- **Data Compression**: Efficient representation learning
- **Anomaly Detection**: Identify outliers using reconstruction error
- **Data Augmentation**: Generate synthetic training samples
- **Representation Learning**: Learn meaningful features

## 📈 Performance

Training on MNIST (60,000 images):
- **Training Time**: ~10 minutes on CPU, ~2 minutes on GPU
- **Model Size**: ~1.8MB
- **Parameters**: ~400K trainable parameters

## 🤝 Contributing

Feel free to contribute by:
- Adding new datasets
- Implementing different VAE variants (β-VAE, WAE, etc.)
- Improving visualizations
- Adding more comprehensive documentation

## 📄 License

This project is open source and available under the MIT License.

## 📬 Contact

For questions or suggestions, please open an issue in this repository.

---

**Happy Learning! 🎓**

*This implementation provides a solid foundation for understanding and experimenting with Variational Autoencoders.*
