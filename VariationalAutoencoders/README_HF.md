---
title: Variational Autoencoder (VAE) - MNIST
emoji: 🎨
colorFrom: blue
colorTo: purple
sdk: pytorch
app_file: Untitled.ipynb
pinned: false
license: mit
tags:
- deep-learning
- generative-ai
- pytorch
- vae
- variational-autoencoder
- mnist
- computer-vision
- unsupervised-learning
- representation-learning
datasets:
- mnist
---

# Variational Autoencoder (VAE) - MNIST Implementation

A comprehensive PyTorch implementation of Variational Autoencoders trained on the MNIST dataset with detailed analysis and visualizations.

## Model Description

This repository contains a complete implementation of a Variational Autoencoder (VAE) trained on the MNIST handwritten digits dataset. The model learns to encode images into a 2-dimensional latent space and decode them back to reconstructed images, enabling both data compression and generation of new digit-like images.

### Architecture Details

- **Model Type**: Variational Autoencoder (VAE)
- **Framework**: PyTorch
- **Input**: 28×28 grayscale images (784 dimensions)
- **Latent Space**: 2 dimensions (for visualization)
- **Hidden Layers**: 256 → 128 (encoder), 128 → 256 (decoder)
- **Total Parameters**: ~400K
- **Model Size**: 1.8MB

### Key Components

1. **Encoder Network**: Maps input images to latent distribution parameters (μ, σ²)
2. **Reparameterization Trick**: Enables differentiable sampling from the latent distribution
3. **Decoder Network**: Reconstructs images from latent space samples
4. **Loss Function**: Combines reconstruction loss (binary cross-entropy) and KL divergence

## Training Details

- **Dataset**: MNIST (60,000 training images, 10,000 test images)
- **Batch Size**: 128
- **Epochs**: 20
- **Optimizer**: Adam
- **Learning Rate**: 1e-3
- **Beta Parameter**: 1.0 (standard VAE)

## Model Performance

### Metrics
- **Final Training Loss**: ~85.2
- **Final Validation Loss**: ~86.1
- **Reconstruction Loss**: ~83.5
- **KL Divergence**: ~1.7

### Capabilities
- ✅ High-quality digit reconstruction
- ✅ Smooth latent space interpolation
- ✅ Generation of new digit-like samples
- ✅ Well-organized latent space with digit clusters

## Usage

### Quick Start

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Load the model (after downloading the files)
class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=2, hidden_dim=256, beta=1.0):
        super(VAE, self).__init__()
        # ... (full implementation in the notebook)
    
    def forward(self, x):
        # ... (full implementation in the notebook)
        pass

# Load trained model
model = VAE()
model.load_state_dict(torch.load('vae_logs_latent2_beta1.0/best_vae_model.pth'))
model.eval()

# Generate new samples
with torch.no_grad():
    # Sample from latent space
    z = torch.randn(16, 2)  # 16 samples, 2D latent space
    generated_images = model.decode(z)
    
    # Reshape and visualize
    generated_images = generated_images.view(-1, 28, 28)
    # Plot the generated images...
```

### Visualizations Available

1. **Latent Space Visualization**: 2D scatter plot showing digit clusters
2. **Reconstructions**: Original vs. reconstructed digit comparisons  
3. **Generated Samples**: New digits sampled from the latent space
4. **Interpolations**: Smooth transitions between different digits
5. **Training Curves**: Loss components over training epochs

## Files and Outputs

- `Untitled.ipynb`: Complete implementation with training and visualization
- `best_vae_model.pth`: Trained model weights
- `training_metrics.csv`: Detailed training metrics
- `generated_samples.png`: Grid of generated digit samples
- `latent_space_visualization.png`: 2D latent space plot
- `reconstruction_comparison.png`: Original vs reconstructed images
- `latent_interpolation.png`: Interpolation between digit pairs
- `comprehensive_training_curves.png`: Training loss curves

## Applications

This VAE implementation can be used for:

- **Generative Modeling**: Create new handwritten digit images
- **Dimensionality Reduction**: Compress images to 2D representations
- **Anomaly Detection**: Identify unusual digits using reconstruction error
- **Data Augmentation**: Generate synthetic training data
- **Representation Learning**: Learn meaningful features for downstream tasks
- **Educational Purposes**: Understand VAE concepts and implementation

## Research and Educational Value

This implementation serves as an excellent educational resource for:

- Understanding Variational Autoencoders theory and practice
- Learning PyTorch implementation techniques
- Exploring generative modeling concepts
- Analyzing latent space representations
- Studying the balance between reconstruction and regularization

## Citation

If you use this implementation in your research or projects, please cite:

```bibtex
@misc{vae_mnist_implementation,
  title={Variational Autoencoder Implementation for MNIST},
  author={Gruhesh Kurra},
  year={2024},
  url={https://huggingface.co/karthik-2905/VariationalAutoencoders}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Additional Resources

- **GitHub Repository**: [VariationalAutoencoders](https://github.com/GruheshKurra/VariationalAutoencoders)
- **Detailed Documentation**: Check `grok.md` for comprehensive VAE explanations
- **Training Logs**: Complete metrics and analysis in the log directories

---

**Tags**: deep-learning, generative-ai, pytorch, vae, mnist, computer-vision, unsupervised-learning

**Model Card Authors**: Gruhesh Kurra 