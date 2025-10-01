# 22AIE304 Deep Learning - Team A15
## Image Generation and Representation Learning with Variational Autoencoders

---

## Abstract

The project explores Variational Autoencoders (VAEs)—a class of generative models that combine deep learning and probabilistic reasoning. The aim is to build and train a VAE on the MNIST dataset to learn a continuous, probabilistic latent space for handwritten digits. By leveraging the principles of approximate variational inference, the model encodes input images into distributions and decodes samples to reconstruct realistic digits. This process allows not only the generation of novel synthetic digits but also the interpretation of how digit features are organized in the latent space. The project provides a practical implementation of uncertainty representation, inference, and generative modelling using custom neural network implementations rather than library-based layers.

---

## Literature Review

### Foundational Work on Variational Autoencoders

| Title | Authors | Year | Dataset | Methodology | Evaluation Score | Remarks |
|-------|---------|------|---------|-------------|-----------------|----------|
| Auto-Encoding Variational Bayes | Kingma, D.P. & Welling, M. | 2013 | MNIST, Frey Faces | **Variational Inference**, **Reparameterization Trick**, **ELBO Optimization** | Log-likelihood: -85.51 (MNIST) | Introduced the foundational VAE framework |
| Stochastic Backpropagation and Approximate Inference | Rezende, D.J. et al. | 2014 | MNIST, Caltech-101 | **Stochastic Gradient Variational Bayes**, **Neural Variational Inference** | Log-likelihood: -84.78 (MNIST) | Alternative derivation of VAE with emphasis on stochastic optimization |
| β-VAE: Learning Basic Visual Concepts | Higgins, I. et al. | 2017 | dSprites, 3D Chairs | **Disentangled Representation Learning**, **β-VAE Framework** | MIG Score: 0.58, SAP Score: 0.13 | Introduced β parameter for controllable disentanglement |
| Understanding disentangling in β-VAE | Burgess, C.P. et al. | 2018 | dSprites, 3D Shapes | **Controlled Capacity Increase**, **β-VAE Analysis** | Disentanglement: 0.82, Completeness: 0.75 | Theoretical analysis of β-VAE disentanglement properties |
| Conditional Variational Autoencoder | Sohn, K. et al. | 2015 | MNIST, CelebA | **Conditional Generation**, **Class-conditioned VAE** | MNIST Accuracy: 96.38%, CelebA IS: 3.51 | Extended VAE for conditional generation |
| WAE: Wasserstein Auto-Encoders | Tolstikhin, I. et al. | 2018 | MNIST, CelebA | **Wasserstein Distance**, **Optimal Transport** | FID: 7.7 (CelebA), IS: 3.1 | Alternative to KL divergence using Wasserstein distance |
| InfoVAE: Balancing Learning and Inference | Zhao, S. et al. | 2019 | MNIST, CIFAR-10 | **Mutual Information Maximization**, **Information-theoretic VAE** | FID: 24.4 (CIFAR-10), IS: 6.7 | Addresses posterior collapse in VAEs |
| Neural Discrete Representation Learning | van den Oord, A. et al. | 2017 | CIFAR-10, ImageNet | **Vector Quantization**, **VQ-VAE** | Reconstruction Loss: 0.25, Codebook Usage: 83% | Discrete latent representations in VAEs |

### Generative Models Comparison

| Title | Authors | Year | Dataset | Methodology | Evaluation Score | Remarks |
|-------|---------|------|---------|-------------|-----------------|----------|
| Generative Adversarial Networks | Goodfellow, I. et al. | 2014 | MNIST, CIFAR-10 | **Adversarial Training**, **Min-Max Game** | IS: 7.8 (CIFAR-10) | Introduced GAN framework for comparison |
| Deep Convolutional GANs | Radford, A. et al. | 2016 | LSUN, CelebA | **Convolutional Architecture**, **Stable GAN Training** | IS: 8.1, FID: 7.8 | Architectural guidelines for stable GAN training |
| Progressive Growing of GANs | Karras, T. et al. | 2018 | CelebA-HQ | **Progressive Training**, **High-resolution Generation** | FID: 4.2, IS: 8.8 | High-quality image generation methodology |

---

## Understanding of Problem Statement

### Core Problem
Traditional autoencoders learn deterministic mappings from input to latent space, limiting their generative capabilities and inability to capture uncertainty. The challenge is to develop a generative model that:

1. **Learns meaningful representations** of high-dimensional data (images)
2. **Captures uncertainty** in the learned representations
3. **Enables controlled generation** of new samples
4. **Provides interpretable latent space** for analysis and manipulation

### Technical Challenges
- **Posterior Intractability**: True posterior p(z|x) is intractable for complex models
- **Non-differentiable Sampling**: Standard sampling operations break gradient flow
- **Mode Collapse**: Risk of learning limited diversity in generated samples
- **Reconstruction-Generation Trade-off**: Balancing reconstruction quality with generative diversity

### Deep Learning Perspective
From a deep learning standpoint, VAEs address:
- **Representation Learning**: Learning compressed, meaningful features
- **Generative Modeling**: Creating new data samples from learned distributions
- **Regularization**: Structured latent space through probabilistic constraints
- **Scalability**: Handling high-dimensional data through neural network parameterization

---

## Approach and Problem Mapping

### Architecture Design

#### Custom Implementation Philosophy
Rather than using pre-built `nn.Module` layers, we implement custom components to demonstrate deep understanding:

```python
# Custom Convolutional Layer Implementation
class CustomConv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        # Custom weight initialization and forward pass
        
# Custom Transpose Convolutional Layer
class CustomTransposeConv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        # Custom deconvolution implementation
```

### Problem Mapping Strategy

#### 1. Encoder Network Design
- **Input**: 28×28 grayscale images (784 dimensions)
- **Architecture**: Custom CNN with multiple convolutional layers
- **Output**: Latent distribution parameters (μ, log σ²)
- **Custom Implementation**: 
  - Hand-crafted convolution operations
  - Custom activation functions
  - Batch normalization from scratch

#### 2. Latent Space Modeling
- **Probabilistic Representation**: z ~ N(μ, σ²)
- **Reparameterization Trick**: z = μ + σ ⊙ ε, where ε ~ N(0,I)
- **Custom Sampling**: Implement sampling without library functions

#### 3. Decoder Network Design
- **Input**: Sampled latent vectors (20 dimensions)
- **Architecture**: Custom transpose CNN
- **Output**: Reconstructed 28×28 images
- **Custom Implementation**:
  - Hand-crafted deconvolution operations
  - Custom upsampling techniques

#### 4. Loss Function Engineering
- **Reconstruction Loss**: Custom binary cross-entropy implementation
- **KL Divergence**: Custom KL divergence calculation
- **Combined Loss**: L = L_recon + β × L_KL

### Training Strategy
1. **Custom Optimizer**: Implement Adam optimizer from scratch
2. **Gradient Computation**: Manual gradient calculation and backpropagation
3. **Batch Processing**: Custom data loading and batching mechanisms
4. **Learning Rate Scheduling**: Custom learning rate decay strategies

---

## Proposed Visualizations

### 1. Architecture Visualizations
- **Network Architecture Diagram**: Custom-drawn VAE architecture showing encoder-decoder structure
- **Layer-wise Feature Maps**: Visualization of intermediate conv layer outputs
- **Weight Visualizations**: Learned convolutional filters and their evolution during training
- **Gradient Flow Diagrams**: Visualization of gradient magnitudes through custom layers

### 2. Training Dynamics
- **Loss Curves**: Real-time plotting of reconstruction loss, KL divergence, and total loss
- **Learning Rate Schedule**: Visualization of custom learning rate decay
- **Gradient Magnitude Tracking**: Monitor gradient health through custom backprop
- **Parameter Evolution**: Tracking weight changes in custom layers over epochs

### 3. Latent Space Analysis
- **2D Latent Space Visualization**: PCA/t-SNE projection of learned representations
- **Latent Interpolation**: Smooth transitions between digit classes in latent space
- **Latent Arithmetic**: Visualization of vector arithmetic in latent space
- **Uncertainty Visualization**: Heatmaps showing variance in latent representations

### 4. Generation Quality Assessment
- **Reconstruction Comparison**: Side-by-side original vs reconstructed images
- **Generated Samples Grid**: Random samples from prior distribution
- **Progressive Generation**: Samples at different training epochs
- **Quality Metrics Visualization**: FID and IS scores over training time

### 5. Custom Implementation Insights
- **Custom vs Library Comparison**: Performance comparison of custom implementations
- **Memory Usage Analysis**: Tracking memory consumption of custom operations
- **Computational Efficiency**: Runtime analysis of custom vs library functions
- **Implementation Correctness**: Gradient checking visualizations

### 6. Probabilistic Analysis
- **Posterior Distribution Visualization**: Distribution of learned μ and σ parameters
- **Prior-Posterior Alignment**: KL divergence visualization between learned and prior distributions
- **Uncertainty Quantification**: Confidence intervals and prediction uncertainty
- **Sampling Diversity**: Analysis of sample diversity from the generative model

### 7. Ablation Studies
- **β Parameter Effects**: Visualization of different β values in β-VAE
- **Latent Dimension Analysis**: Effect of latent space dimensionality
- **Architecture Variations**: Comparison of different custom encoder/decoder designs
- **Training Strategy Impact**: Effect of different custom optimization approaches

### 8. Interactive Visualizations
- **Latent Space Explorer**: Interactive tool to navigate and sample from latent space
- **Real-time Generation**: Live generation during training process
- **Parameter Sensitivity**: Interactive sliders to adjust model parameters
- **Custom Layer Activations**: Interactive exploration of custom layer outputs

---

## Innovation and Future Scope

### Immediate Innovations
1. **Custom Layer Implementation**: Full implementation without deep learning libraries
2. **Probabilistic Uncertainty Quantification**: Advanced uncertainty measures
3. **Architectural Experiments**: Novel encoder-decoder designs
4. **Training Dynamics Analysis**: Deep dive into custom optimization behavior

### Extended Scope
1. **Conditional VAE (C-VAE)**: Class-conditioned generation with custom implementation
2. **β-VAE Analysis**: Disentangled representation learning with custom β scheduling
3. **Anomaly Detection**: Reconstruction-based outlier detection system
4. **Multi-modal VAE**: Extension to handle different data modalities

### Technical Contributions
- **Educational Value**: Complete understanding through custom implementation
- **Performance Analysis**: Detailed comparison of custom vs library implementations
- **Algorithmic Insights**: Deep understanding of VAE mathematics and implementation
- **Scalability Study**: Analysis of custom implementation scalability

---

## Methodology Summary

This project demonstrates a comprehensive understanding of VAEs through:
1. **Custom Implementation**: Building every component from scratch
2. **Theoretical Foundation**: Strong mathematical understanding of variational inference
3. **Practical Application**: Real-world implementation on MNIST dataset
4. **Extensive Visualization**: Comprehensive analysis through multiple visualization techniques
5. **Performance Evaluation**: Rigorous quantitative and qualitative assessment

The custom implementation approach ensures deep understanding of the underlying algorithms while providing valuable insights into the mechanics of variational autoencoders and generative modeling in deep learning.
