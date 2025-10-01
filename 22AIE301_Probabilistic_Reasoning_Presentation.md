# 22AIE301 Probabilistic Reasoning - Team A15
## Image Generation and Representation Learning with Variational Autoencoders

---

## Group Details
- **Course**: 22AIE301 Probabilistic Reasoning
- **Team**: A15
- **Project Title**: Image Generation and Representation Learning with Variational Autoencoders
- **Domain**: Computer Vision & Generative Modeling
- **Dataset**: MNIST Handwritten Digits

---

## Objectives of the Project

### Primary Objectives
1. **Implement a foundational VAE** using convolutional and deconvolutional neural networks
2. **Train the model on MNIST** to learn a compressed probabilistic representation
3. **Generate novel digit images** by sampling from the learned latent space
4. **Visualize and analyze** the structure of the learned latent space
5. **Evaluate model performance** using both visual and quantitative metrics

### Secondary Objectives
- Understand probabilistic representation learning principles
- Compare VAE performance with other generative models
- Explore uncertainty quantification in generative modeling

---

## Mapping Objectives to Course Scope

| Course Concept | Project Implementation |
|---|---|
| **Probabilistic Graphical Models** | VAE as directed graphical model p(x,z) = p(z)p(x\|z) |
| **Bayesian Inference** | Approximate posterior q(z\|x) ≈ p(z\|x) using variational inference |
| **Latent Variable Models** | Hidden latent space z generating observed images x |
| **Variational Methods** | ELBO optimization and reparameterization trick |
| **Learning in PGMs** | Maximum likelihood estimation via ELBO maximization |
| **Uncertainty Representation** | Probabilistic latent encodings with mean and variance |

---

## Methodology - PGM Used

### Probabilistic Graphical Model Structure
- **Model Type**: Directed Graphical Model (Bayesian Network)
- **Variables**: 
  - Latent variables z ~ N(0, I)
  - Observed variables x (images)
- **Dependencies**: x depends on z through decoder p(x|z)

### Domain of Application
- **Computer Vision**: Handwritten digit generation and reconstruction
- **Representation Learning**: Learning meaningful latent representations
- **Anomaly Detection**: Potential application using reconstruction errors

### Integration with Deep Learning
- **Neural Networks as Function Approximators**: 
  - Encoder network approximates q(z|x)
  - Decoder network represents p(x|z)
- **Gradient-Based Learning**: End-to-end training using backpropagation
- **Scalability**: Handles high-dimensional data (28×28 images)

---

## Methodology - Inference & Learning

### Inference Methods
- **Variational Inference**: Approximate intractable posterior p(z|x)
- **Amortized Inference**: Neural network predicts variational parameters
- **Reparameterization Trick**: Enables gradient flow through sampling

### Learning Methods
- **Maximum Likelihood Estimation**: Via Evidence Lower Bound (ELBO)
- **Gradient Descent**: Adam optimizer for parameter updates
- **Loss Function**: L = Reconstruction Loss + β × KL Divergence

### ML/DL Integration Methodology
- **Convolutional Encoder**: CNN extracts features and outputs μ, σ²
- **Probabilistic Sampling**: Sample z from N(μ, σ²)
- **Deconvolutional Decoder**: Transposed CNN reconstructs images from z

---

## Addressing Suggestions from Zeroth Review

### Feedback Received:
*"Appreciate the use of Variational Autoencoders for doing this project, but to be more aligned with the syllabus, briefly contrast VAEs with other probabilistic models encountered in the course, such as Bayesian networks for discrete graphical modeling. You may even look for a comparison with GANs (which are not Bayesian, but often compared in practice). Provide not just visual but quantitative measures of model uncertainty and generative quality."*

### Incorporating Suggestions:

#### ✅ **Implemented Comparisons**
- **VAEs vs Bayesian Networks**: Continuous vs discrete, scalability, inference methods
- **VAEs vs GANs**: Training stability, mode coverage, latent space structure
- **Probabilistic Model Positioning**: VAE as modern probabilistic reasoning approach

#### ✅ **Added Quantitative Metrics**
- **Uncertainty Measures**: Entropy of approximate posterior, KL divergence from prior
- **Generative Quality**: Fréchet Inception Distance (FID), Inception Score (IS)
- **Reconstruction Quality**: Binary cross-entropy loss, pixel-wise accuracy

#### ✅ **Enhanced Course Alignment**
- Connected VAEs to latent variable models, Bayesian inference, variational methods
- Demonstrated practical probabilistic reasoning with high-dimensional data

---

## Results & Discussion

### Quantitative Results
- **Model Uncertainty Metrics**:
  - Entropy of posterior: [Value] ± [Std]
  - KL divergence from prior: [Value] ± [Std]
- **Generative Quality Metrics**:
  - Fréchet Inception Distance: [Value]
  - Inception Score: [Value] ± [Std]
- **Reconstruction Performance**:
  - Test loss: [Value]
  - Reconstruction accuracy: [Value]%

### Key Findings
1. **Structured Latent Space**: Clear clustering of digit classes in 2D visualization
2. **Smooth Interpolation**: Continuous transitions between different digits
3. **Probabilistic Uncertainty**: Model captures uncertainty through posterior distributions
4. **Generative Capability**: Successfully generates diverse, realistic digit samples

### Comparison with Other Models
- **vs Bayesian Networks**: Better scalability but less interpretable structure
- **vs GANs**: More stable training and better mode coverage, but slightly blurrier outputs
- **Probabilistic Advantages**: Built-in uncertainty quantification and principled inference

---

## References

1. Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.
2. Rezende, D. J., Mohamed, S., & Wierstra, D. (2014). Stochastic backpropagation and approximate inference in deep generative models.
3. Doersch, C. (2016). Tutorial on variational autoencoders. arXiv preprint arXiv:1606.05908.
4. Higgins, I., et al. (2017). β-VAE: Learning basic visual concepts with a constrained variational framework.
5. Koller, D., & Friedman, N. (2009). Probabilistic graphical models: principles and techniques.
6. Bishop, C. M. (2006). Pattern recognition and machine learning. Springer.
