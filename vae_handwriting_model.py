"""
Variational Autoencoder for Handwriting Quality Assessment
Uses VAE reconstruction error and latent space analysis to assess handwriting quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2


class HandwritingVAE(nn.Module):
    """Variational Autoencoder specifically designed for handwriting analysis."""
    
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(HandwritingVAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU()
        )
        
        # Latent space
        self.mu_layer = nn.Linear(hidden_dim // 4, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim // 4, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Encode input to latent parameters."""
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent representation to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass through VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class HandwritingQualityAssessor:
    """Main class for assessing handwriting quality using VAE."""
    
    def __init__(self, model_path=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = HandwritingVAE().to(self.device)
        self.model_trained = False
        
        if model_path and torch.cuda.is_available():
            try:
                self.load_model(model_path)
            except Exception as e:
                print(f"Could not load model: {e}")
    
    def preprocess_image(self, image_input):
        """Preprocess image for VAE input."""
        if isinstance(image_input, str):
            # Load from file path
            image = Image.open(image_input).convert('L')
        elif isinstance(image_input, Image.Image):
            image = image_input.convert('L')
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 3:
                image = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
            else:
                image = image_input
            image = Image.fromarray(image)
        else:
            raise ValueError("Unsupported image input type")
        
        # Resize to standard size
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to tensor
        image_array = np.array(image) / 255.0
        image_tensor = torch.FloatTensor(image_array.flatten()).unsqueeze(0).to(self.device)
        
        return image_tensor, image_array
    
    def calculate_reconstruction_error(self, image_input):
        """Calculate reconstruction error as quality metric."""
        if not self.model_trained:
            return {"error": "Model not trained yet"}
        
        self.model.eval()
        with torch.no_grad():
            image_tensor, original = self.preprocess_image(image_input)
            recon, mu, logvar = self.model(image_tensor)
            
            # Calculate MSE reconstruction error
            mse_error = F.mse_loss(recon, image_tensor, reduction='mean').item()
            
            # Calculate latent space metrics
            latent_norm = torch.norm(mu, dim=1).item()
            latent_variance = torch.exp(logvar).mean().item()
            
            return {
                'reconstruction_error': mse_error,
                'latent_norm': latent_norm,
                'latent_variance': latent_variance,
                'original_shape': original.shape,
                'reconstructed': recon.cpu().numpy().reshape(28, 28)
            }
    
    def assess_handwriting_quality(self, image_input):
        """Comprehensive handwriting quality assessment."""
        metrics = self.calculate_reconstruction_error(image_input)
        
        if "error" in metrics:
            return metrics
        
        # Quality scoring based on reconstruction error and latent metrics
        recon_error = metrics['reconstruction_error']
        latent_norm = metrics['latent_norm']
        latent_var = metrics['latent_variance']
        
        # Lower reconstruction error = better quality
        # Moderate latent norm = good structure
        # Moderate latent variance = good encoding
        
        # Normalize scores (these thresholds would be tuned on training data)
        recon_score = max(0, 100 - (recon_error * 1000))  # Lower error = higher score
        structure_score = max(0, 100 - abs(latent_norm - 5) * 10)  # Optimal norm around 5
        consistency_score = max(0, 100 - abs(latent_var - 1) * 50)  # Optimal variance around 1
        
        overall_quality = (recon_score + structure_score + consistency_score) / 3
        
        # Determine quality category
        if overall_quality >= 80:
            quality_category = "Excellent"
        elif overall_quality >= 60:
            quality_category = "Good"
        elif overall_quality >= 40:
            quality_category = "Fair"
        else:
            quality_category = "Poor"
        
        return {
            'overall_quality_score': overall_quality,
            'quality_category': quality_category,
            'reconstruction_score': recon_score,
            'structure_score': structure_score,
            'consistency_score': consistency_score,
            'raw_metrics': metrics
        }
    
    def train_model(self, data_loader, epochs=100, learning_rate=1e-3):
        """Train the VAE model."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model.train()
        
        training_losses = []
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, data in enumerate(data_loader):
                data = data[0].to(self.device) if isinstance(data, (list, tuple)) else data.to(self.device)
                data = data.view(data.size(0), -1)  # Flatten
                
                optimizer.zero_grad()
                recon, mu, logvar = self.model(data)
                loss = self.vae_loss(recon, data, mu, logvar)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(data_loader)
            training_losses.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
        
        self.model_trained = True
        return training_losses
    
    def vae_loss(self, recon_x, x, mu, logvar, beta=1.0):
        """VAE loss function with reconstruction and KL divergence terms."""
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + beta * kl_loss
    
    def save_model(self, path):
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_trained': self.model_trained
        }, path)
    
    def load_model(self, path):
        """Load a trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model_trained = checkpoint.get('model_trained', True)
        self.model.eval()


def create_synthetic_handwriting_data(num_samples=1000):
    """Create synthetic handwriting data for training."""
    data = []
    
    for _ in range(num_samples):
        # Create random handwriting-like patterns
        img = np.zeros((28, 28))
        
        # Add random strokes
        num_strokes = np.random.randint(1, 4)
        for _ in range(num_strokes):
            # Random line/curve
            start_x, start_y = np.random.randint(5, 23, 2)
            end_x, end_y = np.random.randint(5, 23, 2)
            
            # Draw line with some thickness
            cv2.line(img, (start_x, start_y), (end_x, end_y), 255, thickness=np.random.randint(1, 3))
        
        # Add some noise
        noise = np.random.normal(0, 10, img.shape)
        img = np.clip(img + noise, 0, 255)
        
        data.append(img / 255.0)  # Normalize
    
    return torch.FloatTensor(data)


if __name__ == "__main__":
    # Quick test
    assessor = HandwritingQualityAssessor()
    
    # Create some test data
    print("Creating synthetic training data...")
    train_data = create_synthetic_handwriting_data(500)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    
    print("Training VAE model...")
    losses = assessor.train_model(train_loader, epochs=50)
    
    print("Saving model...")
    assessor.save_model("models/handwriting_vae.pth")
    
    print("Model training completed!")