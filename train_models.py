"""
Train both VAE and Bayesian models for handwriting quality assessment
"""

import os
import numpy as np
from PIL import Image
import torch
import cv2
from pathlib import Path

# Import our models
from vae_handwriting_model import HandwritingQualityAssessor as VAEAssessor, create_synthetic_handwriting_data
from bayesian_handwriting_model import BayesianHandwritingAssessor, create_synthetic_handwriting_samples


def ensure_models_dir():
    """Ensure models directory exists."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    return models_dir


def create_diverse_training_data():
    """Create diverse training data for both models."""
    print("🎨 Creating diverse handwriting training data...")
    
    # Create VAE training data (flattened 28x28 images)
    vae_data = create_synthetic_handwriting_data(1000)
    
    # Create Bayesian training data (larger 128x128 images)  
    bayesian_data = create_synthetic_handwriting_samples(400)
    
    print(f"✅ Created {len(vae_data)} samples for VAE training")
    print(f"✅ Created {len(bayesian_data)} samples for Bayesian training")
    
    return vae_data, bayesian_data


def train_vae_model(vae_data):
    """Train the VAE model."""
    print("\n🧠 Training VAE Model...")
    print("=" * 50)
    
    # Initialize VAE assessor
    vae_assessor = VAEAssessor()
    
    # Create data loader
    train_loader = torch.utils.data.DataLoader(vae_data, batch_size=32, shuffle=True)
    
    # Train model
    print("Training VAE (this may take a few minutes)...")
    losses = vae_assessor.train_model(train_loader, epochs=100, learning_rate=1e-3)
    
    # Save model
    model_path = "models/handwriting_vae.pth"
    vae_assessor.save_model(model_path)
    
    print(f"✅ VAE model saved to {model_path}")
    print(f"📊 Final training loss: {losses[-1]:.4f}")
    
    return vae_assessor, losses


def train_bayesian_model(bayesian_data):
    """Train the Bayesian model."""
    print("\n📊 Training Bayesian Model...")
    print("=" * 50)
    
    # Initialize Bayesian assessor
    bayesian_assessor = BayesianHandwritingAssessor()
    
    # Train model (this will create labels automatically)
    features_df, labels = bayesian_assessor.train_bayesian_models(bayesian_data)
    
    # Save model
    model_path = "models/bayesian_handwriting_model.pkl"
    bayesian_assessor.save_model(model_path)
    
    print(f"✅ Bayesian model saved to {model_path}")
    
    # Show label distribution
    from collections import Counter
    label_counts = Counter(labels)
    print(f"📊 Training label distribution:")
    for quality, count in label_counts.items():
        print(f"   {quality}: {count} samples")
    
    return bayesian_assessor, features_df, labels


def test_both_models(vae_assessor, bayesian_assessor):
    """Test both models on sample data."""
    print("\n🧪 Testing Both Models...")
    print("=" * 50)
    
    # Create a test sample
    test_img = np.zeros((128, 128), dtype=np.uint8)
    # Draw a simple letter-like shape
    cv2.circle(test_img, (64, 64), 30, 255, 3)
    cv2.line(test_img, (34, 94), (94, 94), 255, 3)
    
    # Test VAE model
    print("🔍 VAE Assessment:")
    # Resize for VAE (28x28)
    vae_test_img = cv2.resize(test_img, (28, 28))
    vae_result = vae_assessor.assess_handwriting_quality(vae_test_img)
    
    if "error" not in vae_result:
        print(f"   Overall Quality Score: {vae_result['overall_quality_score']:.1f}/100")
        print(f"   Quality Category: {vae_result['quality_category']}")
        print(f"   Reconstruction Score: {vae_result['reconstruction_score']:.1f}")
        print(f"   Structure Score: {vae_result['structure_score']:.1f}")
        print(f"   Consistency Score: {vae_result['consistency_score']:.1f}")
    else:
        print(f"   ❌ {vae_result['error']}")
    
    # Test Bayesian model
    print("\n📊 Bayesian Assessment:")
    bayesian_result = bayesian_assessor.assess_quality_bayesian(test_img)
    
    if "error" not in bayesian_result:
        print(f"   Predicted Quality: {bayesian_result['predicted_quality']}")
        print(f"   Confidence: {bayesian_result['confidence']:.3f}")
        print(f"   Overall Quality Score: {bayesian_result['overall_quality_score']:.1f}/100")
        print(f"   Quality Probabilities:")
        for quality, prob in bayesian_result['quality_probabilities'].items():
            print(f"      {quality}: {prob:.3f}")
    else:
        print(f"   ❌ {bayesian_result['error']}")


def main():
    """Main training function."""
    print("🚀 Handwriting Quality Assessment - Model Training")
    print("=" * 60)
    
    # Ensure models directory exists
    models_dir = ensure_models_dir()
    print(f"📁 Models will be saved to: {models_dir}")
    
    try:
        # Create training data
        vae_data, bayesian_data = create_diverse_training_data()
        
        # Train VAE model
        vae_assessor, vae_losses = train_vae_model(vae_data)
        
        # Train Bayesian model  
        bayesian_assessor, features_df, labels = train_bayesian_model(bayesian_data)
        
        # Test both models
        test_both_models(vae_assessor, bayesian_assessor)
        
        print("\n🎉 Training Completed Successfully!")
        print("=" * 60)
        print("📂 Trained models available:")
        print("   • models/handwriting_vae.pth (VAE model)")
        print("   • models/bayesian_handwriting_model.pkl (Bayesian model)")
        print("\n🚀 Next steps:")
        print("   1. Run the updated Streamlit app to test both models")
        print("   2. Upload handwriting images to see quality assessments")
        print("   3. Compare results from both probabilistic approaches")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()