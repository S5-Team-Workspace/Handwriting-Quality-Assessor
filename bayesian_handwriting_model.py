"""
Bayesian Handwriting Quality Assessment
Uses Bayesian inference to assess handwriting quality based on multiple probabilistic features
"""

import numpy as np
import pandas as pd
from PIL import Image
import cv2
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import pickle
import matplotlib.pyplot as plt


class BayesianHandwritingAssessor:
    """Bayesian approach to handwriting quality assessment."""
    
    def __init__(self):
        self.feature_extractors = {}
        self.quality_models = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Prior beliefs about handwriting quality
        self.quality_priors = {
            'excellent': 0.15,  # Most handwriting is not excellent
            'good': 0.35,       # Good chunk is good
            'fair': 0.35,       # Fair chunk is fair  
            'poor': 0.15        # Some is poor
        }
    
    def extract_stroke_features(self, image):
        """Extract stroke-related features from handwriting."""
        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        elif isinstance(image, Image.Image):
            img = np.array(image.convert('L'))
        else:
            img = image.copy()
        
        if img is None:
            return {}
        
        # Ensure binary image
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        
        features = {}
        
        # 1. Stroke width analysis
        kernel = np.ones((3,3), np.uint8)
        eroded = cv2.erode(binary, kernel, iterations=1)
        stroke_pixels = np.sum(binary > 0)
        eroded_pixels = np.sum(eroded > 0)
        features['stroke_consistency'] = eroded_pixels / max(stroke_pixels, 1)
        
        # 2. Smoothness analysis
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(largest_contour, True)
            area = cv2.contourArea(largest_contour)
            features['compactness'] = (4 * np.pi * area) / max(perimeter**2, 1)
        else:
            features['compactness'] = 0
        
        # 3. Density analysis
        features['density'] = stroke_pixels / (img.shape[0] * img.shape[1])
        
        # 4. Symmetry analysis
        height, width = img.shape
        left_half = binary[:, :width//2]
        right_half = np.fliplr(binary[:, width//2:])
        min_width = min(left_half.shape[1], right_half.shape[1])
        if min_width > 0:
            left_resized = left_half[:, :min_width]
            right_resized = right_half[:, :min_width]
            features['horizontal_symmetry'] = np.corrcoef(left_resized.flatten(), right_resized.flatten())[0,1]
            if np.isnan(features['horizontal_symmetry']):
                features['horizontal_symmetry'] = 0
        else:
            features['horizontal_symmetry'] = 0
        
        # 5. Regularity features
        row_sums = np.sum(binary, axis=1)
        col_sums = np.sum(binary, axis=0)
        features['vertical_regularity'] = 1.0 / (1.0 + np.std(row_sums))
        features['horizontal_regularity'] = 1.0 / (1.0 + np.std(col_sums))
        
        return features
    
    def extract_texture_features(self, image):
        """Extract texture-related features."""
        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        elif isinstance(image, Image.Image):
            img = np.array(image.convert('L'))
        else:
            img = image.copy()
            
        features = {}
        
        # Gradient analysis
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features['gradient_mean'] = np.mean(gradient_magnitude)
        features['gradient_std'] = np.std(gradient_magnitude)
        
        # Local binary pattern-like features
        features['texture_uniformity'] = self.calculate_texture_uniformity(img)
        
        return features
    
    def calculate_texture_uniformity(self, img):
        """Calculate texture uniformity measure."""
        # Simple local variance measure
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        local_mean = cv2.filter2D(img.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((img - local_mean)**2, -1, kernel)
        return 1.0 / (1.0 + np.mean(local_variance))
    
    def extract_all_features(self, image):
        """Extract all features for quality assessment."""
        stroke_features = self.extract_stroke_features(image)
        texture_features = self.extract_texture_features(image)
        
        # Combine all features
        all_features = {**stroke_features, **texture_features}
        
        return all_features
    
    def create_quality_labels(self, features_list):
        """Create quality labels based on feature combinations (for training)."""
        labels = []
        
        for features in features_list:
            # Simple heuristic labeling (in real scenario, you'd have ground truth)
            score = 0
            
            # High stroke consistency is good
            if features.get('stroke_consistency', 0) > 0.7:
                score += 2
            elif features.get('stroke_consistency', 0) > 0.5:
                score += 1
            
            # Good compactness
            if features.get('compactness', 0) > 0.3:
                score += 2
            elif features.get('compactness', 0) > 0.1:
                score += 1
            
            # Good density (not too sparse, not too dense)
            density = features.get('density', 0)
            if 0.1 < density < 0.4:
                score += 2
            elif 0.05 < density < 0.6:
                score += 1
            
            # Good regularity
            v_reg = features.get('vertical_regularity', 0)
            h_reg = features.get('horizontal_regularity', 0)
            if v_reg > 0.3 and h_reg > 0.3:
                score += 2
            elif v_reg > 0.2 or h_reg > 0.2:
                score += 1
            
            # Convert score to label
            if score >= 6:
                labels.append('excellent')
            elif score >= 4:
                labels.append('good')
            elif score >= 2:
                labels.append('fair')
            else:
                labels.append('poor')
        
        return labels
    
    def train_bayesian_models(self, images, labels=None):
        """Train Bayesian models for quality assessment."""
        print("Extracting features from training images...")
        
        # Extract features
        features_list = []
        for i, img in enumerate(images):
            if i % 50 == 0:
                print(f"Processing image {i+1}/{len(images)}")
            features = self.extract_all_features(img)
            features_list.append(features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        features_df = features_df.fillna(0)  # Handle any NaN values
        
        # Create labels if not provided
        if labels is None:
            print("Creating quality labels...")
            labels = self.create_quality_labels(features_list)
        
        # Scale features
        feature_matrix = features_df.values
        self.scaler.fit(feature_matrix)
        scaled_features = self.scaler.transform(feature_matrix)
        
        # Train Gaussian Mixture Models for each quality class
        print("Training Bayesian models for each quality class...")
        unique_labels = list(set(labels))
        
        for quality_class in unique_labels:
            # Get features for this quality class
            class_mask = [label == quality_class for label in labels]
            class_features = scaled_features[class_mask]
            
            if len(class_features) > 0:
                # Train Gaussian Mixture Model
                n_components = min(3, len(class_features))  # Adaptive number of components
                gmm = GaussianMixture(n_components=n_components, random_state=42)
                gmm.fit(class_features)
                self.quality_models[quality_class] = gmm
                
                print(f"Trained model for '{quality_class}' quality with {len(class_features)} samples")
        
        self.is_trained = True
        print("Bayesian model training completed!")
        
        return features_df, labels
    
    def calculate_likelihood(self, features, quality_class):
        """Calculate likelihood of features given quality class."""
        if quality_class not in self.quality_models:
            return 1e-10  # Very small probability
        
        model = self.quality_models[quality_class]
        
        # Convert features to array and scale
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        scaled_features = self.scaler.transform(feature_vector)
        
        # Calculate log likelihood and convert to probability
        log_likelihood = model.score_samples(scaled_features)[0]
        likelihood = np.exp(log_likelihood)
        
        return likelihood
    
    def assess_quality_bayesian(self, image):
        """Assess handwriting quality using Bayesian inference."""
        if not self.is_trained:
            return {"error": "Model not trained yet. Please train the model first."}
        
        # Extract features
        features = self.extract_all_features(image)
        
        # Calculate posterior probabilities for each quality class
        posteriors = {}
        evidence = 0
        
        for quality_class in self.quality_models.keys():
            # P(features|quality) * P(quality)
            likelihood = self.calculate_likelihood(features, quality_class)
            prior = self.quality_priors.get(quality_class, 0.25)
            joint_prob = likelihood * prior
            
            posteriors[quality_class] = joint_prob
            evidence += joint_prob
        
        # Normalize to get posterior probabilities
        if evidence > 0:
            for quality_class in posteriors:
                posteriors[quality_class] /= evidence
        else:
            # Fallback to uniform distribution
            for quality_class in posteriors:
                posteriors[quality_class] = 1.0 / len(posteriors)
        
        # Determine most likely quality
        best_quality = max(posteriors.keys(), key=lambda k: posteriors[k])
        confidence = posteriors[best_quality]
        
        # Calculate overall quality score (0-100)
        quality_scores = {'excellent': 90, 'good': 70, 'fair': 50, 'poor': 30}
        overall_score = sum(posteriors[q] * quality_scores.get(q, 50) for q in posteriors)
        
        return {
            'predicted_quality': best_quality,
            'confidence': confidence,
            'overall_quality_score': overall_score,
            'quality_probabilities': posteriors,
            'extracted_features': features
        }
    
    def save_model(self, filepath):
        """Save the trained Bayesian model."""
        model_data = {
            'quality_models': self.quality_models,
            'scaler': self.scaler,
            'quality_priors': self.quality_priors,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Bayesian model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained Bayesian model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.quality_models = model_data['quality_models']
        self.scaler = model_data['scaler']
        self.quality_priors = model_data['quality_priors']
        self.is_trained = model_data['is_trained']
        
        print(f"Bayesian model loaded from {filepath}")


def create_synthetic_handwriting_samples(num_samples=200):
    """Create synthetic handwriting samples for training."""
    samples = []
    
    for i in range(num_samples):
        # Create a 128x128 image
        img = np.zeros((128, 128), dtype=np.uint8)
        
        # Add different quality levels
        quality_level = i % 4  # 0=poor, 1=fair, 2=good, 3=excellent
        
        if quality_level == 0:  # Poor quality
            # Irregular, shaky lines
            for _ in range(np.random.randint(3, 8)):
                pts = np.random.randint(10, 118, (np.random.randint(5, 15), 2))
                for j in range(len(pts)-1):
                    cv2.line(img, tuple(pts[j]), tuple(pts[j+1]), 255, np.random.randint(1, 4))
            # Add noise
            noise = np.random.randint(0, 100, img.shape)
            img = np.clip(img + noise * 0.3, 0, 255)
            
        elif quality_level == 1:  # Fair quality
            # Somewhat regular lines
            for _ in range(np.random.randint(2, 5)):
                start = (np.random.randint(20, 40), np.random.randint(30, 100))
                end = (np.random.randint(80, 108), np.random.randint(30, 100))
                cv2.line(img, start, end, 255, 2)
            
        elif quality_level == 2:  # Good quality
            # Regular, clean strokes
            for _ in range(np.random.randint(2, 4)):
                start = (np.random.randint(25, 35), np.random.randint(40, 80))
                end = (np.random.randint(85, 105), np.random.randint(40, 80))
                cv2.line(img, start, end, 255, 2)
                # Add some curves
                cv2.ellipse(img, (64, 64), (20, 30), 0, 0, 180, 255, 2)
        
        else:  # Excellent quality
            # Very clean, symmetric strokes
            center = (64, 64)
            cv2.circle(img, center, 25, 255, 2)
            cv2.line(img, (39, 40), (89, 40), 255, 2)
            cv2.line(img, (39, 88), (89, 88), 255, 2)
        
        samples.append(img.astype(np.uint8))
    
    return samples


if __name__ == "__main__":
    # Quick test
    print("Creating Bayesian Handwriting Quality Assessor...")
    assessor = BayesianHandwritingAssessor()
    
    # Create synthetic training data
    print("Generating synthetic handwriting samples...")
    training_samples = create_synthetic_handwriting_samples(200)
    
    # Train the model
    features_df, labels = assessor.train_bayesian_models(training_samples)
    
    # Save the model
    assessor.save_model("models/bayesian_handwriting_model.pkl")
    
    # Test on a sample
    print("\nTesting on a sample...")
    test_result = assessor.assess_quality_bayesian(training_samples[0])
    print(f"Quality assessment: {test_result}")
    
    print("Bayesian model training completed!")