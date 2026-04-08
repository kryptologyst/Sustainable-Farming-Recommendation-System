#!/usr/bin/env python3
"""Quick demo script for sustainable farming recommendation system."""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data import FarmingDataGenerator
from models import FarmingModelTrainer
from evaluation import ModelEvaluator


def quick_demo():
    """Run a quick demonstration of the system."""
    print("🌱 Sustainable Farming Recommendation System - Quick Demo")
    print("=" * 60)
    
    # Generate small dataset
    print("📊 Generating sample data...")
    data_generator = FarmingDataGenerator()
    features_df, recommendations = data_generator.create_dataset(50)
    
    print(f"Generated {len(features_df)} samples")
    print(f"Recommendation distribution:")
    print(recommendations.value_counts().sort_index())
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = data_generator.prepare_train_test(
        features_df, recommendations
    )
    
    # Train a simple model
    print("\n🤖 Training Logistic Regression model...")
    trainer = FarmingModelTrainer()
    result = trainer.train_logistic_regression(X_train, y_train, X_test, y_test)
    
    print(f"✅ Model trained successfully!")
    print(f"📈 Test Accuracy: {result['test_accuracy']:.4f}")
    
    # Make a sample prediction
    print("\n🎯 Sample Prediction:")
    sample_features = np.array([[2, 45, 28, 1]])  # Clay soil, 45% water, 28°C, Rice
    sample_scaled = scaler.transform(sample_features)
    prediction = trainer.predict('logistic_regression', sample_scaled)
    
    recommendation_names = data_generator.get_recommendation_names()
    print(f"Input: Clay soil, 45% water availability, 28°C, Rice crop")
    print(f"Recommendation: {recommendation_names[prediction[0]]}")
    
    print("\n🚀 System is ready! Run 'streamlit run demo/app.py' for the full interactive demo.")


if __name__ == "__main__":
    quick_demo()
