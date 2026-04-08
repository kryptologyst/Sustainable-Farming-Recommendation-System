#!/usr/bin/env python3
"""Main training script for sustainable farming recommendation system."""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data import FarmingDataGenerator
from models import FarmingModelTrainer
from evaluation import ModelEvaluator


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train sustainable farming recommendation models")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--output-dir", type=str, default="assets", help="Output directory for results")
    parser.add_argument("--config", type=str, default="configs/data.yaml", help="Configuration file path")
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    
    print("🌱 Sustainable Farming Recommendation System")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize components
    print("📊 Generating synthetic farming data...")
    data_generator = FarmingDataGenerator(args.config)
    features_df, recommendations = data_generator.create_dataset(args.n_samples)
    
    print(f"Generated {len(features_df)} samples with {len(features_df.columns)} features")
    print(f"Recommendation distribution:")
    print(recommendations.value_counts().sort_index())
    
    # Prepare train/test split
    print("\n🔄 Preparing train/test split...")
    X_train, X_test, y_train, y_test, scaler = data_generator.prepare_train_test(
        features_df, recommendations
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train models
    print("\n🤖 Training models...")
    trainer = FarmingModelTrainer(args.config)
    results = trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    # Add test labels to results for evaluation
    for model_name in results:
        results[model_name]['y_test'] = y_test
    
    # Evaluate models
    print("\n📈 Evaluating models...")
    evaluator = ModelEvaluator(args.config)
    evaluation_summary = evaluator.generate_evaluation_report(results, args.output_dir)
    
    # Print results
    print("\n🏆 Model Performance Summary:")
    print("=" * 50)
    leaderboard = evaluation_summary['leaderboard']
    print(leaderboard.to_string(index=False))
    
    print(f"\n🥇 Best Model: {evaluation_summary['best_model']}")
    print(f"🎯 Best Accuracy: {evaluation_summary['best_accuracy']:.4f}")
    
    # Domain-specific evaluation
    print("\n🌾 Domain-Specific Analysis:")
    print("=" * 50)
    domain_eval = evaluator.evaluate_recommendation_quality(
        y_test, results[evaluation_summary['best_model'].lower().replace(' ', '_')]['predictions'], 
        features_df.iloc[-len(y_test):]  # Get test features
    )
    
    print(f"Overall Accuracy: {domain_eval['overall_accuracy']:.4f}")
    print("\nPer-Class Performance:")
    for class_name, metrics in domain_eval['class_performance'].items():
        print(f"  {class_name}: Accuracy={metrics['accuracy']:.4f}, Support={metrics['support']}")
    
    # Save model artifacts
    print(f"\n💾 Saving model artifacts to {args.output_dir}/")
    
    # Save scaler
    import joblib
    joblib.dump(scaler, f"{args.output_dir}/scaler.pkl")
    
    # Save best model
    best_model_name = evaluation_summary['best_model'].lower().replace(' ', '_')
    if best_model_name in trainer.models:
        if best_model_name == 'neural_network':
            torch.save(trainer.models[best_model_name].state_dict(), f"{args.output_dir}/best_model.pth")
        else:
            joblib.dump(trainer.models[best_model_name], f"{args.output_dir}/best_model.pkl")
    
    # Save feature names and recommendations
    import json
    with open(f"{args.output_dir}/feature_names.json", 'w') as f:
        json.dump(data_generator.get_feature_names(), f)
    
    with open(f"{args.output_dir}/recommendations.json", 'w') as f:
        json.dump(data_generator.get_recommendation_names(), f)
    
    print("✅ Training completed successfully!")
    print(f"📁 Results saved to: {args.output_dir}/")
    print("🚀 Run 'streamlit run demo/app.py' to launch the interactive demo")


if __name__ == "__main__":
    main()
