"""Unit tests for sustainable farming recommendation system."""

import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import patch, mock_open
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data import FarmingDataGenerator
from models import FarmingModelTrainer, NeuralNetwork
from evaluation import ModelEvaluator


class TestFarmingDataGenerator:
    """Test cases for FarmingDataGenerator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock config
        self.mock_config = {
            'data': {
                'n_samples': 100,
                'test_size': 0.2,
                'random_seed': 42,
                'soil_types': 5,
                'crop_types': 5,
                'water_availability': {'mean': 60, 'std': 20, 'min': 0, 'max': 100},
                'temperature': {'mean': 26, 'std': 3, 'min': 15, 'max': 40}
            },
            'model': {
                'recommendations': {0: 'Crop Rotation', 1: 'Organic Farming', 2: 'Drip Irrigation', 3: 'Compost Usage'}
            }
        }
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('yaml.safe_load')
    def test_init(self, mock_yaml_load, mock_file):
        """Test FarmingDataGenerator initialization."""
        mock_yaml_load.return_value = self.mock_config
        
        generator = FarmingDataGenerator("test_config.yaml")
        
        assert generator.random_seed == 42
        assert generator.soil_types == ['Sandy', 'Loamy', 'Clay', 'Silt', 'Peat']
        assert generator.crop_types == ['Wheat', 'Rice', 'Corn', 'Soybean', 'Vegetables']
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('yaml.safe_load')
    def test_generate_features(self, mock_yaml_load, mock_file):
        """Test feature generation."""
        mock_yaml_load.return_value = self.mock_config
        
        generator = FarmingDataGenerator("test_config.yaml")
        features_df = generator.generate_features(10)
        
        assert len(features_df) == 10
        assert 'soil_type' in features_df.columns
        assert 'water_availability' in features_df.columns
        assert 'avg_temperature' in features_df.columns
        assert 'crop_type' in features_df.columns
        assert 'soil_type_name' in features_df.columns
        assert 'crop_type_name' in features_df.columns
        
        # Check value ranges
        assert features_df['soil_type'].min() >= 0
        assert features_df['soil_type'].max() < 5
        assert features_df['crop_type'].min() >= 0
        assert features_df['crop_type'].max() < 5
        assert features_df['water_availability'].min() >= 0
        assert features_df['water_availability'].max() <= 100
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('yaml.safe_load')
    def test_generate_recommendations(self, mock_yaml_load, mock_file):
        """Test recommendation generation."""
        mock_yaml_load.return_value = self.mock_config
        
        generator = FarmingDataGenerator("test_config.yaml")
        
        # Create test dataframe
        test_df = pd.DataFrame({
            'soil_type': [2, 1, 4, 0, 3],
            'water_availability': [30, 60, 80, 40, 50],
            'avg_temperature': [25, 30, 20, 28, 26],
            'crop_type': [0, 1, 2, 3, 4]
        })
        
        recommendations = generator.generate_recommendations(test_df)
        
        assert len(recommendations) == 5
        assert recommendations.dtype == int
        assert recommendations.min() >= 0
        assert recommendations.max() <= 3
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('yaml.safe_load')
    def test_create_dataset(self, mock_yaml_load, mock_file):
        """Test complete dataset creation."""
        mock_yaml_load.return_value = self.mock_config
        
        generator = FarmingDataGenerator("test_config.yaml")
        features_df, recommendations = generator.create_dataset(50)
        
        assert len(features_df) == 50
        assert len(recommendations) == 50
        assert len(features_df) == len(recommendations)


class TestNeuralNetwork:
    """Test cases for NeuralNetwork."""
    
    def test_init(self):
        """Test NeuralNetwork initialization."""
        model = NeuralNetwork(input_size=4, hidden_sizes=[64, 32], num_classes=4)
        
        assert model is not None
        assert len(list(model.parameters())) > 0
    
    def test_forward(self):
        """Test forward pass."""
        model = NeuralNetwork(input_size=4, hidden_sizes=[64, 32], num_classes=4)
        
        # Create dummy input
        x = torch.randn(1, 4)
        output = model(x)
        
        assert output.shape == (1, 4)  # batch_size, num_classes


class TestFarmingModelTrainer:
    """Test cases for FarmingModelTrainer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = {
            'data': {
                'random_seed': 42,
                'test_size': 0.2
            },
            'model': {
                'neural_network': {
                    'hidden_layers': [32, 16],
                    'activation': 'relu',
                    'epochs': 5,  # Reduced for testing
                    'batch_size': 16,
                    'learning_rate': 0.001
                },
                'xgboost': {
                    'n_estimators': 10,  # Reduced for testing
                    'max_depth': 3,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            }
        }
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('yaml.safe_load')
    def test_init(self, mock_yaml_load, mock_file):
        """Test FarmingModelTrainer initialization."""
        mock_yaml_load.return_value = self.mock_config
        
        trainer = FarmingModelTrainer("test_config.yaml")
        
        assert trainer.device is not None
        assert isinstance(trainer.models, dict)
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('yaml.safe_load')
    def test_train_logistic_regression(self, mock_yaml_load, mock_file):
        """Test logistic regression training."""
        mock_yaml_load.return_value = self.mock_config
        
        trainer = FarmingModelTrainer("test_config.yaml")
        
        # Create dummy data
        X_train = np.random.randn(80, 4)
        y_train = np.random.randint(0, 4, 80)
        X_test = np.random.randn(20, 4)
        y_test = np.random.randint(0, 4, 20)
        
        result = trainer.train_logistic_regression(X_train, y_train, X_test, y_test)
        
        assert 'model' in result
        assert 'train_accuracy' in result
        assert 'test_accuracy' in result
        assert 'predictions' in result
        assert len(result['predictions']) == len(y_test)


class TestModelEvaluator:
    """Test cases for ModelEvaluator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = {
            'model': {
                'recommendations': {0: 'Crop Rotation', 1: 'Organic Farming', 2: 'Drip Irrigation', 3: 'Compost Usage'}
            }
        }
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('yaml.safe_load')
    def test_init(self, mock_yaml_load, mock_file):
        """Test ModelEvaluator initialization."""
        mock_yaml_load.return_value = self.mock_config
        
        evaluator = ModelEvaluator("test_config.yaml")
        
        assert evaluator.recommendations is not None
        assert evaluator.feature_names is not None
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('yaml.safe_load')
    def test_calculate_metrics(self, mock_yaml_load, mock_file):
        """Test metrics calculation."""
        mock_yaml_load.return_value = self.mock_config
        
        evaluator = ModelEvaluator("test_config.yaml")
        
        y_true = np.array([0, 1, 2, 3, 0, 1])
        y_pred = np.array([0, 1, 2, 3, 0, 1])
        
        metrics = evaluator.calculate_metrics(y_true, y_pred)
        
        assert 'accuracy' in metrics
        assert 'precision_macro' in metrics
        assert 'recall_macro' in metrics
        assert 'f1_macro' in metrics
        assert metrics['accuracy'] == 1.0  # Perfect prediction
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('yaml.safe_load')
    def test_create_leaderboard(self, mock_yaml_load, mock_file):
        """Test leaderboard creation."""
        mock_yaml_load.return_value = self.mock_config
        
        evaluator = ModelEvaluator("test_config.yaml")
        
        # Mock results
        results = {
            'model1': {
                'y_test': np.array([0, 1, 2, 3]),
                'predictions': np.array([0, 1, 2, 3]),
                'train_accuracy': 0.9
            },
            'model2': {
                'y_test': np.array([0, 1, 2, 3]),
                'predictions': np.array([0, 1, 2, 2]),  # One wrong prediction
                'train_accuracy': 0.85
            }
        }
        
        leaderboard = evaluator.create_leaderboard(results)
        
        assert len(leaderboard) == 2
        assert 'Model' in leaderboard.columns
        assert 'Test Accuracy' in leaderboard.columns
        assert 'F1 Score (Macro)' in leaderboard.columns


if __name__ == "__main__":
    pytest.main([__file__])
