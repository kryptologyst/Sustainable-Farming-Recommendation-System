"""Machine learning models for sustainable farming recommendations."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yaml


class NeuralNetwork(nn.Module):
    """Neural network for farming recommendation classification."""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int):
        """Initialize the neural network.
        
        Args:
            input_size: Number of input features.
            hidden_sizes: List of hidden layer sizes.
            num_classes: Number of output classes.
        """
        super(NeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output logits.
        """
        return self.network(x)


class FarmingModelTrainer:
    """Trainer for multiple farming recommendation models."""
    
    def __init__(self, config_path: str = "configs/data.yaml"):
        """Initialize the trainer with configuration.
        
        Args:
            config_path: Path to the configuration file.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = self._get_device()
        self.models = {}
        self.scalers = {}
        
    def _get_device(self) -> torch.device:
        """Get the best available device (CUDA -> MPS -> CPU).
        
        Returns:
            PyTorch device.
        """
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def train_logistic_regression(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Train logistic regression model.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            X_test: Test features.
            y_test: Test labels.
            
        Returns:
            Dictionary with model and metrics.
        """
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        y_pred = model.predict(X_test)
        
        self.models['logistic_regression'] = model
        
        return {
            'model': model,
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'predictions': y_pred,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def train_random_forest(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Train random forest model.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            X_test: Test features.
            y_test: Test labels.
            
        Returns:
            Dictionary with model and metrics.
        """
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        y_pred = model.predict(X_test)
        
        self.models['random_forest'] = model
        
        return {
            'model': model,
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'predictions': y_pred,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': model.feature_importances_
        }
    
    def train_xgboost(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Train XGBoost model.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            X_test: Test features.
            y_test: Test labels.
            
        Returns:
            Dictionary with model and metrics.
        """
        xgb_config = self.config['model']['xgboost']
        
        model = xgb.XGBClassifier(
            n_estimators=xgb_config['n_estimators'],
            max_depth=xgb_config['max_depth'],
            learning_rate=xgb_config['learning_rate'],
            random_state=xgb_config['random_state'],
            eval_metric='mlogloss'
        )
        
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        y_pred = model.predict(X_test)
        
        self.models['xgboost'] = model
        
        return {
            'model': model,
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'predictions': y_pred,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': model.feature_importances_
        }
    
    def train_neural_network(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Train neural network model.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            X_test: Test features.
            y_test: Test labels.
            
        Returns:
            Dictionary with model and metrics.
        """
        nn_config = self.config['model']['neural_network']
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.LongTensor(y_test).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=nn_config['batch_size'], 
            shuffle=True
        )
        
        # Initialize model
        input_size = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        model = NeuralNetwork(
            input_size=input_size,
            hidden_sizes=nn_config['hidden_layers'],
            num_classes=num_classes
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=nn_config['learning_rate'])
        
        # Training loop
        model.train()
        for epoch in range(nn_config['epochs']):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            train_outputs = model(X_train_tensor)
            train_pred = torch.argmax(train_outputs, dim=1)
            train_accuracy = (train_pred == y_train_tensor).float().mean().item()
            
            test_outputs = model(X_test_tensor)
            test_pred = torch.argmax(test_outputs, dim=1)
            test_accuracy = (test_pred == y_test_tensor).float().mean().item()
            
            y_pred = test_pred.cpu().numpy()
        
        self.models['neural_network'] = model
        
        return {
            'model': model,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'predictions': y_pred,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def train_all_models(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        """Train all available models.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            X_test: Test features.
            y_test: Test labels.
            
        Returns:
            Dictionary with results for all models.
        """
        results = {}
        
        print("Training Logistic Regression...")
        results['logistic_regression'] = self.train_logistic_regression(
            X_train, y_train, X_test, y_test
        )
        
        print("Training Random Forest...")
        results['random_forest'] = self.train_random_forest(
            X_train, y_train, X_test, y_test
        )
        
        print("Training XGBoost...")
        results['xgboost'] = self.train_xgboost(
            X_train, y_train, X_test, y_test
        )
        
        print("Training Neural Network...")
        results['neural_network'] = self.train_neural_network(
            X_train, y_train, X_test, y_test
        )
        
        return results
    
    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Make predictions using a trained model.
        
        Args:
            model_name: Name of the model to use.
            X: Features for prediction.
            
        Returns:
            Predictions array.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        if model_name == 'neural_network':
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                outputs = model(X_tensor)
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        else:
            predictions = model.predict(X)
        
        return predictions
