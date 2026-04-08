"""Evaluation and metrics for sustainable farming recommendation models."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml


class ModelEvaluator:
    """Evaluator for farming recommendation models with domain-specific metrics."""
    
    def __init__(self, config_path: str = "configs/data.yaml"):
        """Initialize the evaluator with configuration.
        
        Args:
            config_path: Path to the configuration file.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.recommendations = self.config['model']['recommendations']
        self.feature_names = ['soil_type', 'water_availability', 'avg_temperature', 'crop_type']
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive metrics for model evaluation.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            
        Returns:
            Dictionary with calculated metrics.
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted')
        }
        
        return metrics
    
    def create_leaderboard(self, results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Create a leaderboard comparing all models.
        
        Args:
            results: Dictionary with model results.
            
        Returns:
            DataFrame with model comparison.
        """
        leaderboard_data = []
        
        for model_name, result in results.items():
            metrics = self.calculate_metrics(
                result.get('y_test', []), 
                result.get('predictions', [])
            )
            
            leaderboard_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Test Accuracy': metrics['accuracy'],
                'F1 Score (Macro)': metrics['f1_macro'],
                'Precision (Macro)': metrics['precision_macro'],
                'Recall (Macro)': metrics['recall_macro'],
                'Train Accuracy': result.get('train_accuracy', 0)
            })
        
        leaderboard = pd.DataFrame(leaderboard_data)
        leaderboard = leaderboard.sort_values('Test Accuracy', ascending=False)
        
        return leaderboard
    
    def plot_confusion_matrices(self, results: Dict[str, Dict[str, Any]], save_path: str = None):
        """Plot confusion matrices for all models.
        
        Args:
            results: Dictionary with model results.
            save_path: Optional path to save the plot.
        """
        n_models = len(results)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, (model_name, result) in enumerate(results.items()):
            if idx >= 4:  # Limit to 4 models
                break
                
            cm = result.get('confusion_matrix', np.array([[0]]))
            
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=list(self.recommendations.values()),
                yticklabels=list(self.recommendations.values()),
                ax=axes[idx]
            )
            axes[idx].set_title(f'{model_name.replace("_", " ").title()} Confusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        # Hide unused subplots
        for idx in range(n_models, 4):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_comparison(self, leaderboard: pd.DataFrame, save_path: str = None):
        """Plot model comparison charts.
        
        Args:
            leaderboard: DataFrame with model comparison.
            save_path: Optional path to save the plot.
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Test Accuracy', 'F1 Score (Macro)', 'Precision (Macro)', 'Recall (Macro)'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        metrics = ['Test Accuracy', 'F1 Score (Macro)', 'Precision (Macro)', 'Recall (Macro)']
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        colors = ['#2E8B57', '#228B22', '#32CD32', '#8B4513']
        
        for metric, (row, col), color in zip(metrics, positions, colors):
            fig.add_trace(
                go.Bar(
                    x=leaderboard['Model'],
                    y=leaderboard[metric],
                    name=metric,
                    marker_color=color,
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title_text="Model Performance Comparison",
            height=800,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    def analyze_feature_importance(self, results: Dict[str, Dict[str, Any]], save_path: str = None):
        """Analyze and plot feature importance for tree-based models.
        
        Args:
            results: Dictionary with model results.
            save_path: Optional path to save the plot.
        """
        tree_models = ['random_forest', 'xgboost']
        
        fig, axes = plt.subplots(1, len(tree_models), figsize=(15, 6))
        if len(tree_models) == 1:
            axes = [axes]
        
        for idx, model_name in enumerate(tree_models):
            if model_name in results and 'feature_importance' in results[model_name]:
                importance = results[model_name]['feature_importance']
                
                # Create DataFrame for easier plotting
                importance_df = pd.DataFrame({
                    'Feature': self.feature_names,
                    'Importance': importance
                }).sort_values('Importance', ascending=True)
                
                axes[idx].barh(importance_df['Feature'], importance_df['Importance'])
                axes[idx].set_title(f'{model_name.replace("_", " ").title()} Feature Importance')
                axes[idx].set_xlabel('Importance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_evaluation_report(
        self, 
        results: Dict[str, Dict[str, Any]], 
        save_dir: str = "assets"
    ) -> Dict[str, Any]:
        """Generate comprehensive evaluation report.
        
        Args:
            results: Dictionary with model results.
            save_dir: Directory to save plots and reports.
            
        Returns:
            Dictionary with evaluation summary.
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Create leaderboard
        leaderboard = self.create_leaderboard(results)
        
        # Generate plots
        self.plot_confusion_matrices(results, f"{save_dir}/confusion_matrices.png")
        self.plot_model_comparison(leaderboard, f"{save_dir}/model_comparison.html")
        self.analyze_feature_importance(results, f"{save_dir}/feature_importance.png")
        
        # Save leaderboard
        leaderboard.to_csv(f"{save_dir}/model_leaderboard.csv", index=False)
        
        # Generate summary
        best_model = leaderboard.iloc[0]['Model']
        best_accuracy = leaderboard.iloc[0]['Test Accuracy']
        
        summary = {
            'best_model': best_model,
            'best_accuracy': best_accuracy,
            'leaderboard': leaderboard,
            'total_models': len(results),
            'recommendation_classes': len(self.recommendations)
        }
        
        return summary
    
    def evaluate_recommendation_quality(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        feature_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Evaluate recommendation quality with domain-specific insights.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            feature_df: DataFrame with features.
            
        Returns:
            Dictionary with domain-specific evaluation metrics.
        """
        # Calculate per-class performance
        class_performance = {}
        for class_id, class_name in self.recommendations.items():
            mask = y_true == class_id
            if mask.sum() > 0:
                class_pred = y_pred[mask]
                class_true = y_true[mask]
                class_performance[class_name] = {
                    'accuracy': accuracy_score(class_true, class_pred),
                    'support': mask.sum(),
                    'precision': precision_score(class_true, class_pred, average='macro'),
                    'recall': recall_score(class_true, class_pred, average='macro')
                }
        
        # Analyze recommendations by soil type
        soil_analysis = {}
        for soil_type in range(5):
            soil_mask = feature_df['soil_type'] == soil_type
            if soil_mask.sum() > 0:
                soil_pred = y_pred[soil_mask]
                soil_true = y_true[soil_mask]
                soil_analysis[f'Soil_{soil_type}'] = {
                    'accuracy': accuracy_score(soil_true, soil_pred),
                    'support': soil_mask.sum(),
                    'most_common_pred': np.bincount(soil_pred).argmax(),
                    'most_common_true': np.bincount(soil_true).argmax()
                }
        
        return {
            'class_performance': class_performance,
            'soil_analysis': soil_analysis,
            'overall_accuracy': accuracy_score(y_true, y_pred)
        }
