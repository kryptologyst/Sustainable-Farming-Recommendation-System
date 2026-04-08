"""Data generation and preprocessing module for sustainable farming recommendations."""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml


class FarmingDataGenerator:
    """Generates synthetic farming data for sustainable agriculture recommendations."""
    
    def __init__(self, config_path: str = "configs/data.yaml"):
        """Initialize the data generator with configuration.
        
        Args:
            config_path: Path to the data configuration file.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.random_seed = self.config['data']['random_seed']
        np.random.seed(self.random_seed)
        
        # Feature mappings
        self.soil_types = ['Sandy', 'Loamy', 'Clay', 'Silt', 'Peat']
        self.crop_types = ['Wheat', 'Rice', 'Corn', 'Soybean', 'Vegetables']
        self.recommendations = self.config['model']['recommendations']
        
    def generate_features(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic farming features.
        
        Args:
            n_samples: Number of samples to generate.
            
        Returns:
            DataFrame with farming features.
        """
        data_config = self.config['data']
        
        # Generate features
        soil_type = np.random.randint(0, data_config['soil_types'], n_samples)
        water_availability = np.clip(
            np.random.normal(
                data_config['water_availability']['mean'],
                data_config['water_availability']['std'],
                n_samples
            ),
            data_config['water_availability']['min'],
            data_config['water_availability']['max']
        )
        avg_temperature = np.clip(
            np.random.normal(
                data_config['temperature']['mean'],
                data_config['temperature']['std'],
                n_samples
            ),
            data_config['temperature']['min'],
            data_config['temperature']['max']
        )
        crop_type = np.random.randint(0, data_config['crop_types'], n_samples)
        
        # Create DataFrame
        df = pd.DataFrame({
            'soil_type': soil_type,
            'water_availability': water_availability,
            'avg_temperature': avg_temperature,
            'crop_type': crop_type
        })
        
        # Add categorical labels
        df['soil_type_name'] = df['soil_type'].map(dict(enumerate(self.soil_types)))
        df['crop_type_name'] = df['crop_type'].map(dict(enumerate(self.crop_types)))
        
        return df
    
    def generate_recommendations(self, df: pd.DataFrame) -> pd.Series:
        """Generate sustainable farming recommendations based on heuristic rules.
        
        Args:
            df: DataFrame with farming features.
            
        Returns:
            Series with recommendation labels.
        """
        recommendations = np.zeros(len(df), dtype=int)
        
        # Heuristic rules for sustainable farming recommendations
        # Rule 1: Drip irrigation for clay soil with low water availability
        drip_condition = (df['soil_type'] == 2) & (df['water_availability'] < 50)
        recommendations[drip_condition] = 2
        
        # Rule 2: Organic farming for rice crops in hot climates
        organic_condition = (df['crop_type'] == 1) & (df['avg_temperature'] > 28)
        recommendations[organic_condition] = 1
        
        # Rule 3: Crop rotation for peat soil with high water availability
        rotation_condition = (df['soil_type'] == 4) & (df['water_availability'] > 70)
        recommendations[rotation_condition] = 0
        
        # Rule 4: Compost usage for all other cases
        compost_condition = ~(drip_condition | organic_condition | rotation_condition)
        recommendations[compost_condition] = 3
        
        return pd.Series(recommendations, name='recommendation')
    
    def create_dataset(self, n_samples: int = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Create complete dataset with features and recommendations.
        
        Args:
            n_samples: Number of samples. If None, uses config default.
            
        Returns:
            Tuple of (features_df, recommendations_series).
        """
        if n_samples is None:
            n_samples = self.config['data']['n_samples']
            
        features_df = self.generate_features(n_samples)
        recommendations = self.generate_recommendations(features_df)
        
        return features_df, recommendations
    
    def prepare_train_test(
        self, 
        features_df: pd.DataFrame, 
        recommendations: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
        """Prepare training and test datasets with scaling.
        
        Args:
            features_df: DataFrame with features.
            recommendations: Series with recommendations.
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, scaler).
        """
        # Select numerical features
        feature_cols = ['soil_type', 'water_availability', 'avg_temperature', 'crop_type']
        X = features_df[feature_cols].values
        y = recommendations.values
        
        # Train-test split
        test_size = self.config['data']['test_size']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_seed, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    
    def get_feature_names(self) -> list:
        """Get feature column names.
        
        Returns:
            List of feature names.
        """
        return ['soil_type', 'water_availability', 'avg_temperature', 'crop_type']
    
    def get_recommendation_names(self) -> Dict[int, str]:
        """Get recommendation label mapping.
        
        Returns:
            Dictionary mapping recommendation IDs to names.
        """
        return self.recommendations
