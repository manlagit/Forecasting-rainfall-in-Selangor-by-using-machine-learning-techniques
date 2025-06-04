"""
Data preprocessing module for rainfall forecasting.
Handles data cleaning, outlier detection, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Tuple, Optional
import logging
import yaml
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles all data preprocessing tasks."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize preprocessor with configuration."""
        self.config = self._load_config(config_path)
        self.preprocessing_config = self.config['preprocessing']
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by handling missing values and outliers.
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        logger.info("Starting data cleaning...")
        df = df.copy()
        
        # Handle missing values with mean imputation
        numeric_cols = ['Temp_avg', 'Relative_Humidity', 'Wind_kmh', 'Precipitation_mm']
        
        for col in numeric_cols:
            if df[col].isnull().any():
                mean_val = df[col].mean()
                df[col].fillna(mean_val, inplace=True)
                logger.info(f"Imputed {col} with mean value: {mean_val:.2f}")
        
        # Remove outliers using IQR method
        df = self._remove_outliers(df)
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers based on configured ranges."""
        logger.info("Removing outliers...")
        initial_count = len(df)
        
        ranges = self.preprocessing_config['valid_ranges']
        
        # Apply range filters
        df = df[
            (df['Temp_avg'] >= ranges['temperature'][0]) & 
            (df['Temp_avg'] <= ranges['temperature'][1]) &
            (df['Relative_Humidity'] >= ranges['humidity'][0]) & 
            (df['Relative_Humidity'] <= ranges['humidity'][1]) &
            (df['Wind_kmh'] >= ranges['wind'][0]) & 
            (df['Wind_kmh'] <= ranges['wind'][1]) &
            (df['Precipitation_mm'] >= ranges['precipitation'][0]) & 
            (df['Precipitation_mm'] <= ranges['precipitation'][1])
        ]
        
        removed_count = initial_count - len(df)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} outlier records")
        
        return df.reset_index(drop=True)
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with engineered features
        """
        logger.info("Creating engineered features...")
        df = df.copy()
        
        # Create lag features
        lag_features = self.preprocessing_config['lag_features']
        for feature in lag_features:
            base_col = feature.replace('_lag_1', '')
            if base_col == 'precipitation':
                base_col = 'Precipitation_mm'
            elif base_col == 'temp':
                base_col = 'Temp_avg'
            elif base_col == 'humidity':
                base_col = 'Relative_Humidity'
            
            df[feature] = df[base_col].shift(1)
        
        # Create moving averages
        ma_config = self.preprocessing_config['moving_averages']
        df['precipitation_ma_3'] = df['Precipitation_mm'].rolling(window=ma_config['precipitation_ma']).mean()
        df['temp_ma_4'] = df['Temp_avg'].rolling(window=ma_config['temp_ma']).mean()
        df['humidity_ma_3'] = df['Relative_Humidity'].rolling(window=ma_config['humidity_ma']).mean()
        
        # Create seasonal features
        df['Month'] = df['Date'].dt.month
        df['monsoon_season'] = df['Month'].isin(self.preprocessing_config['monsoon_months']).astype(int)
        df['dry_season'] = df['Month'].isin(self.preprocessing_config['dry_months']).astype(int)
        
        # Cyclical encoding for week of year
        df['week_of_year'] = df['Date'].dt.isocalendar().week
        df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
        
        # Create interaction features
        df['temp_humidity_interaction'] = df['Temp_avg'] * df['Relative_Humidity']
        df['wind_precipitation_ratio'] = df['Wind_kmh'] / (df['Precipitation_mm'] + 1)
        
        # Drop rows with NaN values created by lag and rolling features
        df = df.dropna().reset_index(drop=True)
        
        logger.info(f"Created {len(df.columns) - 7} new features")
        logger.info(f"Dataset now has {len(df)} records after feature engineering")
        
        return df
    
    def normalize_data(self, df: pd.DataFrame, 
                      target_col: str = 'Precipitation_mm') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Normalize features and target variable.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            
        Returns:
            Tuple of normalized features and target
        """
        logger.info("Normalizing data...")
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in ['Date', target_col]]
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Fit and transform features
        X_normalized = pd.DataFrame(
            self.feature_scaler.fit_transform(X),
            columns=feature_cols
        )
        
        # Fit and transform target
        y_normalized = pd.Series(
            self.target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten(),
            name=target_col
        )
        
        logger.info("Data normalization completed")
        
        return X_normalized, y_normalized
    
    def save_scalers(self, output_dir: str = "models/scalers"):
        """Save fitted scalers for later use."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.feature_scaler, output_path / 'feature_scaler.pkl')
        joblib.dump(self.target_scaler, output_path / 'target_scaler.pkl')
        logger.info(f"Saved scalers to {output_path}")
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Tuple of (normalized features, normalized target, processed DataFrame with Date)
        """
        # Clean data
        df_cleaned = self.clean_data(df)
        
        # Create features
        df_features = self.create_features(df_cleaned)
        
        # Save processed data with Date for visualization
        processed_path = Path(self.config['data']['processed_path'])
        processed_path.mkdir(parents=True, exist_ok=True)
        
        df_features.to_csv(processed_path / 'processed_data.csv', index=False)
        
        # Normalize data
        X_normalized, y_normalized = self.normalize_data(df_features)
        
        # Save scalers
        self.save_scalers()
        
        # Create DataFrame with Date for later use
        df_with_date = pd.DataFrame(X_normalized)
        df_with_date['Date'] = df_features['Date'].values
        df_with_date['Precipitation_mm'] = y_normalized.values
        
        return X_normalized, y_normalized, df_with_date
