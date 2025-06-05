"""
Feature Engineering Module
Builds additional features for the rainfall forecasting model.
"""

import pandas as pd
import logging
from src.utils.helpers import load_yaml
from typing import Dict, Any

class FeatureBuilder:
    """
    Builds features for the rainfall forecasting model.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize with configuration.
        """
        self.config = load_yaml(config_path).get('feature_engineering', {})
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized FeatureBuilder")
        
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build features including lag, moving averages, and seasonal indicators.
        """
        df = df.copy()
        
        try:
            # Column name mapping for shorthand names
            column_mapping = {
                'precipitation': 'Precipitation_mm',
                'temp': 'Temp_avg',
                'humidity': 'Relative_Humidity'
            }
            
            # Lag features with robust configuration handling
            lag_features = self.config.get('lag_features', [])
            for feature in lag_features:
                if isinstance(feature, str):
                    # Handle string format: "precipitation_lag_1"
                    base_name = feature.split('_lag_')[0]
                    col_name = column_mapping.get(base_name, base_name)
                    lag = int(feature.split('_')[-1])
                    df[feature] = df[col_name].shift(lag)
                elif isinstance(feature, dict):
                    # Handle dictionary format: {column: 'precipitation', lags: [1,2,3]}
                    col_name = feature.get('column')
                    lags = feature.get('lags', [])
                    for lag in lags:
                        df[f'{col_name}_lag_{lag}'] = df[col_name].shift(lag)
            
            # Moving averages with robust configuration handling
            moving_avgs = self.config.get('moving_averages', {})
            if isinstance(moving_avgs, list):
                # Handle list format: [{'column': 'precipitation', 'windows': [3,4]}]
                for feature in moving_avgs:
                    base_name = feature.get('column')
                    col_name = column_mapping.get(base_name, base_name)
                    windows = feature.get('windows', [])
                    for window in windows:
                        df[f'{col_name}_ma_{window}'] = df[col_name].rolling(window=window).mean()
            elif isinstance(moving_avgs, dict):
                # Handle dictionary format: {precipitation: [3,4], temp: [3,4]}
                for base_name, windows in moving_avgs.items():
                    col_name = column_mapping.get(base_name, base_name)
                    for window in windows:
                        df[f'{col_name}_ma_{window}'] = df[col_name].rolling(window=window).mean()
            
            # Seasonal indicators (e.g., monsoon)
            seasonal_config = self.config.get('seasonal_features', {})
            if 'monsoon_months' in seasonal_config:
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df['is_monsoon'] = df['Date'].dt.month.isin(seasonal_config['monsoon_months'])
            
            # Time-based features
            if 'Date' in df.columns:
                df['Month'] = df['Date'].dt.month
                df['DayOfWeek'] = df['Date'].dt.dayofweek
            
            # Rolling standard deviation
            rolling_std_cols = ['Temp_avg', 'Relative_Humidity', 'Wind_kmh', 'Precipitation_mm']
            for col in rolling_std_cols:
                if col in df.columns:
                    df[f'{col}_rolling_std'] = df[col].rolling(window=3).std()
            
            # Interaction features
            if 'Temp_avg' in df.columns and 'Relative_Humidity' in df.columns:
                df['Temp_Humidity'] = df['Temp_avg'] * df['Relative_Humidity']
        except Exception as e:
            self.logger.error(f"Error building features: {str(e)}")
            self.logger.info("Continuing without additional features")
        
        # Drop rows with missing values created by lag/ma
        df = df.dropna()
        
        return df
