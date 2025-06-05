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
            # Lag features
            for feature_config in self.config.get('lag_features', []):
                column = feature_config['column']
                for lag in feature_config['lags']:
                    df[f'{column}_lag_{lag}'] = df[column].shift(lag)
            
            # Moving averages
            for feature_config in self.config.get('moving_averages', []):
                column = feature_config['column']
                for window in feature_config['windows']:
                    df[f'{column}_ma_{window}'] = df[column].rolling(window=window).mean()
            
            # Seasonal indicators (e.g., monsoon)
            seasonal_config = self.config.get('seasonal_features', {})
            if 'monsoon_months' in seasonal_config:
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df['is_monsoon'] = df['Date'].dt.month.isin(seasonal_config['monsoon_months'])
        except Exception as e:
            self.logger.error(f"Error building features: {str(e)}")
            self.logger.info("Continuing without additional features")
        
        # Drop rows with missing values created by lag/ma
        df = df.dropna()
        
        return df
