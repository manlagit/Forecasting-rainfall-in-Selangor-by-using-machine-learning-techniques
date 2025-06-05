"""
Enhanced Data Preprocessing Pipeline
Handles data cleaning, outlier handling, and feature scaling.
"""

import pandas as pd
import logging
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler
from src.utils.helpers import load_yaml
from typing import Tuple


class DataPreprocessor:
    """
    Data preprocessing focused on cleaning and scaling.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize preprocessor with configuration.
        """
        try:
            self.config = load_yaml(config_path).get('preprocessing', {})
            self.logger = logging.getLogger(__name__)
            
            # Initialize scalers
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()
            
            self.logger.info("Initialized DataPreprocessor")
        except Exception as e:
            logging.error(f"Error initializing DataPreprocessor: {str(e)}")
            raise

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values with mean for numeric columns.
        """
        numeric_cols = [
            'Temp_avg', 'Relative_Humidity', 
            'Wind_kmh', 'Precipitation_mm'
        ]
        for col in numeric_cols:
            if col in df.columns and df[col].isnull().sum() > 0:
                mean_val = df[col].mean()
                df[col].fillna(mean_val, inplace=True)
                self.logger.info(f"Imputed {col} with mean: {mean_val:.2f}")
        return df

    def cap_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cap outliers to the predefined ranges in the config.
        """
        # Get the valid ranges from config
        valid_ranges = self.config.get('valid_ranges', {})
        if not valid_ranges:
            self.logger.warning(
                "No valid ranges found in config for capping outliers"
            )
            return df

        # Map config keys to dataframe columns
        range_mapping = {
            'temperature': 'Temp_avg',
            'humidity': 'Relative_Humidity',
            'wind': 'Wind_kmh',
            'precipitation': 'Precipitation_mm'
        }

        for config_key, col in range_mapping.items():
            if config_key in valid_ranges and col in df.columns:
                min_val, max_val = valid_ranges[config_key]
                # Cap the values
                df[col] = df[col].clip(lower=min_val, upper=max_val)
                self.logger.info(
                    f"Capped {col} to [{min_val}, {max_val}]"
                )

        return df

    def preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess the data: handle missing values, 
        cap outliers, and separate features and target.
        Returns features and target.
        """
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Cap outliers
        df = self.cap_outliers(df)
        
        # Create lag features
        lags = [1, 2, 3]  # 1, 2, and 3 weeks lag
        for lag in lags:
            df[f'precipitation_lag_{lag}'] = df['Precipitation_mm'].shift(lag)
            df[f'temp_lag_{lag}'] = df['Temp_avg'].shift(lag)
            df[f'humidity_lag_{lag}'] = df['Relative_Humidity'].shift(lag)
        
        # Create moving average features
        moving_average_windows = [3, 4]  # 3 and 4 weeks moving average
        for window in moving_average_windows:
            df[f'precipitation_ma_{window}'] = (
                df['Precipitation_mm'].rolling(window=window).mean()
            )
            df[f'temp_ma_{window}'] = (
                df['Temp_avg'].rolling(window=window).mean()
            )
            df[f'humidity_ma_{window}'] = (
                df['Relative_Humidity'].rolling(window=window).mean()
            )
        
        # Drop rows with missing values created by lag/ma features
        df = df.dropna()
        
        # Separate features and target
        X = df.drop(columns=['Precipitation_mm'])
        y = df['Precipitation_mm']
        
        return X, y

    def fit_scalers(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit the scalers to the training data.
        Only numeric features are scaled. Non-numeric features are ignored.
        """
        # Identify numeric columns
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        self.numeric_cols_ = numeric_cols  # store for transform
        
        # Fit the scaler only on numeric features
        self.feature_scaler.fit(X[numeric_cols])
        self.target_scaler.fit(y.values.reshape(-1, 1))
        self.logger.info("Fitted scalers to training data")

    def transform(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply scaling to the given features and target.
        Only numeric features are scaled. Non-numeric features are left unchanged.
        """
        # Separate numeric and non-numeric features
        X_numeric = X[self.numeric_cols_]
        X_non_numeric = X.drop(columns=self.numeric_cols_)
        
        # Scale numeric features
        X_scaled_numeric = self.feature_scaler.transform(X_numeric)
        # Create a DataFrame for scaled numeric features
        X_scaled_numeric_df = pd.DataFrame(
            X_scaled_numeric, 
            columns=self.numeric_cols_, 
            index=X.index
        )
        # Combine with non-numeric features
        X_scaled = pd.concat([X_non_numeric, X_scaled_numeric_df], axis=1)
        
        # Scale the target
        y_scaled = self.target_scaler.transform(
            y.values.reshape(-1, 1)
        ).flatten()
        y_scaled = pd.Series(
            y_scaled, 
            name=y.name
        )
        
        return X_scaled, y_scaled

    def save_scalers(self, scaler_dir: str) -> None:
        """
        Save the scalers to disk.
        """
        Path(scaler_dir).mkdir(parents=True, exist_ok=True)
        # Save feature scaler
        feature_path = Path(scaler_dir) / "feature_scaler.pkl"
        with open(feature_path, 'wb') as f:
            joblib.dump(
                self.feature_scaler,
                f
            )
        
        # Save target scaler  
        target_path = Path(scaler_dir) / "target_scaler.pkl"
        with open(target_path, 'wb') as f:
            joblib.dump(
                self.target_scaler,
                f
            )
        self.logger.info("Saved scalers to directory")
        self.logger.debug(f"Directory path: {scaler_dir}")
