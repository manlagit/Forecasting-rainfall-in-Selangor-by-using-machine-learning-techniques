"""
Enhanced Data Preprocessing Pipeline
Comprehensive preprocessing with feature engineering and normalization.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, List
from pathlib import Path
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline with feature engineering.
    """
    
    def __init__(self, output_dir: str = "data/processed"):
        """
        Initialize preprocessor with output directory.
        
        Args:
            output_dir: Directory to save processed data and scalers
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Scalers for features and target
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        
        # Outlier bounds (IQR-based)
        self.outlier_bounds = {}
        
        # Feature engineering parameters
        self.lag_periods = [1]  # Previous week values
        self.ma_windows = {'precipitation': 3, 'temp': 4, 'humidity': 3}
        
    def detect_and_remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and remove outliers using IQR method.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with outliers removed
        """
        df_clean = df.copy()
        outlier_counts = {}
        
        # Define columns to check for outliers
        outlier_columns = ['Temp_avg', 'Relative_Humidity', 'Wind_kmh', 'Precipitation_mm']
        
        for column in outlier_columns:
            if column in df_clean.columns:
                Q1 = df_clean[column].quantile(0.25)
                Q3 = df_clean[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Store bounds for logging
                self.outlier_bounds[column] = (lower_bound, upper_bound)
                
                # Count outliers
                outliers = df_clean[
                    (df_clean[column] < lower_bound) | 
                    (df_clean[column] > upper_bound)
                ]
                outlier_counts[column] = len(outliers)
                
                # Remove outliers
                df_clean = df_clean[
                    (df_clean[column] >= lower_bound) & 
                    (df_clean[column] <= upper_bound)
                ]
                
                self.logger.info(f"Removed {outlier_counts[column]} outliers from {column} "
                               f"(bounds: {lower_bound:.2f} - {upper_bound:.2f})")
        
        total_removed = len(df) - len(df_clean)
        self.logger.info(f"Total records removed due to outliers: {total_removed}")
        
        return df_clean.reset_index(drop=True)
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values with mean imputation and validation.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        df_imputed = df.copy()
        
        # Log missing values before imputation
        missing_before = df_imputed.isnull().sum()
        self.logger.info(f"Missing values before imputation:\n{missing_before}")
        
        # Define columns for mean imputation
        numeric_columns = ['Temp_avg', 'Relative_Humidity', 'Wind_kmh', 'Precipitation_mm']
        
        for column in numeric_columns:
            if column in df_imputed.columns and df_imputed[column].isnull().sum() > 0:
                mean_value = df_imputed[column].mean()
                df_imputed[column].fillna(mean_value, inplace=True)
                self.logger.info(f"Imputed {column} with mean value: {mean_value:.4f}")
        
        # Log missing values after imputation
        missing_after = df_imputed.isnull().sum()
        self.logger.info(f"Missing values after imputation:\n{missing_after}")
        
        # Validation check
        if missing_after.sum() > 0:
            self.logger.warning("Some missing values remain after imputation")
        else:
            self.logger.info("All missing values successfully imputed")
        
        return df_imputed
    
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lag variables for temporal features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with lag features
        """
        df_lagged = df.copy()
        
        # Create lag features for key variables
        lag_columns = ['Precipitation_mm', 'Temp_avg', 'Relative_Humidity']
        
        for column in lag_columns:
            if column in df_lagged.columns:
                for lag in self.lag_periods:
                    lag_col_name = f"{column.lower()}_lag_{lag}"
                    df_lagged[lag_col_name] = df_lagged[column].shift(lag)
                    self.logger.info(f"Created lag feature: {lag_col_name}")
        
        return df_lagged
    
    def create_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create moving average features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with moving average features
        """
        df_ma = df.copy()
        
        # Create moving averages
        if 'Precipitation_mm' in df_ma.columns:
            window = self.ma_windows['precipitation']
            df_ma[f'precipitation_ma_{window}'] = df_ma['Precipitation_mm'].rolling(
                window=window, min_periods=1
            ).mean()
            self.logger.info(f"Created precipitation moving average (window={window})")
        
        if 'Temp_avg' in df_ma.columns:
            window = self.ma_windows['temp']
            df_ma[f'temp_ma_{window}'] = df_ma['Temp_avg'].rolling(
                window=window, min_periods=1
            ).mean()
            self.logger.info(f"Created temperature moving average (window={window})")
        
        if 'Relative_Humidity' in df_ma.columns:
            window = self.ma_windows['humidity']
            df_ma[f'humidity_ma_{window}'] = df_ma['Relative_Humidity'].rolling(
                window=window, min_periods=1
            ).mean()
            self.logger.info(f"Created humidity moving average (window={window})")
        
        return df_ma
    
    def create_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create seasonal features based on date.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with seasonal features
        """
        df_seasonal = df.copy()
        
        # Extract month from date
        df_seasonal['Month'] = df_seasonal['Date'].dt.month
        
        # Create monsoon season feature (Oct-Dec, Apr)
        df_seasonal['monsoon_season'] = (
            (df_seasonal['Month'].isin([10, 11, 12, 4]))
        ).astype(int)
        
        # Create dry season feature (Jun-Aug)
        df_seasonal['dry_season'] = (
            (df_seasonal['Month'].isin([6, 7, 8]))
        ).astype(int)
        
        # Create cyclical encoding for week of year
        week_of_year = df_seasonal['Week_Number']
        df_seasonal['week_sin'] = np.sin(2 * np.pi * week_of_year / 52)
        df_seasonal['week_cos'] = np.cos(2 * np.pi * week_of_year / 52)
        
        self.logger.info("Created seasonal features: monsoon_season, dry_season, week_sin, week_cos")
        
        return df_seasonal
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between variables.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        df_interaction = df.copy()
        
        # Temperature-humidity interaction
        if 'Temp_avg' in df_interaction.columns and 'Relative_Humidity' in df_interaction.columns:
            df_interaction['temp_humidity_interaction'] = (
                df_interaction['Temp_avg'] * df_interaction['Relative_Humidity']
            )
            self.logger.info("Created temp_humidity_interaction feature")
        
        # Wind-precipitation ratio (add 1 to avoid division by zero)
        if 'Wind_kmh' in df_interaction.columns and 'Precipitation_mm' in df_interaction.columns:
            df_interaction['wind_precipitation_ratio'] = (
                df_interaction['Wind_kmh'] / (df_interaction['Precipitation_mm'] + 1)
            )
            self.logger.info("Created wind_precipitation_ratio feature")
        
        return df_interaction
    
    def normalize_features(self, X: pd.DataFrame, y: pd.Series, 
                          fit_scalers: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Normalize features and target using MinMaxScaler.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            fit_scalers: Whether to fit scalers (True for training, False for test)
            
        Returns:
            Tuple of normalized features and target
        """
        if fit_scalers:
            # Fit and transform
            X_normalized = pd.DataFrame(
                self.feature_scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            
            y_normalized = pd.Series(
                self.target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten(),
                index=y.index
            )
            
            # Save scalers
            scaler_dir = self.output_dir / "scalers"
            scaler_dir.mkdir(exist_ok=True)
            
            joblib.dump(self.feature_scaler, scaler_dir / "feature_scaler.pkl")
            joblib.dump(self.target_scaler, scaler_dir / "target_scaler.pkl")
            
            self.logger.info("Fitted and saved feature and target scalers")
            
        else:
            # Transform only
            X_normalized = pd.DataFrame(
                self.feature_scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
            
            y_normalized = pd.Series(
                self.target_scaler.transform(y.values.reshape(-1, 1)).flatten(),
                index=y.index
            )
            
            self.logger.info("Applied existing scalers to features and target")
        
        return X_normalized, y_normalized
    
    def split_data_time_aware(self, X: pd.DataFrame, y: pd.Series, 
                             test_size: float = 0.2) -> Tuple:
        """
        Split data in time-series aware manner maintaining chronological order.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Proportion of data for testing
            
        Returns:
            Tuple of train-test splits
        """
        # Calculate split index
        split_index = int(len(X) * (1 - test_size))
        
        # Split maintaining temporal order
        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]
        
        self.logger.info(f"Data split - Train: {len(X_train)}, Test: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Main preprocessing pipeline that orchestrates all steps.
        
        Args:
            df: Raw input DataFrame
            
        Returns:
            Tuple of (normalized_features, normalized_target, processed_dataframe)
        """
        self.logger.info("Starting comprehensive data preprocessing...")
        
        # Step 1: Handle missing values
        self.logger.info("Step 1: Handling missing values...")
        df_clean = self.handle_missing_values(df)
        
        # Step 2: Remove outliers
        self.logger.info("Step 2: Removing outliers...")
        df_no_outliers = self.detect_and_remove_outliers(df_clean)
        
        # Step 3: Feature engineering
        self.logger.info("Step 3: Creating engineered features...")
        
        # Create lag features
        df_lagged = self.create_lag_features(df_no_outliers)
        
        # Create moving averages
        df_ma = self.create_moving_averages(df_lagged)
        
        # Create seasonal features
        df_seasonal = self.create_seasonal_features(df_ma)
        
        # Create interaction features
        df_processed = self.create_interaction_features(df_seasonal)
        
        # Step 4: Prepare features and target
        self.logger.info("Step 4: Preparing features and target...")
        
        # Define feature columns (exclude Date, target, and Year)
        feature_columns = [col for col in df_processed.columns 
                          if col not in ['Date', 'Precipitation_mm', 'Year']]
        
        # Remove rows with NaN values (due to lag features)
        df_processed = df_processed.dropna().reset_index(drop=True)
        
        # Extract features and target
        X = df_processed[feature_columns]
        y = df_processed['Precipitation_mm']
        
        self.logger.info(f"Feature matrix shape: {X.shape}")
        self.logger.info(f"Target vector length: {len(y)}")
        self.logger.info(f"Features: {list(X.columns)}")
        
        # Step 5: Normalize data
        self.logger.info("Step 5: Normalizing features and target...")
        X_normalized, y_normalized = self.normalize_features(X, y, fit_scalers=True)
        
        # Save processed data
        self.save_processed_data(df_processed, X_normalized, y_normalized)
        
        self.logger.info("Data preprocessing completed successfully!")
        
        return X_normalized, y_normalized, df_processed
    
    def save_processed_data(self, df_processed: pd.DataFrame, 
                           X_normalized: pd.DataFrame, y_normalized: pd.Series) -> None:
        """Save processed data to files."""
        # Save full processed data
        df_processed.to_csv(self.output_dir / "processed_data.csv", index=False)
        
        # Save normalized features and target
        X_normalized.to_csv(self.output_dir / "X_normalized.csv", index=False)
        y_normalized.to_csv(self.output_dir / "y_normalized.csv", index=False)
        
        # Save feature info
        import json
        feature_info = {
            'feature_columns': list(X_normalized.columns),
            'target_column': 'Precipitation_mm',
            'preprocessing_params': {
                'lag_periods': self.lag_periods,
                'ma_windows': self.ma_windows,
                'outlier_bounds': self.outlier_bounds
            }
        }
        
        with open(self.output_dir / "feature_info.json", 'w') as f:
            json.dump(feature_info, f, indent=2, default=str)
        
        self.logger.info(f"Saved processed data to {self.output_dir}")
