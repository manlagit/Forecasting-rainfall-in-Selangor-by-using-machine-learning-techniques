"""
Model training module for rainfall forecasting.
Implements various ML models with hyperparameter tuning.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras import layers
import optuna
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import joblib
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training and hyperparameter tuning."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize trainer with configuration."""
        self.config = self._load_config(config_path)
        self.model_config = self.config['models']
        self.models = {}
        self.best_params = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """
        Split data into train and test sets maintaining chronological order.
        """
        # Calculate split index
        split_idx = int(len(X) * (1 - self.model_config['test_size']))
        
        # Time series aware split
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_linear_regression(self, X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
        """Train Multiple Linear Regression model."""
        logger.info("Training Linear Regression model...")
        
        # Feature selection using RFE
        estimator = LinearRegression()
        selector = RFE(estimator, n_features_to_select=10, step=1)
        selector.fit(X_train, y_train)
        
        # Train final model with selected features
        model = LinearRegression()
        model.fit(X_train.iloc[:, selector.support_], y_train)
        
        self.models['linear_regression'] = model
        self.best_params['linear_regression'] = {
            'selected_features': X_train.columns[selector.support_].tolist()
        }
        
        return model
    
    def train_knn(self, X_train: pd.DataFrame, y_train: pd.Series) -> KNeighborsRegressor:
        """Train KNN model with hyperparameter tuning."""
        logger.info("Training KNN model...")
        
        param_grid = {
            'n_neighbors': self.model_config['knn']['n_neighbors'],
            'weights': self.model_config['knn']['weights'],
            'metric': self.model_config['knn']['metric']
        }
        
        knn = KNeighborsRegressor()
        grid_search = GridSearchCV(
            knn, param_grid, 
            cv=self.model_config['cv_folds'],
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.models['knn'] = grid_search.best_estimator_
        self.best_params['knn'] = grid_search.best_params_
        
        logger.info(f"Best KNN params: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
