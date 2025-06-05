"""
Model Training Module
Trains machine learning models for rainfall forecasting.
"""

import logging
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from src.utils.helpers import load_yaml
from pathlib import Path

class ModelTrainer:
    """
    Trains machine learning models for rainfall forecasting.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize with configuration.
        """
        self.config = load_yaml(config_path).get('model_training', {})
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.best_params = {}
        self.logger.info("Initialized ModelTrainer")
        
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        """
        Train all specified models with hyperparameter tuning.
        """
        # Train Random Forest
        if 'random_forest' in self.config:
            rf_config = self.config['random_forest']
            rf = self.train_random_forest(X_train, y_train, rf_config)
            self.models['random_forest'] = rf
        
        # Train KNN
        if 'knn' in self.config:
            knn_config = self.config['knn']
            knn = self.train_knn(X_train, y_train, knn_config)
            self.models['knn'] = knn
        
        # Train XGBoost
        if 'xgboost' in self.config:
            xgb_config = self.config['xgboost']
            xgb = self.train_xgboost(X_train, y_train, xgb_config)
            self.models['xgboost'] = xgb
        
        return self.models
    
    def train_random_forest(self, X_train, y_train, config):
        """
        Train Random Forest model with hyperparameter tuning.
        """
        self.logger.info("Training Random Forest model...")
        param_grid = config.get('param_grid', {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20]
        })
        
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        self.best_params['random_forest'] = grid_search.best_params_
        self.logger.info(f"Best RF params: {grid_search.best_params_}")
        return grid_search.best_estimator_
    
    def train_knn(self, X_train, y_train, config):
        """
        Train K-Nearest Neighbors model with hyperparameter tuning.
        """
        self.logger.info("Training KNN model...")
        param_grid = config.get('param_grid', {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        })
        
        knn = KNeighborsRegressor()
        grid_search = GridSearchCV(
            estimator=knn,
            param_grid=param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        self.best_params['knn'] = grid_search.best_params_
        self.logger.info(f"Best KNN params: {grid_search.best_params_}")
        return grid_search.best_estimator_
    
    def train_xgboost(self, X_train, y_train, config):
        """
        Train XGBoost model with hyperparameter tuning.
        """
        self.logger.info("Training XGBoost model...")
        param_grid = config.get('param_grid', {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        })
        
        xgb = XGBRegressor(random_state=42)
        grid_search = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        self.best_params['xgboost'] = grid_search.best_params_
        self.logger.info(f"Best XGBoost params: {grid_search.best_params_}")
        return grid_search.best_estimator_
    
    def save_models(self, model_dir: str = "models/saved_models"):
        """
        Save trained models to disk.
        """
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        for model_name, model in self.models.items():
            model_path = Path(model_dir) / f"{model_name}_model.pkl"
            with open(model_path, 'wb') as f:
                joblib.dump(model, f)
            self.logger.info(f"Saved {model_name} model to {model_path}")
