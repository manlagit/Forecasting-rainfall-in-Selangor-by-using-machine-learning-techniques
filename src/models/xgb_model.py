"""
XGBoost Model Implementation
for Rainfall Forecasting in Selangor
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yaml
import logging
import pickle
import os

class XGBModel:
    """
    XGBoost model for rainfall prediction
    """
    
    def __init__(self, config_path="config/hyperparameters.yaml"):
        """
        Initialize XGBoost model with configuration
        
        Args:
            config_path (str): Path to hyperparameters configuration file
        """
        self.model = None
        self.best_params = None
        self.grid_search = None
        self.feature_importance = None
        self.logger = logging.getLogger(__name__)
        
        # Load hyperparameters from config
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            self.hyperparams = config['xgboost']
    
    def optimize_hyperparameters(self, X_train, y_train, cv=5, scoring='neg_mean_squared_error'):
        """
        Optimize hyperparameters using GridSearchCV
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training target
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric
            
        Returns:
            dict: Best hyperparameters
        """
        
        # Create parameter grid
        param_grid = {
            'n_estimators': self.hyperparams['n_estimators'],
            'learning_rate': self.hyperparams['learning_rate'],
            'max_depth': self.hyperparams['max_depth'],
            'subsample': self.hyperparams['subsample'],
            'colsample_bytree': self.hyperparams['colsample_bytree'],
            'reg_alpha': self.hyperparams['reg_alpha'],
            'reg_lambda': self.hyperparams['reg_lambda']
        }
        
        # Initialize XGBoost regressor
        xgb_reg = xgb.XGBRegressor(
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        
        # Perform grid search
        self.grid_search = GridSearchCV(
            estimator=xgb_reg,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        self.grid_search.fit(X_train, y_train)
        self.best_params = self.grid_search.best_params_
        
        self.logger.info(f"Best XGBoost hyperparameters: {self.best_params}")
        return self.best_params    
    def train(self, X_train, y_train, optimize=True):
        """
        Train the XGBoost model
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training target
            optimize (bool): Whether to optimize hyperparameters
            
        Returns:
            XGBRegressor: Trained model
        """
        
        if optimize:
            # Optimize hyperparameters
            self.optimize_hyperparameters(X_train, y_train)
            
            # Train final model with best parameters
            self.model = xgb.XGBRegressor(
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                **self.best_params
            )
        else:
            # Use default parameters
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0,
                reg_lambda=1,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
        
        self.model.fit(X_train, y_train)
        
        # Get feature importance
        self.feature_importance = self.model.feature_importances_
        
        self.logger.info("XGBoost model training completed")
        return self.model
    
    def predict(self, X_test):
        """
        Make predictions using trained model
        
        Args:
            X_test (np.array): Test features
            
        Returns:
            np.array: Predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(X_test)
        return predictions    
    def save_model(self, filepath):
        """
        Save trained model
        
        Args:
            filepath (str): Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model and related components
        model_data = {
            'model': self.model,
            'best_params': self.best_params,
            'feature_importance': self.feature_importance,
            'grid_search_results': self.grid_search.cv_results_ if self.grid_search else None
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"XGBoost model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load trained model
        
        Args:
            filepath (str): Path to load model from
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.best_params = model_data.get('best_params')
        self.feature_importance = model_data.get('feature_importance')
        
        self.logger.info(f"XGBoost model loaded from {filepath}")
    
    def get_feature_importance(self):
        """
        Get feature importance scores
        
        Returns:
            np.array: Feature importance scores
        """
        return self.feature_importance
    
    def get_model_summary(self):
        """
        Get model summary
        
        Returns:
            dict: Model summary
        """
        if self.model is None:
            return "No model available"
        
        summary = {
            'model_type': 'XGBRegressor',
            'n_estimators': self.model.n_estimators,
            'learning_rate': self.model.learning_rate,
            'max_depth': self.model.max_depth,
            'subsample': self.model.subsample,
            'colsample_bytree': self.model.colsample_bytree,
            'reg_alpha': self.model.reg_alpha,
            'reg_lambda': self.model.reg_lambda,
            'feature_importance_mean': np.mean(self.feature_importance) if self.feature_importance is not None else None,
            'best_params': self.best_params
        }
        
        return summary
