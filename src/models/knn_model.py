"""
K-Nearest Neighbors (KNN) Model Implementation
for Rainfall Forecasting in Selangor
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import yaml
import logging
import pickle
import os

class KNNModel:
    """
    K-Nearest Neighbors model for rainfall prediction
    """
    
    def __init__(self, config_path="config/hyperparameters.yaml"):
        """
        Initialize KNN model with configuration
        
        Args:
            config_path (str): Path to hyperparameters configuration file
        """
        self.model = None
        self.best_params = None
        self.grid_search = None
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)
        
        # Load hyperparameters from config
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            self.hyperparams = config['knn']
    
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
            'n_neighbors': self.hyperparams['n_neighbors'],
            'weights': self.hyperparams['weights'],
            'metric': self.hyperparams['metric']
        }
        
        # Add p parameter only for minkowski metric
        expanded_grid = []
        for params in self._expand_grid(param_grid):
            if params['metric'] == 'minkowski':
                for p in self.hyperparams['p']:
                    expanded_params = params.copy()
                    expanded_params['p'] = p
                    expanded_grid.append(expanded_params)
            else:
                expanded_grid.append(params)
        
        # Initialize KNN regressor
        knn = KNeighborsRegressor()
        
        # Perform grid search
        self.grid_search = GridSearchCV(
            estimator=knn,
            param_grid=expanded_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        self.grid_search.fit(X_train, y_train)
        self.best_params = self.grid_search.best_params_
        
        self.logger.info(f"Best KNN hyperparameters: {self.best_params}")
        return self.best_params    
    def _expand_grid(self, param_grid):
        """
        Helper function to expand parameter grid
        
        Args:
            param_grid (dict): Parameter grid
            
        Returns:
            list: Expanded parameter combinations
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        combinations = []
        self._generate_combinations(keys, values, 0, {}, combinations)
        
        return combinations
    
    def _generate_combinations(self, keys, values, index, current, combinations):
        """
        Recursively generate parameter combinations
        """
        if index == len(keys):
            combinations.append(current.copy())
            return
        
        for value in values[index]:
            current[keys[index]] = value
            self._generate_combinations(keys, values, index + 1, current, combinations)
    
    def train(self, X_train, y_train, optimize=True, scale_features=True):
        """
        Train the KNN model
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training target
            optimize (bool): Whether to optimize hyperparameters
            scale_features (bool): Whether to scale features
            
        Returns:
            KNeighborsRegressor: Trained model
        """
        
        # Scale features if requested
        if scale_features:
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            X_train_scaled = X_train
        
        if optimize:
            # Optimize hyperparameters
            self.optimize_hyperparameters(X_train_scaled, y_train)
            
            # Train final model with best parameters
            self.model = KNeighborsRegressor(**self.best_params)
        else:
            # Use default parameters
            self.model = KNeighborsRegressor(
                n_neighbors=5,
                weights='uniform',
                metric='euclidean'
            )
        
        self.model.fit(X_train_scaled, y_train)
        
        self.logger.info("KNN model training completed")
        return self.model        Save trained model
        
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
            'scaler': self.scaler,
            'grid_search_results': self.grid_search.cv_results_ if self.grid_search else None
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"KNN model saved to {filepath}")
    
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
        self.scaler = model_data.get('scaler', StandardScaler())
        
        self.logger.info(f"KNN model loaded from {filepath}")
    
    def get_model_summary(self):
        """
        Get model summary
        
        Returns:
            dict: Model summary
        """
        if self.model is None:
            return "No model available"
        
        summary = {
            'model_type': 'KNeighborsRegressor',
            'n_neighbors': self.model.n_neighbors,
            'weights': self.model.weights,
            'metric': self.model.metric,
            'best_params': self.best_params
        }
        
        if hasattr(self.model, 'p'):
            summary['p'] = self.model.p
        
        return summary
