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
        self.config = load_yaml(config_path).get('models', {})
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.best_params = {}
        self.logger.info("Initialized ModelTrainer")
        
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        """
        Train all specified models with hyperparameter tuning.
        """
        # Map config keys to model names (only include implemented models)
        model_mapping = {
            'rf': ('random_forest', self.train_random_forest),
            'xgb': ('xgboost', self.train_xgboost),
            'knn': ('knn', self.train_knn)
        }
        
        for config_key, (model_name, train_func) in model_mapping.items():
            if config_key in self.config:
                config = self.config[config_key]
                model = train_func(X_train, y_train, config)
                self.models[model_name] = model
        
        return self.models
    
    def train_random_forest(self, X_train, y_train, config):
        """
        Train Random Forest model with Optuna hyperparameter optimization.
        """
        self.logger.info("Training Random Forest model with Optuna...")
        import optuna
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestRegressor
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 10, 50, step=10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': 42
            }
            model = RandomForestRegressor(**params)
            score = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error').mean()
            return score
        
        study = optuna.create_study(direction='maximize')
        try:
            study.optimize(objective, n_trials=config.get('n_trials', 50))
        except KeyboardInterrupt:
            self.logger.info("Optimization was interrupted. Using best model from current trials.")
        
        best_params = study.best_params
        self.best_params['random_forest'] = best_params
        self.logger.info(f"Best RF params: {best_params}")
        
        # Train the model with best parameters on the entire training set
        best_rf = RandomForestRegressor(**best_params)
        best_rf.fit(X_train, y_train)
        return best_rf
    
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
