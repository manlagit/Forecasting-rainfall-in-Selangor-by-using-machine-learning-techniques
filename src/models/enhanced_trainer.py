"""
Enhanced Model Trainer using Modular Model Classes
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Any, Optional
from pathlib import Path
import json

# Import modular model classes
from .ann_model import ANNModel
from .mlr_model import MLRModel
from .knn_model import KNNModel
from .rf_model import RFModel
from .xgb_model import XGBModel
from .arima_model import ARIMAModel

class EnhancedModelTrainer:
    """
    Enhanced model trainer using modular model classes.
    """
    
    def __init__(self, models_dir: str = "models", config_path: str = "config", random_state: int = 42):
        """
        Initialize enhanced model trainer.
        
        Args:
            models_dir: Directory to save trained models
            config_path: Path to configuration directory
            random_state: Random seed for reproducibility
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.random_state = random_state
        self.config_path = config_path
        
        # Set random seeds
        np.random.seed(random_state)
        
        # Initialize model classes
        self.models = {
            'ann': ANNModel(f"{config_path}/hyperparameters.yaml"),
            'mlr': MLRModel(f"{config_path}/hyperparameters.yaml"),
            'knn': KNNModel(f"{config_path}/hyperparameters.yaml"),
            'rf': RFModel(f"{config_path}/hyperparameters.yaml"),
            'xgb': XGBModel(f"{config_path}/hyperparameters.yaml"),
            'arima': ARIMAModel(f"{config_path}/hyperparameters.yaml")
        }
        
        # Store training results
        self.training_results = {}
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2) -> Tuple:
        """
        Split data maintaining temporal order for time series.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion for testing
            
        Returns:
            Tuple of train-test splits
        """
        split_index = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]
        
        self.logger.info(f"Data split - Train: {len(X_train)}, Test: {len(X_test)}")
        return X_train, X_test, y_train, y_test    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                        dates_train: pd.Series = None, optimize: bool = True) -> Dict[str, Any]:
        """
        Train all models using their respective classes.
        
        Args:
            X_train: Training features
            y_train: Training target
            dates_train: Training dates for ARIMA
            optimize: Whether to perform hyperparameter optimization
            
        Returns:
            Dictionary of training results
        """
        self.logger.info("Starting training of all models...")
        
        # Prepare data
        X_train_array = X_train.values
        y_train_array = y_train.values
        
        # Split training data for validation (for ANN)
        split_idx = int(0.8 * len(X_train))
        X_train_opt = X_train_array[:split_idx]
        y_train_opt = y_train_array[:split_idx]
        X_val_opt = X_train_array[split_idx:]
        y_val_opt = y_train_array[split_idx:]
        
        # Train Multiple Linear Regression
        self.logger.info("Training Multiple Linear Regression...")
        try:
            self.models['mlr'].train(X_train_array, y_train_array, perform_feature_selection=True)
            self.training_results['mlr'] = {'status': 'success', 'model': self.models['mlr']}
        except Exception as e:
            self.logger.error(f"MLR training failed: {e}")
            self.training_results['mlr'] = {'status': 'failed', 'error': str(e)}
        
        # Train K-Nearest Neighbors
        self.logger.info("Training K-Nearest Neighbors...")
        try:
            self.models['knn'].train(X_train_array, y_train_array, optimize=optimize)
            self.training_results['knn'] = {'status': 'success', 'model': self.models['knn']}
        except Exception as e:
            self.logger.error(f"KNN training failed: {e}")
            self.training_results['knn'] = {'status': 'failed', 'error': str(e)}
        
        # Train Random Forest
        self.logger.info("Training Random Forest...")
        try:
            self.models['rf'].train(X_train_array, y_train_array, optimize=optimize)
            self.training_results['rf'] = {'status': 'success', 'model': self.models['rf']}
        except Exception as e:
            self.logger.error(f"Random Forest training failed: {e}")
            self.training_results['rf'] = {'status': 'failed', 'error': str(e)}
        
        # Train XGBoost
        self.logger.info("Training XGBoost...")
        try:
            self.models['xgb'].train(X_train_array, y_train_array, optimize=optimize)
            self.training_results['xgb'] = {'status': 'success', 'model': self.models['xgb']}
        except Exception as e:
            self.logger.error(f"XGBoost training failed: {e}")
            self.training_results['xgb'] = {'status': 'failed', 'error': str(e)}
        
        # Train Artificial Neural Network
        self.logger.info("Training Artificial Neural Network...")
        try:
            self.models['ann'].train(X_train_opt, y_train_opt, X_val_opt, y_val_opt, optimize=optimize)
            self.training_results['ann'] = {'status': 'success', 'model': self.models['ann']}
        except Exception as e:
            self.logger.error(f"ANN training failed: {e}")
            self.training_results['ann'] = {'status': 'failed', 'error': str(e)}        
        # Train ARIMA
        self.logger.info("Training ARIMA...")
        try:
            if dates_train is not None:
                # Create time series with proper index
                timeseries = pd.Series(y_train_array, index=pd.to_datetime(dates_train))
                self.models['arima'].train(timeseries, optimize=optimize)
                self.training_results['arima'] = {'status': 'success', 'model': self.models['arima']}
            else:
                self.logger.warning("No dates provided for ARIMA training")
                self.training_results['arima'] = {'status': 'skipped', 'error': 'No dates provided'}
        except Exception as e:
            self.logger.error(f"ARIMA training failed: {e}")
            self.training_results['arima'] = {'status': 'failed', 'error': str(e)}
        
        # Count successful trainings
        successful_models = sum(1 for result in self.training_results.values() 
                               if result['status'] == 'success')
        self.logger.info(f"Successfully trained {successful_models} out of {len(self.models)} models")
        
        return self.training_results
    
    def predict_all_models(self, X_test: pd.DataFrame, 
                          dates_test: pd.Series = None) -> Dict[str, np.ndarray]:
        """
        Make predictions using all trained models.
        
        Args:
            X_test: Test features
            dates_test: Test dates for ARIMA
            
        Returns:
            Dictionary of predictions
        """
        predictions = {}
        X_test_array = X_test.values
        
        for model_name, result in self.training_results.items():
            if result['status'] == 'success':
                try:
                    model = result['model']
                    
                    if model_name == 'arima':
                        # ARIMA prediction requires special handling
                        if dates_test is not None:
                            pred = model.predict(steps=len(X_test))
                        else:
                            self.logger.warning(f"Cannot predict with ARIMA without dates")
                            continue
                    else:
                        # Standard prediction for other models
                        pred = model.predict(X_test_array)
                    
                    predictions[model_name] = pred
                    self.logger.info(f"Predictions generated for {model_name}")
                    
                except Exception as e:
                    self.logger.error(f"Prediction failed for {model_name}: {e}")
                    continue
        
        return predictions    
    def save_models(self) -> None:
        """Save all trained models to disk."""
        self.logger.info("Saving trained models...")
        
        saved_models_dir = self.models_dir / "saved_models"
        saved_models_dir.mkdir(exist_ok=True)
        
        for model_name, result in self.training_results.items():
            if result['status'] == 'success':
                try:
                    model = result['model']
                    
                    if model_name == 'ann':
                        # Save ANN model
                        filepath = saved_models_dir / f"{model_name}_model.h5"
                        model.save_model(str(filepath))
                    else:
                        # Save other models
                        filepath = saved_models_dir / f"{model_name}_model.pkl"
                        model.save_model(str(filepath))
                    
                    self.logger.info(f"Saved {model_name} model")
                    
                except Exception as e:
                    self.logger.error(f"Failed to save {model_name} model: {e}")
        
        # Save training results summary
        summary_file = self.models_dir / "training_summary.json"
        summary = {}
        for model_name, result in self.training_results.items():
            if result['status'] == 'success':
                model = result['model']
                try:
                    summary[model_name] = {
                        'status': result['status'],
                        'best_params': getattr(model, 'best_params', None),
                        'model_summary': model.get_model_summary() if hasattr(model, 'get_model_summary') else None
                    }
                except:
                    summary[model_name] = {'status': result['status']}
            else:
                summary[model_name] = result
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Models and summary saved to {self.models_dir}")
    
    def load_models(self) -> Dict[str, Any]:
        """Load saved models from disk."""
        self.logger.info("Loading trained models...")
        
        saved_models_dir = self.models_dir / "saved_models"
        loaded_results = {}
        
        if not saved_models_dir.exists():
            self.logger.warning("No saved models directory found")
            return {}
        
        # Load each model type
        for model_name in self.models.keys():
            try:
                if model_name == 'ann':
                    filepath = saved_models_dir / f"{model_name}_model.h5"
                else:
                    filepath = saved_models_dir / f"{model_name}_model.pkl"
                
                if filepath.exists():
                    model = self.models[model_name]
                    model.load_model(str(filepath))
                    loaded_results[model_name] = {'status': 'success', 'model': model}
                    self.logger.info(f"Loaded {model_name} model")
                else:
                    self.logger.warning(f"Model file not found for {model_name}")
                    
            except Exception as e:
                self.logger.error(f"Failed to load {model_name} model: {e}")
                loaded_results[model_name] = {'status': 'failed', 'error': str(e)}
        
        self.training_results = loaded_results
        self.logger.info(f"Loaded {len(loaded_results)} models")
        return loaded_results
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """
        Get feature importance from models that support it.
        
        Returns:
            Dictionary of feature importance scores
        """
        importance_scores = {}
        
        for model_name, result in self.training_results.items():
            if result['status'] == 'success':
                model = result['model']
                
                if hasattr(model, 'get_feature_importance'):
                    try:
                        importance = model.get_feature_importance()
                        if importance is not None:
                            importance_scores[model_name] = importance
                    except Exception as e:
                        self.logger.error(f"Failed to get feature importance for {model_name}: {e}")
        
        return importance_scores
