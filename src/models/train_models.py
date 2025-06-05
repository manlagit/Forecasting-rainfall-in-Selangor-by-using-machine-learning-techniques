from pathlib import Path
import pickle
import yaml
from typing import Dict, Any
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import pandas as pd
from src.utils.helpers import load_yaml
from .arima_model import ARIMAModel
from .ann_model import ANNModel
from .knn_model import KNNModel
from .rf_model import RFModel
from .xgb_model import XGBModel
from .mlr_model import MLRModel


class ModelTrainer:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize model trainer with configuration.
        
        Args:
            config_path: Path to config file with training parameters
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.scorers = {
            'neg_mse': make_scorer(
                mean_squared_error,
                greater_is_better=False),
            'neg_mae': make_scorer(
                mean_absolute_error,
                greater_is_better=False),
            'r2': make_scorer(r2_score)
        }

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate training configuration.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Dictionary with training parameters
        """
        try:
            config = load_yaml(config_path)
            training_config = config.get('training', {})
            
            # Validate required fields exist
            required = ['cv_folds', 'scoring', 'n_jobs']
            if not all(k in training_config for k in required):
                raise ValueError("Missing required training config fields")
                
            return training_config
        except Exception:
            logging.error("Config load error")
            raise

    def _setup_logging(self):
        """Configure logging for training operations."""
        logging.basicConfig(
            filename='logs/training.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def train_model(self, model_name: str, X: pd.DataFrame, y: pd.Series) -> Any:
        """Train a specified model with hyperparameter tuning.
        
        Args:
            model_name: Name of model to train
            X: Features dataframe
            y: Target series
            
        Returns:
            Trained model instance
        """
        try:
            logging.info(f"Starting training for {model_name}")
            
            if model_name == 'arima':
                model = ARIMAModel()
                model.train(y)  # ARIMA uses time series directly
            else:
                # Get model class and parameters
                model_class, params = self._get_model_config(model_name)
                
                # Initialize model
                model = model_class()
                
                # Hyperparameter tuning
                if params:
                    # Assign parameters to variables for better readability
                    estimator = model
                    param_grid = params
                    cv = self.config['cv_folds']
                    scoring = self.config['scoring']
                    n_jobs = self.config['n_jobs']
                    verbose = 2
                    
                    grid_search = GridSearchCV(
                        estimator=estimator,
                        param_grid=param_grid,
                        cv=cv,
                        scoring=scoring,
                        n_jobs=n_jobs,
                        verbose=verbose
                    )
                    grid_search.fit(X, y)
                    model = grid_search.best_estimator_
                    logging.info(f"Best params found for {model_name}")
                    logging.debug(f"Params: {grid_search.best_params_}")
                else:
                    model.fit(X, y)
            
            logging.info(f"Completed training for {model_name}")
            return model
        except Exception as e:
            logging.error(f"Training failed: {model_name}")
            logging.debug(str(e)[:50])
            raise

    def _get_model_config(self, model_name: str) -> tuple:
        """Get model class and hyperparameters from config.
        
        Args:
            model_name: Name of model to get configuration for
            
        Returns:
            Tuple of (model_class, hyperparameters)
        """
        try:
            with open('config/hyperparameters.yaml') as f:
                hyperparams = yaml.safe_load(f)
                
            model_classes = {
                'ann': ANNModel,
                'knn': KNNModel,
                'rf': RFModel,
                'xgb': XGBModel,
                'mlr': MLRModel
            }
            
            return model_classes[model_name], hyperparams.get(model_name, None)
        except Exception:
            logging.error("Model config error")
            raise

    def save_model(self, model: Any, model_name: str) -> None:
        """Save trained model to file.
        
        Args:
            model: Trained model instance
            model_name: Name of model being saved
        """
        try:
            save_dir = Path("models/saved_models")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            save_path = save_dir / f"{model_name}_model.pkl"
            
            if hasattr(model, 'save'):
                model.save(str(save_path))
            else:
                with open(save_path, 'wb') as f:
                    pickle.dump(model, f)
            
            logging.info(f"Saved {model_name} model to {save_path}")
        except Exception as e:
            logging.error(f"Save failed: {model_name}")
            logging.debug(str(e)[:60])
            raise
