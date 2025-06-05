"""
Model Implementation & Hyperparameter Tuning Module
Comprehensive machine learning models with automated optimization.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Any, Optional
from pathlib import Path
import joblib
import json
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

# Deep learning imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import optuna

class ModelTrainer:
    """
    Comprehensive model trainer with hyperparameter optimization.
    """
    
    def __init__(self, models_dir: str = "models", random_state: int = 42):
        """
        Initialize model trainer.
        
        Args:
            models_dir: Directory to save trained models
            random_state: Random seed for reproducibility
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.random_state = random_state
        
        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        # Store trained models and best parameters
        self.trained_models = {}
        self.best_params = {}
        
        # Define hyperparameter grids
        self._define_hyperparameter_grids()
    
    def _define_hyperparameter_grids(self):
        """Define hyperparameter grids for all models."""
        
        # KNN hyperparameters
        self.knn_param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'p': [1, 2]
        }
        
        # Random Forest hyperparameters
        self.rf_param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 20, 30, 50, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        }
        
        # XGBoost hyperparameters
        self.xgb_param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 6, 9],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [1, 1.5, 2]
        }
        
        # ANN hyperparameters for Optuna
        self.ann_param_ranges = {
            'n_layers': (2, 4),
            'n_neurons': (32, 128),
            'activation': ['relu', 'tanh', 'sigmoid'],
            'learning_rate': (0.001, 0.1),
            'batch_size': [16, 32, 64],
            'epochs': [50, 100, 200],
            'dropout_rate': (0.1, 0.3)
        }
    
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
    
    def train_linear_regression(self, X_train: pd.DataFrame, 
                               y_train: pd.Series) -> LinearRegression:
        """
        Train Multiple Linear Regression with feature selection.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained model
        """
        self.logger.info("Training Multiple Linear Regression...")
        
        # Exclude non-numeric columns
        X_train = X_train.select_dtypes(include=['number']).copy()
        
        # Convert remaining non-numeric features to numeric
        X_train = X_train.apply(pd.to_numeric, errors='coerce')
        y_train = pd.to_numeric(y_train, errors='coerce')
        
        # Handle missing values
        if X_train.isnull().sum().sum() > 0:
            imputer = SimpleImputer(strategy='mean')
            X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
            self.logger.info("Imputed missing values in features")
        
        if y_train.isnull().sum() > 0:
            y_train = y_train.fillna(y_train.mean())
            self.logger.info("Imputed missing values in target")
        
        # Feature selection using RFE
        lr_base = LinearRegression()
        rfe = RFE(estimator=lr_base, n_features_to_select=10)
        rfe.fit(X_train, y_train)
        
        # Get selected features
        selected_features = X_train.columns[rfe.support_].tolist()
        self.logger.info(f"Selected features: {selected_features}")
        
        # Train final model with selected features
        model = LinearRegression()
        model.fit(X_train[selected_features], y_train)
        
        # Store selected features for prediction
        self.best_params['linear_regression'] = {
            'selected_features': selected_features
        }
        
        self.logger.info("Linear Regression training completed")
        return model
    
    def train_knn(self, X_train: pd.DataFrame, y_train: pd.Series) -> KNeighborsRegressor:
        """
        Train K-Nearest Neighbors with hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained model
        """
        self.logger.info("Training K-Nearest Neighbors with hyperparameter tuning...")
        
        # Exclude non-numeric columns
        X_train = X_train.select_dtypes(include=['number']).copy()
        
        # Convert remaining non-numeric features to numeric
        X_train = X_train.apply(pd.to_numeric, errors='coerce')
        y_train = pd.to_numeric(y_train, errors='coerce')
        
        # Handle missing values
        if X_train.isnull().sum().sum() > 0:
            imputer = SimpleImputer(strategy='mean')
            X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
            self.logger.info("Imputed missing values in features")
        
        if y_train.isnull().sum() > 0:
            y_train = y_train.fillna(y_train.mean())
            self.logger.info("Imputed missing values in target")
        
        knn = KNeighborsRegressor()
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            knn, 
            self.knn_param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Store best parameters
        self.best_params['knn'] = grid_search.best_params_
        self.logger.info(f"Best KNN parameters: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    def train_random_forest(self, X_train: pd.DataFrame, 
                           y_train: pd.Series) -> RandomForestRegressor:
        """
        Train Random Forest with hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained model
        """
        self.logger.info("Training Random Forest with hyperparameter tuning...")
        
        # Exclude non-numeric columns
        X_train = X_train.select_dtypes(include=['number']).copy()
        
        # Convert remaining non-numeric features to numeric
        X_train = X_train.apply(pd.to_numeric, errors='coerce')
        y_train = pd.to_numeric(y_train, errors='coerce')
        
        # Handle missing values
        if X_train.isnull().sum().sum() > 0:
            imputer = SimpleImputer(strategy='mean')
            X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
            self.logger.info("Imputed missing values in features")
        
        if y_train.isnull().sum() > 0:
            y_train = y_train.fillna(y_train.mean())
            self.logger.info("Imputed missing values in target")
        
        rf = RandomForestRegressor(random_state=self.random_state)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            rf,
            self.rf_param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Store best parameters
        self.best_params['random_forest'] = grid_search.best_params_
        self.logger.info(f"Best RF parameters: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    def train_xgboost(self, X_train: pd.DataFrame, 
                     y_train: pd.Series) -> xgb.XGBRegressor:
        """
        Train XGBoost with hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained model
        """
        self.logger.info("Training XGBoost with hyperparameter tuning...")
        
        # Exclude non-numeric columns
        X_train = X_train.select_dtypes(include=['number']).copy()
        
        # Convert remaining non-numeric features to numeric
        X_train = X_train.apply(pd.to_numeric, errors='coerce')
        y_train = pd.to_numeric(y_train, errors='coerce')
        
        # Handle missing values
        if X_train.isnull().sum().sum() > 0:
            imputer = SimpleImputer(strategy='mean')
            X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
            self.logger.info("Imputed missing values in features")
        
        if y_train.isnull().sum() > 0:
            y_train = y_train.fillna(y_train.mean())
            self.logger.info("Imputed missing values in target")
        
        xgb_model = xgb.XGBRegressor(random_state=self.random_state)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            xgb_model,
            self.xgb_param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Store best parameters
        self.best_params['xgboost'] = grid_search.best_params_
        self.logger.info(f"Best XGBoost parameters: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    def create_ann_model(self, trial: optuna.Trial, input_dim: int) -> Sequential:
        """
        Create ANN model with Optuna hyperparameter optimization.
        
        Args:
            trial: Optuna trial object
            input_dim: Number of input features
            
        Returns:
            Keras Sequential model
        """
        n_layers = trial.suggest_int('n_layers', *self.ann_param_ranges['n_layers'])
        
        model = Sequential()
        
        # First layer
        n_neurons = trial.suggest_int('n_neurons_1', *self.ann_param_ranges['n_neurons'])
        activation = trial.suggest_categorical('activation', self.ann_param_ranges['activation'])
        dropout_rate = trial.suggest_float('dropout_rate', *self.ann_param_ranges['dropout_rate'])
        
        model.add(Dense(n_neurons, activation=activation, input_dim=input_dim))
        model.add(Dropout(dropout_rate))
        
        # Additional layers
        for i in range(2, n_layers + 1):
            n_neurons = trial.suggest_int(f'n_neurons_{i}', *self.ann_param_ranges['n_neurons'])
            model.add(Dense(n_neurons, activation=activation))
            model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(1, activation='linear'))
        
        # Compile model
        learning_rate = trial.suggest_float('learning_rate', *self.ann_param_ranges['learning_rate'], log=True)
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def ann_objective(self, trial: optuna.Trial, X_train: pd.DataFrame, 
                     y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> float:
        """
        Objective function for ANN hyperparameter optimization.
        
        Args:
            trial: Optuna trial
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Validation loss
        """
        model = self.create_ann_model(trial, X_train.shape[1])
        
        batch_size = trial.suggest_categorical('batch_size', self.ann_param_ranges['batch_size'])
        epochs = trial.suggest_categorical('epochs', self.ann_param_ranges['epochs'])
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ]
        )
        
        # Return validation loss
        return min(history.history['val_loss'])
    
    def train_ann(self, X_train: pd.DataFrame, y_train: pd.Series) -> Sequential:
        """
        Train Artificial Neural Network with Optuna optimization.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained model
        """
        self.logger.info("Training Artificial Neural Network with Optuna optimization...")
        
        # Exclude non-numeric columns
        X_train = X_train.select_dtypes(include=['number']).copy()
        
        # Convert remaining non-numeric features to numeric
        X_train = X_train.apply(pd.to_numeric, errors='coerce')
        y_train = pd.to_numeric(y_train, errors='coerce')
        
        # Handle missing values
        if X_train.isnull().sum().sum() > 0:
            imputer = SimpleImputer(strategy='mean')
            X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
            self.logger.info("Imputed missing values in features")
        
        if y_train.isnull().sum() > 0:
            y_train = y_train.fillna(y_train.mean())
            self.logger.info("Imputed missing values in target")
        
        # Split training data for validation
        split_idx = int(0.8 * len(X_train))
        X_train_opt = X_train.iloc[:split_idx]
        y_train_opt = y_train.iloc[:split_idx]
        X_val_opt = X_train.iloc[split_idx:]
        y_val_opt = y_train.iloc[split_idx:]
        
        # Create Optuna study
        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda trial: self.ann_objective(trial, X_train_opt, y_train_opt, X_val_opt, y_val_opt),
            n_trials=50
        )
        
        # Train final model with best parameters
        best_params = study.best_params
        self.best_params['ann'] = best_params
        self.logger.info(f"Best ANN parameters: {best_params}")
        
        # Create and train final model
        model = self.create_ann_model(study.best_trial, X_train.shape[1])
        
        model.fit(
            X_train, y_train,
            batch_size=best_params['batch_size'],
            epochs=best_params['epochs'],
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)
            ]
        )
        
        return model
    
    def train_arima(self, dates: pd.Series, values: pd.Series) -> ARIMA:
        """
        Train ARIMA model with automatic parameter selection.
        
        Args:
            dates: Date series
            values: Target values series
            
        Returns:
            Trained ARIMA model
        """
        self.logger.info("Training ARIMA model...")
        
        # Convert and handle missing values
        values = pd.to_numeric(values, errors='coerce')
        if values.isnull().sum() > 0:
            values = values.fillna(values.mean())
            self.logger.info("Imputed missing values in target")
        
        # Create time series
        ts = pd.Series(values.values, index=pd.to_datetime(dates))
        
        # Test for stationarity
        adf_result = adfuller(ts.dropna())
        self.logger.info(f"ADF Statistic: {adf_result[0]:.4f}")
        self.logger.info(f"p-value: {adf_result[1]:.4f}")
        
        # Determine differencing order
        d = 0
        if adf_result[1] > 0.05:
            d = 1
            ts_diff = ts.diff().dropna()
            adf_result_diff = adfuller(ts_diff)
            self.logger.info(f"After differencing - ADF p-value: {adf_result_diff[1]:.4f}")
        
        # Grid search for best ARIMA parameters
        best_aic = float('inf')
        best_params = (0, 0, 0)
        
        p_values = range(0, 4)
        q_values = range(0, 4)
        
        for p in p_values:
            for q in q_values:
                try:
                    model = ARIMA(ts, order=(p, d, q))
                    fitted_model = model.fit()
                    
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_params = (p, d, q)
                        
                except:
                    continue
        
        # Train final model with best parameters
        self.logger.info(f"Best ARIMA parameters: {best_params} (AIC: {best_aic:.4f})")
        final_model = ARIMA(ts, order=best_params)
        fitted_final = final_model.fit()
        
        self.best_params['arima'] = {
            'order': best_params,
            'aic': best_aic
        }
        
        return fitted_final
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                        dates_train: pd.Series) -> Dict[str, Any]:
        """
        Train all models with their respective hyperparameter optimization.
        
        Args:
            X_train: Training features
            y_train: Training target
            dates_train: Training dates for ARIMA
            
        Returns:
            Dictionary of trained models
        """
        self.logger.info("Starting training of all models...")
        
        # Train Multiple Linear Regression
        self.trained_models['linear_regression'] = self.train_linear_regression(X_train, y_train)
        
        # Train K-Nearest Neighbors
        self.trained_models['knn'] = self.train_knn(X_train, y_train)
        
        # Train Random Forest
        self.trained_models['random_forest'] = self.train_random_forest(X_train, y_train)
        
        # Train XGBoost
        self.trained_models['xgboost'] = self.train_xgboost(X_train, y_train)
        
        # Train Artificial Neural Network
        self.trained_models['ann'] = self.train_ann(X_train, y_train)
        
        # Train ARIMA
        self.trained_models['arima'] = self.train_arima(dates_train, y_train)
        
        self.logger.info(f"Successfully trained {len(self.trained_models)} models")
        
        return self.trained_models
    
    def save_models(self) -> None:
        """Save all trained models to disk."""
        self.logger.info("Saving trained models...")
        
        for model_name, model in self.trained_models.items():
            if model_name == 'ann':
                # Save Keras model
                model.save(self.models_dir / f"{model_name}_model.h5")
            else:
                # Save sklearn/statsmodels models
                joblib.dump(model, self.models_dir / f"{model_name}_model.pkl")
        
        # Save best parameters
        with open(self.models_dir / "best_parameters.json", 'w') as f:
            json.dump(self.best_params, f, indent=2, default=str)
        
        self.logger.info(f"Models saved to {self.models_dir}")
    
    def load_models(self) -> Dict[str, Any]:
        """Load saved models from disk."""
        self.logger.info("Loading trained models...")
        
        loaded_models = {}
        
        # Load sklearn/statsmodels models
        for model_file in self.models_dir.glob("*_model.pkl"):
            model_name = model_file.stem.replace("_model", "")
            loaded_models[model_name] = joblib.load(model_file)
        
        # Load Keras model if exists
        ann_model_path = self.models_dir / "ann_model.h5"
        if ann_model_path.exists():
            loaded_models['ann'] = keras.models.load_model(ann_model_path)
        
        # Load best parameters
        params_file = self.models_dir / "best_parameters.json"
        if params_file.exists():
            with open(params_file, 'r') as f:
                self.best_params = json.load(f)
        
        self.logger.info(f"Loaded {len(loaded_models)} models")
        return loaded_models
