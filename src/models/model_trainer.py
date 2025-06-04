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
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
        """Train Random Forest model with hyperparameter tuning."""
        logger.info("Training Random Forest model...")
        
        param_grid = {
            'n_estimators': self.model_config['rf']['n_estimators'],
            'max_depth': self.model_config['rf']['max_depth'],
            'min_samples_split': self.model_config['rf']['min_samples_split'],
            'min_samples_leaf': self.model_config['rf']['min_samples_leaf']
        }
        
        rf = RandomForestRegressor(random_state=self.model_config['random_state'])
        grid_search = GridSearchCV(
            rf, param_grid, 
            cv=self.model_config['cv_folds'],
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.models['random_forest'] = grid_search.best_estimator_
        self.best_params['random_forest'] = grid_search.best_params_
        
        logger.info(f"Best RF params: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBRegressor:
        """Train XGBoost model with hyperparameter tuning."""
        logger.info("Training XGBoost model...")
        
        param_grid = {
            'n_estimators': self.model_config['xgb']['n_estimators'],
            'learning_rate': self.model_config['xgb']['learning_rate'],
            'max_depth': self.model_config['xgb']['max_depth'],
            'subsample': self.model_config['xgb']['subsample'],
            'colsample_bytree': self.model_config['xgb']['colsample_bytree']
        }
        
        xgb_model = xgb.XGBRegressor(random_state=self.model_config['random_state'])
        grid_search = GridSearchCV(
            xgb_model, param_grid, 
            cv=self.model_config['cv_folds'],
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.models['xgboost'] = grid_search.best_estimator_
        self.best_params['xgboost'] = grid_search.best_params_
        
        logger.info(f"Best XGBoost params: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    def train_ann(self, X_train: pd.DataFrame, y_train: pd.Series) -> keras.Model:
        """Train ANN model with Optuna hyperparameter optimization."""
        logger.info("Training ANN model with Optuna...")
        
        def objective(trial):
            # Define hyperparameters to optimize
            n_layers = trial.suggest_int('n_layers', 2, 4)
            n_neurons = trial.suggest_categorical('n_neurons', [32, 64, 128])
            activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.3)
            learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            
            # Build model
            model = keras.Sequential()
            model.add(layers.Dense(n_neurons, activation=activation, input_shape=(X_train.shape[1],)))
            model.add(layers.Dropout(dropout_rate))
            
            for _ in range(n_layers - 1):
                model.add(layers.Dense(n_neurons // 2, activation=activation))
                model.add(layers.Dropout(dropout_rate))
            
            model.add(layers.Dense(1, activation='linear'))
            
            # Compile model
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            
            # Train with early stopping
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            )
            
            history = model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=100,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Return validation loss for optimization
            return min(history.history['val_loss'])
        
        # Optimize hyperparameters
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)
        
        # Train final model with best parameters
        best_params = study.best_params
        
        model = keras.Sequential()
        model.add(layers.Dense(best_params['n_neurons'], 
                              activation=best_params['activation'], 
                              input_shape=(X_train.shape[1],)))
        model.add(layers.Dropout(best_params['dropout_rate']))
        
        for _ in range(best_params['n_layers'] - 1):
            model.add(layers.Dense(best_params['n_neurons'] // 2, 
                                  activation=best_params['activation']))
            model.add(layers.Dropout(best_params['dropout_rate']))
        
        model.add(layers.Dense(1, activation='linear'))
        
        optimizer = keras.optimizers.Adam(learning_rate=best_params['learning_rate'])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        )
        
        model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=200,
            batch_size=best_params['batch_size'],
            callbacks=[early_stopping],
            verbose=1
        )
        
        self.models['ann'] = model
        self.best_params['ann'] = best_params
        
        logger.info(f"Best ANN params: {best_params}")
        
        return model
    
    def train_arima(self, y_train: pd.Series, dates: pd.Series) -> ARIMA:
        """Train ARIMA model with automatic parameter selection."""
        logger.info("Training ARIMA model...")
        
        # Create time series with proper index
        ts = pd.Series(y_train.values, index=dates)
        
        # Test for stationarity
        adf_result = adfuller(ts.dropna())
        logger.info(f"ADF Statistic: {adf_result[0]:.6f}")
        logger.info(f"p-value: {adf_result[1]:.6f}")
        
        # Determine differencing order
        d = 0 if adf_result[1] < 0.05 else 1
        
        # Grid search for best ARIMA parameters
        best_aic = np.inf
        best_params = None
        best_model = None
        
        p_range = self.model_config['arima']['p_range']
        q_range = self.model_config['arima']['q_range']
        
        for p in p_range:
            for q in q_range:
                try:
                    model = ARIMA(ts, order=(p, d, q))
                    fitted_model = model.fit()
                    
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_params = (p, d, q)
                        best_model = fitted_model
                        
                except Exception as e:
                    continue
        
        self.models['arima'] = best_model
        self.best_params['arima'] = {
            'order': best_params,
            'aic': best_aic
        }
        
        logger.info(f"Best ARIMA params: {best_params}, AIC: {best_aic:.2f}")
        
        return best_model
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                        dates: pd.Series = None) -> Dict[str, Any]:
        """Train all models and return them."""
        logger.info("Training all models...")
        
        models = {}
        
        # Train traditional ML models
        models['linear_regression'] = self.train_linear_regression(X_train, y_train)
        models['knn'] = self.train_knn(X_train, y_train)
        models['random_forest'] = self.train_random_forest(X_train, y_train)
        models['xgboost'] = self.train_xgboost(X_train, y_train)
        
        # Train ANN
        models['ann'] = self.train_ann(X_train, y_train)
        
        # Train ARIMA (time series specific)
        if dates is not None:
            models['arima'] = self.train_arima(y_train, dates)
        
        return models
    
    def save_models(self, output_dir: str = "models/saved_models"):
        """Save all trained models."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            if name == 'ann':
                # Save Keras model
                model.save(output_path / f'{name}_model.h5')
            else:
                # Save sklearn/other models
                joblib.dump(model, output_path / f'{name}_model.pkl')
        
        # Save best parameters
        joblib.dump(self.best_params, output_path / 'best_parameters.pkl')
        
        logger.info(f"Saved {len(self.models)} models to {output_path}")
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                      model_name: str) -> Dict[str, float]:
        """Evaluate a single model."""
        if model_name == 'linear_regression':
            # Use selected features for MLR
            selected_features = self.best_params['linear_regression']['selected_features']
            y_pred = model.predict(X_test[selected_features])
        elif model_name == 'ann':
            y_pred = model.predict(X_test).flatten()
        elif model_name == 'arima':
            # ARIMA prediction logic would be different
            # For now, return placeholder metrics
            return {
                'mae': 0.0, 'mse': 0.0, 'rmse': 0.0, 'r2': 0.0
            }
        else:
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'mae': mae,
            'mse': mse, 
            'rmse': rmse,
            'r2': r2
        }
