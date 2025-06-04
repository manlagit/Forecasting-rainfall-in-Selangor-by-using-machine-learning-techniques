"""
Artificial Neural Network (ANN) Model Implementation
for Rainfall Forecasting in Selangor
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import yaml
import logging
import pickle
import os

class ANNModel:
    """
    Artificial Neural Network model for rainfall prediction
    """
    
    def __init__(self, config_path="config/hyperparameters.yaml"):
        """
        Initialize ANN model with configuration
        
        Args:
            config_path (str): Path to hyperparameters configuration file
        """
        self.model = None
        self.best_params = None
        self.history = None
        self.logger = logging.getLogger(__name__)
        
        # Load hyperparameters from config
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            self.hyperparams = config['ann']
    
    def create_model(self, input_dim, layers=2, neurons=64, activation='relu', 
                     dropout_rate=0.2, learning_rate=0.001):
        """
        Create ANN model architecture
        
        Args:
            input_dim (int): Number of input features
            layers (int): Number of hidden layers
            neurons (int): Number of neurons per layer
            activation (str): Activation function
            dropout_rate (float): Dropout rate
            learning_rate (float): Learning rate
            
        Returns:
            keras.Model: Compiled neural network model
        """
        model = Sequential()
        
        # Input layer
        model.add(Dense(neurons, input_dim=input_dim, activation=activation))
        model.add(Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(layers - 1):
            model.add(Dense(neurons // (2 ** i), activation=activation))
            model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(1, activation='linear'))
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model    
    def optimize_hyperparameters_optuna(self, X_train, y_train, X_val, y_val, n_trials=50):
        """
        Optimize hyperparameters using Optuna
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training target
            X_val (np.array): Validation features
            y_val (np.array): Validation target
            n_trials (int): Number of Optuna trials
            
        Returns:
            dict: Best hyperparameters
        """
        
        def objective(trial):
            # Suggest hyperparameters
            layers = trial.suggest_categorical('layers', self.hyperparams['architecture']['layers'])
            neurons = trial.suggest_categorical('neurons', self.hyperparams['architecture']['neurons'])
            activation = trial.suggest_categorical('activation', self.hyperparams['architecture']['activation'])
            dropout_rate = trial.suggest_categorical('dropout_rate', self.hyperparams['architecture']['dropout_rate'])
            learning_rate = trial.suggest_categorical('learning_rate', self.hyperparams['training']['learning_rate'])
            batch_size = trial.suggest_categorical('batch_size', self.hyperparams['training']['batch_size'])
            epochs = trial.suggest_categorical('epochs', self.hyperparams['training']['epochs'])
            
            # Create and train model
            model = self.create_model(
                input_dim=X_train.shape[1],
                layers=layers,
                neurons=neurons,
                activation=activation,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate
            )
            
            # Early stopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            # Train model
            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Evaluate on validation set
            y_pred = model.predict(X_val, verbose=0)
            mse = mean_squared_error(y_val, y_pred)
            
            return mse
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params = study.best_params
        self.logger.info(f"Best ANN hyperparameters: {self.best_params}")
        
        return self.best_params    
    def train(self, X_train, y_train, X_val=None, y_val=None, optimize=True):
        """
        Train the ANN model
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training target
            X_val (np.array): Validation features (optional)
            y_val (np.array): Validation target (optional)
            optimize (bool): Whether to optimize hyperparameters
            
        Returns:
            keras.Model: Trained model
        """
        
        if optimize and X_val is not None and y_val is not None:
            # Optimize hyperparameters
            self.optimize_hyperparameters_optuna(X_train, y_train, X_val, y_val)
            
            # Train final model with best parameters
            self.model = self.create_model(
                input_dim=X_train.shape[1],
                **self.best_params
            )
        else:
            # Use default parameters
            self.model = self.create_model(input_dim=X_train.shape[1])
        
        # Prepare validation data
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        # Early stopping
        callbacks = []
        if validation_data:
            early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
            callbacks.append(early_stopping)
        
        # Train model
        epochs = self.best_params.get('epochs', 100) if self.best_params else 100
        batch_size = self.best_params.get('batch_size', 32) if self.best_params else 32
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.logger.info("ANN model training completed")
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
        
        predictions = self.model.predict(X_test, verbose=0)
        return predictions.flatten()
    
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
        
        # Save model
        self.model.save(filepath)
        
        # Save best parameters if available
        if self.best_params:
            params_filepath = filepath.replace('.h5', '_params.pkl')
            with open(params_filepath, 'wb') as f:
                pickle.dump(self.best_params, f)
        
        self.logger.info(f"ANN model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load trained model
        
        Args:
            filepath (str): Path to load model from
        """
        from tensorflow.keras.models import load_model
        
        self.model = load_model(filepath)
        
        # Load parameters if available
        params_filepath = filepath.replace('.h5', '_params.pkl')
        if os.path.exists(params_filepath):
            with open(params_filepath, 'rb') as f:
                self.best_params = pickle.load(f)
        
        self.logger.info(f"ANN model loaded from {filepath}")
    
    def get_model_summary(self):
        """
        Get model architecture summary
        
        Returns:
            str: Model summary
        """
        if self.model is None:
            return "No model available"
        
        return self.model.summary()
