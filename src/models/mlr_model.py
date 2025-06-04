"""
Multiple Linear Regression (MLR) Model Implementation
for Rainfall Forecasting in Selangor
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import logging
import pickle
import os

class MLRModel:
    """
    Multiple Linear Regression model for rainfall prediction
    """
    
    def __init__(self, config_path="config/hyperparameters.yaml"):
        """
        Initialize MLR model with configuration
        
        Args:
            config_path (str): Path to hyperparameters configuration file
        """
        self.model = None
        self.feature_selector = None
        self.selected_features = None
        self.feature_importance = None
        self.diagnostics = {}
        self.logger = logging.getLogger(__name__)
        
        # Load hyperparameters from config
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            self.config = config.get('mlr', {})
    
    def select_features(self, X_train, y_train, method='RFE', n_features='auto'):
        """
        Perform feature selection
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training target
            method (str): Feature selection method ('RFE' or 'SelectKBest')
            n_features (str/int): Number of features to select
            
        Returns:
            np.array: Selected features
        """
        
        if n_features == 'auto':
            n_features = min(10, X_train.shape[1])  # Select up to 10 features
        
        if method == 'RFE':
            # Recursive Feature Elimination
            estimator = LinearRegression()
            self.feature_selector = RFE(estimator, n_features_to_select=n_features)
        elif method == 'SelectKBest':
            # Select K best features based on F-statistic
            self.feature_selector = SelectKBest(score_func=f_regression, k=n_features)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        # Fit feature selector
        X_selected = self.feature_selector.fit_transform(X_train, y_train)
        self.selected_features = self.feature_selector.get_support()
        
        self.logger.info(f"Selected {X_selected.shape[1]} features using {method}")
        return X_selected    
    def train(self, X_train, y_train, perform_feature_selection=True):
        """
        Train the MLR model
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training target
            perform_feature_selection (bool): Whether to perform feature selection
            
        Returns:
            LinearRegression: Trained model
        """
        
        # Feature selection
        if perform_feature_selection:
            method = self.config.get('feature_selection', {}).get('method', 'RFE')
            n_features = self.config.get('feature_selection', {}).get('n_features_to_select', 'auto')
            X_train_selected = self.select_features(X_train, y_train, method, n_features)
        else:
            X_train_selected = X_train
            self.selected_features = np.ones(X_train.shape[1], dtype=bool)
        
        # Initialize and train model
        self.model = LinearRegression()
        self.model.fit(X_train_selected, y_train)
        
        # Calculate feature importance (absolute coefficients)
        self.feature_importance = np.abs(self.model.coef_)
        
        self.logger.info("MLR model training completed")
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
        
        # Apply feature selection if used during training
        if self.feature_selector is not None:
            X_test_selected = self.feature_selector.transform(X_test)
        else:
            X_test_selected = X_test
        
        predictions = self.model.predict(X_test_selected)
        return predictions    
    def check_assumptions(self, X_train, y_train, y_pred):
        """
        Check linear regression assumptions
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training target
            y_pred (np.array): Predictions
            
        Returns:
            dict: Diagnostic results
        """
        
        residuals = y_train - y_pred
        
        # 1. Linearity (correlation between features and target)
        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_train)
        else:
            X_selected = X_train
            
        linearity_scores = []
        for i in range(X_selected.shape[1]):
            corr, _ = stats.pearsonr(X_selected[:, i], y_train)
            linearity_scores.append(abs(corr))
        
        # 2. Homoscedasticity (Breusch-Pagan test)
        # Simplified check: correlation between residuals and fitted values
        fitted_vs_residuals_corr, bp_pvalue = stats.pearsonr(y_pred, residuals**2)
        
        # 3. Normality of residuals (Shapiro-Wilk test)
        if len(residuals) <= 5000:  # Shapiro-Wilk works best for smaller samples
            _, normality_pvalue = stats.shapiro(residuals)
        else:
            # Use Kolmogorov-Smirnov test for larger samples
            _, normality_pvalue = stats.kstest(residuals, 'norm')
        
        # 4. Independence (Durbin-Watson test approximation)
        # Simplified: check for autocorrelation in residuals
        residuals_shifted = np.roll(residuals, 1)[1:]
        residuals_current = residuals[1:]
        autocorr, _ = stats.pearsonr(residuals_current, residuals_shifted)
        
        self.diagnostics = {
            'linearity_scores': linearity_scores,
            'mean_linearity': np.mean(linearity_scores),
            'homoscedasticity_pvalue': bp_pvalue,
            'normality_pvalue': normality_pvalue,
            'autocorrelation': autocorr,
            'residuals_mean': np.mean(residuals),
            'residuals_std': np.std(residuals)
        }
        
        return self.diagnostics    
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
            'feature_selector': self.feature_selector,
            'selected_features': self.selected_features,
            'feature_importance': self.feature_importance,
            'diagnostics': self.diagnostics
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"MLR model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load trained model
        
        Args:
            filepath (str): Path to load model from
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_selector = model_data.get('feature_selector')
        self.selected_features = model_data.get('selected_features')
        self.feature_importance = model_data.get('feature_importance')
        self.diagnostics = model_data.get('diagnostics', {})
        
        self.logger.info(f"MLR model loaded from {filepath}")
    
    def get_feature_importance(self):
        """
        Get feature importance scores
        
        Returns:
            np.array: Feature importance scores
        """
        return self.feature_importance
    
    def get_model_summary(self):
        """
        Get model summary including coefficients and diagnostics
        
        Returns:
            dict: Model summary
        """
        if self.model is None:
            return "No model available"
        
        summary = {
            'intercept': self.model.intercept_,
            'coefficients': self.model.coef_,
            'n_features': len(self.model.coef_),
            'feature_importance': self.feature_importance,
            'diagnostics': self.diagnostics
        }
        
        return summary
