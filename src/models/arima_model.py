"""
ARIMA Model Implementation
for Rainfall Forecasting in Selangor
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import itertools
import warnings
import yaml
import logging
import pickle
import os
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class ARIMAModel:
    """
    ARIMA model for rainfall prediction
    """
    
    def __init__(self, config_path="config/hyperparameters.yaml"):
        """
        Initialize ARIMA model with configuration
        
        Args:
            config_path (str): Path to hyperparameters configuration file
        """
        self.model = None
        self.fitted_model = None
        self.best_params = None
        self.aic_scores = {}
        self.bic_scores = {}
        self.logger = logging.getLogger(__name__)
        
        # Load hyperparameters from config
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            self.hyperparams = config['arima']
    
    def check_stationarity(self, timeseries, title="Time Series"):
        """
        Check if time series is stationary using Augmented Dickey-Fuller test
        
        Args:
            timeseries (pd.Series): Time series data
            title (str): Title for the test
            
        Returns:
            bool: True if stationary, False otherwise
        """
        
        # Perform Augmented Dickey-Fuller test
        adf_result = adfuller(timeseries.dropna())
        
        adf_statistic = adf_result[0]
        p_value = adf_result[1]
        critical_values = adf_result[4]
        
        self.logger.info(f"ADF Test Results for {title}:")
        self.logger.info(f"ADF Statistic: {adf_statistic}")
        self.logger.info(f"p-value: {p_value}")
        self.logger.info(f"Critical Values: {critical_values}")
        
        # Check if p-value is less than 0.05 (stationary)
        is_stationary = p_value < 0.05
        
        if is_stationary:
            self.logger.info("Time series is stationary")
        else:
            self.logger.info("Time series is not stationary")
        
        return is_stationary    
    def find_optimal_order(self, timeseries, max_p=None, max_d=None, max_q=None):
        """
        Find optimal ARIMA order using grid search based on AIC/BIC
        
        Args:
            timeseries (pd.Series): Time series data
            max_p (int): Maximum AR order
            max_d (int): Maximum differencing order
            max_q (int): Maximum MA order
            
        Returns:
            tuple: Best (p, d, q) order
        """
        
        # Use hyperparameters if not specified
        p_range = range(max_p + 1) if max_p else self.hyperparams['p_range']
        d_range = range(max_d + 1) if max_d else self.hyperparams['d_range']
        q_range = range(max_q + 1) if max_q else self.hyperparams['q_range']
        
        # Generate all combinations of parameters
        pdq_combinations = list(itertools.product(p_range, d_range, q_range))
        
        best_aic = float('inf')
        best_bic = float('inf')
        best_order_aic = None
        best_order_bic = None
        
        self.logger.info(f"Testing {len(pdq_combinations)} ARIMA parameter combinations...")
        
        for order in pdq_combinations:
            try:
                # Fit ARIMA model
                model = ARIMA(timeseries, order=order)
                fitted_model = model.fit()
                
                # Store AIC and BIC scores
                aic = fitted_model.aic
                bic = fitted_model.bic
                
                self.aic_scores[order] = aic
                self.bic_scores[order] = bic
                
                # Update best parameters
                if aic < best_aic:
                    best_aic = aic
                    best_order_aic = order
                
                if bic < best_bic:
                    best_bic = bic
                    best_order_bic = order
                
                self.logger.debug(f"ARIMA{order} - AIC: {aic:.2f}, BIC: {bic:.2f}")
                
            except Exception as e:
                self.logger.debug(f"Failed to fit ARIMA{order}: {e}")
                continue
        
        # Use AIC as primary criterion
        self.best_params = best_order_aic
        
        self.logger.info(f"Best ARIMA order (AIC): {best_order_aic} with AIC: {best_aic:.2f}")
        self.logger.info(f"Best ARIMA order (BIC): {best_order_bic} with BIC: {best_bic:.2f}")
        
        return best_order_aic    
    def train(self, timeseries, optimize=True, order=None):
        """
        Train the ARIMA model
        
        Args:
            timeseries (pd.Series): Time series data with datetime index
            optimize (bool): Whether to optimize order parameters
            order (tuple): Manual ARIMA order (p, d, q)
            
        Returns:
            ARIMAResults: Fitted ARIMA model
        """
        
        # Check stationarity
        self.check_stationarity(timeseries)
        
        if optimize and order is None:
            # Find optimal order
            order = self.find_optimal_order(timeseries)
        elif order is None:
            # Use default order
            order = (1, 1, 1)
        
        # Fit ARIMA model with best parameters
        try:
            self.model = ARIMA(timeseries, order=order)
            self.fitted_model = self.model.fit()
            self.best_params = order
            
            self.logger.info(f"ARIMA{order} model fitted successfully")
            self.logger.info(f"AIC: {self.fitted_model.aic:.2f}")
            self.logger.info(f"BIC: {self.fitted_model.bic:.2f}")
            
        except Exception as e:
            self.logger.error(f"Failed to fit ARIMA{order}: {e}")
            raise
        
        return self.fitted_model
    
    def predict(self, steps=1, start=None, end=None):
        """
        Make predictions using trained model
        
        Args:
            steps (int): Number of steps to forecast
            start (int): Start index for prediction
            end (int): End index for prediction
            
        Returns:
            np.array: Predictions
        """
        if self.fitted_model is None:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            if start is not None and end is not None:
                # In-sample and out-of-sample predictions
                forecast = self.fitted_model.predict(start=start, end=end)
            else:
                # Out-of-sample forecast
                forecast = self.fitted_model.forecast(steps=steps)
            
            return forecast.values if hasattr(forecast, 'values') else forecast
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise        figsize (tuple): Figure size
        """
        if self.fitted_model is None:
            raise ValueError("Model must be trained before plotting diagnostics")
        
        # Create diagnostic plots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot residuals
        residuals = self.fitted_model.resid
        
        # 1. Residuals vs Fitted
        axes[0, 0].scatter(self.fitted_model.fittedvalues, residuals)
        axes[0, 0].set_title('Residuals vs Fitted')
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        
        # 2. Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot')
        
        # 3. ACF of residuals
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(residuals, ax=axes[1, 0], lags=20)
        axes[1, 0].set_title('ACF of Residuals')
        
        # 4. Histogram of residuals
        axes[1, 1].hist(residuals, bins=20, density=True, alpha=0.7)
        axes[1, 1].set_title('Histogram of Residuals')
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Density')
        
        plt.tight_layout()
        return fig
