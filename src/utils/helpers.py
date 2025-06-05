"""
Utility functions for the rainfall forecasting project.
Contains helper functions used across multiple modules.
"""

import pandas as pd
import numpy as np
import joblib
import logging
import yaml
from typing import Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


def load_model(model_path: str) -> Any:
    """
    Load a saved model from file.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded model object
    """
    try:
        if model_path.endswith('.h5'):
            from tensorflow import keras
            return keras.models.load_model(model_path)
        else:
            return joblib.load(model_path)
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        return None


def save_results_to_csv(results_dict: Dict[str, Any], filename: str) -> None:
    """
    Save results dictionary to CSV file.
    
    Args:
        results_dict: Dictionary containing results
        filename: Output filename
    """
    try:
        df = pd.DataFrame(results_dict)
        df.to_csv(filename, index=False)
        logger.info(f"Saved results to {filename}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")


def calculate_percentage_improvement(
        baseline_rmse: float, model_rmse: float) -> float:
    """
    Calculate percentage improvement over baseline.
    
    Args:
        baseline_rmse: RMSE of baseline model
        model_rmse: RMSE of comparison model
        
    Returns:
        Percentage improvement (negative if worse)
    """
    return ((baseline_rmse - model_rmse) / baseline_rmse) * 100


def create_feature_importance_dict(
        feature_names: List[str], 
        importance_values: np.ndarray) -> Dict[str, float]:
    """
    Create feature importance dictionary.
    
    Args:
        feature_names: List of feature names
        importance_values: Array of importance values
        
    Returns:
        Dictionary mapping features to importance values
    """
    return dict(zip(feature_names, importance_values))


def ensure_directory_exists(directory_path: str) -> Path:
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        directory_path: Path to directory
        
    Returns:
        Path object for the directory
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_data_schema(
        df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that DataFrame contains required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if all required columns present
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    return True


def get_model_summary_stats(
        y_true: np.ndarray, 
        y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive model statistics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of statistics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Additional statistics
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    residuals = y_true - y_pred
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2,
        'MAPE': mape,
        'Mean_Residual': mean_residual,
        'Std_Residual': std_residual
    }


def format_number(value: float, decimal_places: int = 4) -> str:
    """
    Format number for display.
    
    Args:
        value: Number to format
        decimal_places: Number of decimal places
        
    Returns:
        Formatted string
    """
    return f"{value:.{decimal_places}f}"


def load_config(
        config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return {}


def load_yaml(file_path: str) -> Dict[str, Any]:
    """
    Load data from a YAML file.
    
    Args:
        file_path: Path to YAML file
        
    Returns:
        Dictionary with the loaded data
    """
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load YAML from {file_path}: {e}")
        return {}
