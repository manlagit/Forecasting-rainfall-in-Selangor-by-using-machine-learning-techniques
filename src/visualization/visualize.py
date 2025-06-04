"""
Visualization module for rainfall forecasting project.
Creates plots and figures for analysis and reporting.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import tikzplotlib
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Set style for consistent plots
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    plt.style.use('seaborn')
sns.set_palette("husl")


class RainfallVisualizer:
    """Handles all visualization tasks for the rainfall forecasting project."""
    
    def __init__(self, output_dir: str = "reports/figures"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_time_series(self, data: pd.DataFrame, date_col: str = 'Date', 
                        target_col: str = 'Precipitation_mm', 
                        title: str = "Rainfall Time Series") -> str:
        """
        Plot time series of rainfall data.
        
        Args:
            data: DataFrame containing time series data
            date_col: Name of date column
            target_col: Name of target variable column
            title: Plot title
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(data[date_col], data[target_col], linewidth=1, alpha=0.7)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Precipitation (mm)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        x_numeric = np.arange(len(data))
        z = np.polyfit(x_numeric, data[target_col], 1)
        p = np.poly1d(z)
        ax.plot(data[date_col], p(x_numeric), "r--", alpha=0.8, 
                label=f'Trend (slope: {z[0]:.4f})')
        ax.legend()
        
        plt.tight_layout()
        
        # Save plot
        filename = "rainfall_time_series.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        # Save as LaTeX/PGF
        tikz_filepath = self.output_dir / "rainfall_time_series.tex"
        tikzplotlib.save(tikz_filepath)
        
        plt.close()
        logger.info(f"Saved time series plot to {filepath}")
        
        return str(filepath)    
    def plot_correlation_matrix(self, data: pd.DataFrame, 
                               title: str = "Feature Correlation Matrix") -> str:
        """
        Plot correlation matrix heatmap.
        
        Args:
            data: DataFrame with features
            title: Plot title
            
        Returns:
            Path to saved figure
        """
        # Select only numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        corr_matrix = data[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax, fmt='.2f', cbar_kws={'shrink': 0.8})
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        filename = "correlation_matrix.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        # Save as LaTeX/PGF
        tikz_filepath = self.output_dir / "correlation_matrix.tex"
        tikzplotlib.save(tikz_filepath)
        
        plt.close()
        logger.info(f"Saved correlation matrix to {filepath}")
        
        return str(filepath)
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame, 
                             metric: str = 'RMSE',
                             title: str = "Model Performance Comparison") -> str:
        """
        Plot model comparison bar chart.
        
        Args:
            comparison_df: DataFrame with model comparison results
            metric: Metric to plot
            title: Plot title
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar plot
        bars = ax.bar(comparison_df.index, comparison_df[metric], 
                     color='skyblue', alpha=0.8, edgecolor='navy')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot
        filename = "model_comparison.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        # Save as LaTeX/PGF
        tikz_filepath = self.output_dir / "model_comparison.tex"
        tikzplotlib.save(tikz_filepath)
        
        plt.close()
        logger.info(f"Saved model comparison plot to {filepath}")
        
        return str(filepath)
    
    def plot_predictions_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  model_name: str, 
                                  title: str = None) -> str:
        """
        Plot predicted vs actual values scatter plot.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            title: Plot title
            
        Returns:
            Path to saved figure
        """
        if title is None:
            title = f"Predicted vs Actual - {model_name}"
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, s=50)
        
        # Perfect prediction line (y = x)
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, 
                label='Perfect Prediction')
        
        # Calculate R²
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        
        ax.set_title(f"{title}\nR² = {r2:.4f}", fontsize=14, fontweight='bold')
        ax.set_xlabel('Actual Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        filename = f"predictions_vs_actual_{model_name.lower()}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        # Save as LaTeX/PGF
        tikz_filepath = self.output_dir / f"predictions_vs_actual_{model_name.lower()}.tex"
        tikzplotlib.save(tikz_filepath)
        
        plt.close()
        logger.info(f"Saved predictions vs actual plot to {filepath}")
        
        return str(filepath)
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                      model_name: str, title: str = None) -> str:
        """
        Plot residual analysis.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            title: Plot title
            
        Returns:
            Path to saved figure
        """
        if title is None:
            title = f"Residual Analysis - {model_name}"
        
        residuals = y_true - y_pred
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Residuals vs Predicted
        ax1.scatter(y_pred, residuals, alpha=0.6)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_title('Residuals vs Predicted')
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.grid(True, alpha=0.3)
        
        # 2. Histogram of residuals
        ax2.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_title('Distribution of Residuals')
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # 3. Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot')
        ax3.grid(True, alpha=0.3)
        
        # 4. Residuals vs Index (time series)
        ax4.plot(residuals, alpha=0.7)
        ax4.axhline(y=0, color='r', linestyle='--')
        ax4.set_title('Residuals vs Index')
        ax4.set_xlabel('Index')
        ax4.set_ylabel('Residuals')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        filename = f"residuals_{model_name.lower()}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        plt.close()
        logger.info(f"Saved residual analysis plot to {filepath}")
        
        return str(filepath)
    
    def plot_feature_importance(self, feature_names: List[str], 
                               importance_values: np.ndarray,
                               model_name: str, title: str = None) -> str:
        """
        Plot feature importance.
        
        Args:
            feature_names: List of feature names
            importance_values: Array of importance values
            model_name: Name of the model
            title: Plot title
            
        Returns:
            Path to saved figure
        """
        if title is None:
            title = f"Feature Importance - {model_name}"
        
        # Sort features by importance
        indices = np.argsort(importance_values)[::-1]
        sorted_features = [feature_names[i] for i in indices]
        sorted_importance = importance_values[indices]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(sorted_features)), sorted_importance, 
                      color='lightcoral', alpha=0.8)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels(sorted_features)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        # Save plot
        filename = f"feature_importance_{model_name.lower()}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        # Save as LaTeX/PGF
        tikz_filepath = self.output_dir / f"feature_importance_{model_name.lower()}.tex"
        tikzplotlib.save(tikz_filepath)
        
        plt.close()
        logger.info(f"Saved feature importance plot to {filepath}")
        
        return str(filepath)
    
    def generate_all_plots(self, data: pd.DataFrame, 
                          comparison_df: pd.DataFrame,
                          predictions_dict: Dict) -> List[str]:
        """
        Generate all visualization plots.
        
        Args:
            data: Original dataset
            comparison_df: Model comparison results
            predictions_dict: Dictionary of predictions for each model
            
        Returns:
            List of paths to generated plots
        """
        plot_paths = []
        
        # Time series plot
        if 'Date' in data.columns and 'Precipitation_mm' in data.columns:
            path = self.plot_time_series(data)
            plot_paths.append(path)
        
        # Correlation matrix
        path = self.plot_correlation_matrix(data)
        plot_paths.append(path)
        
        # Model comparison
        if not comparison_df.empty:
            path = self.plot_model_comparison(comparison_df)
            plot_paths.append(path)
        
        # Predictions vs actual and residuals for each model
        for model_name, pred_data in predictions_dict.items():
            if 'y_true' in pred_data and 'y_pred' in pred_data:
                # Predictions vs actual
                path = self.plot_predictions_vs_actual(
                    pred_data['y_true'], pred_data['y_pred'], model_name
                )
                plot_paths.append(path)
                
                # Residual analysis
                path = self.plot_residuals(
                    pred_data['y_true'], pred_data['y_pred'], model_name
                )
                plot_paths.append(path)
        
        logger.info(f"Generated {len(plot_paths)} visualization plots")
        
        return plot_paths
