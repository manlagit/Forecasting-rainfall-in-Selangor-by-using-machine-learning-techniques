"""
Visualization Module
Generates visualizations for rainfall forecasting results.
"""

import logging
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
import seaborn as sns

class RainfallVisualizer:
    """
    Generates visualizations for rainfall forecasting results.
    """
    
    def __init__(self):
        """
        Initialize the visualizer.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized RainfallVisualizer")
        
    def generate_all_plots(self, df: pd.DataFrame, comparison_df: pd.DataFrame, 
                          predictions: dict, output_dir: str = "reports/figures") -> list:
        """
        Generate all visualizations and return paths to saved figures.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plot_paths = []
        
        # Time series plot
        if 'Date' in df.columns:
            ts_path = os.path.join(output_dir, "time_series.png")
            self.plot_time_series(df, ts_path)
            plot_paths.append(ts_path)
        
        # Prediction vs Actual for each model
        for model_name in predictions:
            if model_name in predictions:
                pred_path = os.path.join(output_dir, f"{model_name}_pred_vs_actual.png")
                self.plot_prediction_vs_actual(
                    predictions[model_name]['true'],
                    predictions[model_name]['pred'],
                    model_name,
                    pred_path
                )
                plot_paths.append(pred_path)
        
        # Model comparison bar chart
        if not comparison_df.empty:
            comp_path = os.path.join(output_dir, "model_comparison.png")
            self.plot_model_comparison(comparison_df, comp_path)
            plot_paths.append(comp_path)
        
        # Feature importance (if available)
        # This would require models that support feature importance
        
        self.logger.info(f"Generated {len(plot_paths)} visualization plots")
        return plot_paths
    
    def plot_time_series(self, df: pd.DataFrame, output_path: str):
        """
        Plot time series of actual rainfall.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(df['Date'], df['Precipitation_mm'], label='Actual Rainfall')
        plt.title('Rainfall Time Series')
        plt.xlabel('Date')
        plt.ylabel('Precipitation (mm)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
    def plot_prediction_vs_actual(self, y_true: pd.Series, y_pred: pd.Series, 
                                 model_name: str, output_path: str):
        """
        Plot predicted vs actual values for a model.
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
        plt.title(f'Predicted vs Actual Rainfall ({model_name})')
        plt.xlabel('Actual Rainfall (mm)')
        plt.ylabel('Predicted Rainfall (mm)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
    def plot_model_comparison(self, comparison_df: pd.DataFrame, output_path: str):
        """
        Create bar chart comparing model performance metrics.
        """
        plt.figure(figsize=(12, 8))
        comparison_df[['RMSE', 'MAE']].plot(kind='bar')
        plt.title('Model Performance Comparison')
        plt.xlabel('Model')
        plt.ylabel('Error')
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
    def plot_feature_importance(self, feature_importances: pd.Series, 
                               model_name: str, output_path: str):
        """
        Plot feature importances for a model (if available).
        """
        plt.figure(figsize=(12, 8))
        feature_importances.sort_values(ascending=False).plot(kind='bar')
        plt.title(f'Feature Importance ({model_name})')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
