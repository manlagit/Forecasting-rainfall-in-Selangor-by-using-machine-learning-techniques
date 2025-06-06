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
        
    def plot_roc_curve(self, classification_results, output_path):
        """
        Plot ROC curve for multiple models.

        Args:
            classification_results: dict, where keys are model names and values are dicts containing:
                'fpr': array of false positive rates
                'tpr': array of true positive rates
                'roc_auc': AUC value
            output_path: path to save the plot
        """
        plt.figure(figsize=(10, 8))
        for model_name, results in classification_results.items():
            plt.plot(results['fpr'], results['tpr'], 
                     label=f'{model_name} (AUC = {results["roc_auc"]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
        
    def plot_confusion_matrix(self, y_true, y_pred, model_name, output_path):
        """
        Plot confusion matrix for a classification model.

        Args:
            y_true: true labels
            y_pred: predicted labels
            model_name: name of the model (for title)
            output_path: path to save the plot
        """
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Rain', 'Rain'], 
                    yticklabels=['No Rain', 'Rain'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(output_path)
        plt.close()
        
    def plot_feature_importance(self, model, feature_names, model_name, output_path):
        """
        Plot feature importances for a model that supports it.

        Args:
            model: trained model with feature_importances_ attribute
            feature_names: list of feature names
            model_name: name of the model (for title)
            output_path: path to save the plot
        """
        if not hasattr(model, 'feature_importances_'):
            raise ValueError(f"Model {model_name} does not support feature importance visualization")
            
        importances = model.feature_importances_
        indices = importances.argsort()[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importance ({model_name})')
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.xlim([-1, len(importances)])
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
