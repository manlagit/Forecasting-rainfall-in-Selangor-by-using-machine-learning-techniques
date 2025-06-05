"""
Visualization & Automation Module
Automated plot generation with LaTeX integration.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')

class RainfallVisualizer:
    """
    Comprehensive visualization generator with LaTeX integration.
    """
    
    def __init__(self, figures_dir: str = "reports/figures"):
        """
        Initialize visualizer.
        
        Args:
            figures_dir: Directory to save figures
        """
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Configure matplotlib for LaTeX
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
    
    def plot_time_series_comparison(
        self, 
        df: pd.DataFrame, 
        predictions: Dict[str, Any],
        model_name: str = 'best_model'
    ) -> str:
        """
        Create time series plot comparing actual vs predicted rainfall.
        
        Args:
            df: Processed DataFrame with dates
            predictions: Dictionary of model predictions
            model_name: Name of model to plot
            
        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Get prediction data
        if model_name in predictions:
            pred_data = predictions[model_name]
            
            # Create date index for plotting (use last portion of dates)
            dates = df['Date'].tail(len(pred_data['y_true']))
            
            # Plot actual vs predicted
            ax.plot(
                dates, pred_data['y_true'], 
                label='Actual Rainfall', linewidth=2, alpha=0.8
            )
            ax.plot(
                dates, pred_data['y_pred'], 
                label='Predicted Rainfall', linewidth=2, alpha=0.8
            )
            
            # Add confidence bands
            residuals = pred_data['residuals']
            std_residual = np.std(residuals)
            upper_bound = pred_data['y_pred'] + 1.96 * std_residual
            lower_bound = pred_data['y_pred'] - 1.96 * std_residual
            
            ax.fill_between(
                dates, lower_bound, upper_bound, 
                alpha=0.2, label='95% Confidence Interval'
            )
            
            # Formatting
            ax.set_title(
                (
                    f'Rainfall Forecasting: Actual vs Predicted '
                    f'({model_name.title()})'
                ),
                fontsize=16, 
                fontweight='bold'
            )
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Precipitation (mm)', fontsize=12)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Remove legend to avoid tikzplotlib issue
            ax.get_legend().remove()
            
            # Save plot
            plot_path = self.figures_dir / f"time_series_{model_name}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            
            # Convert to LaTeX (commented out due to tikzplotlib error)
            # latex_path = self.figures_dir / f"time_series_{model_name}.tex"
            # tikzplotlib.save(latex_path)
            
            plt.close()
            
            self.logger.info(f"Time series plot saved: {plot_path}")
            return str(plot_path)
        
        else:
            self.logger.warning(f"Model {model_name} not found in predictions")
            return ""
    
    def plot_scatter_actual_vs_predicted(
        self, 
        predictions: Dict[str, Any]
    ) -> str:
        """
        Create scatter plots of predicted vs actual values for all models.
        
        Args:
            predictions: Dictionary of model predictions
            
        Returns:
            Path to saved plot
        """
        n_models = len(predictions)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (model_name, pred_data) in enumerate(predictions.items()):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            y_true = pred_data['y_true']
            y_pred = pred_data['y_pred']
            
            # Scatter plot
            ax.scatter(y_true, y_pred, alpha=0.6, s=50)
            
            # Perfect prediction line
            min_val = min(min(y_true), min(y_pred))
            max_val = max(max(y_true), max(y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], 
                   'r--', linewidth=2, label='Perfect Prediction')
            
            # Calculate R²
            r2 = np.corrcoef(y_true, y_pred)[0, 1]**2
            
            ax.set_title(f'{model_name.title()} (R² = {r2:.4f})', fontweight='bold')
            ax.set_xlabel('Actual Precipitation (mm)')
            ax.set_ylabel('Predicted Precipitation (mm)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_models, rows * cols):
            row = idx // cols
            col = idx % cols
            if rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.suptitle('Actual vs Predicted Rainfall Comparison', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Remove legends to avoid tikzplotlib issue
        for ax in axes.flat:
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        
        # Save plot
        plot_path = self.figures_dir / "scatter_actual_vs_predicted.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        # Convert to LaTeX (commented out due to tikzplotlib error)
        # latex_path = self.figures_dir / "scatter_actual_vs_predicted.tex"
        # tikzplotlib.save(latex_path)
        
        plt.close()
        
        self.logger.info(f"Scatter plot saved: {plot_path}")
        return str(plot_path)
    
    def plot_model_performance_comparison(
        self, 
        comparison_df: pd.DataFrame
    ) -> str:
        """
        Create bar chart comparing model performance.
        
        Args:
            comparison_df: DataFrame with model comparison results
            
        Returns:
            Path to saved plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        models = comparison_df.index
        
        # RMSE comparison
        bars1 = ax1.bar(models, comparison_df['RMSE'], 
                       color=sns.color_palette("husl", len(models)))
        ax1.set_title('Root Mean Square Error (RMSE)', fontweight='bold')
        ax1.set_ylabel('RMSE')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10)
        
        # MAE comparison
        bars2 = ax2.bar(models, comparison_df['MAE'],
                       color=sns.color_palette("husl", len(models)))
        ax2.set_title('Mean Absolute Error (MAE)', fontweight='bold')
        ax2.set_ylabel('MAE')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10)
        
        # R² comparison
        bars3 = ax3.bar(models, comparison_df['R2'],
                       color=sns.color_palette("husl", len(models)))
        ax3.set_title('Coefficient of Determination (R²)', fontweight='bold')
        ax3.set_ylabel('R²')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10)
        
        # MAPE comparison
        bars4 = ax4.bar(models, comparison_df['MAPE'],
                       color=sns.color_palette("husl", len(models)))
        ax4.set_title(
            'Mean Absolute Percentage Error (MAPE)', 
            fontweight='bold'
        )
        ax4.set_ylabel('MAPE (%)')
        ax4.tick_params(axis='x', rotation=45)
        
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%', ha='center', va='bottom', fontsize=10)
        
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = self.figures_dir / "model_performance_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        # Convert to LaTeX (commented out due to tikzplotlib error)
        # latex_path = self.figures_dir / "model_performance_comparison.tex"
        # tikzplotlib.save(latex_path)
        
        plt.close()
        
        self.logger.info(f"Performance comparison plot saved: {plot_path}")
        return str(plot_path)
    
    def plot_residual_analysis(
        self, 
        predictions: Dict[str, Any]
    ) -> str:
        """
        Create residual plots for error analysis.
        
        Args:
            predictions: Dictionary of model predictions
            
        Returns:
            Path to saved plot
        """
        n_models = len(predictions)
        cols = 2
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 6*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (model_name, pred_data) in enumerate(predictions.items()):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            residuals = pred_data['residuals']
            y_pred = pred_data['y_pred']
            
            # Residual scatter plot
            ax.scatter(y_pred, residuals, alpha=0.6, s=30)
            ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
            
            ax.set_title(f'{model_name.title()} - Residual Analysis', fontweight='bold')
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Residuals')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals)
            ax.text(
                0.05, 0.95, 
                f'Mean: {mean_residual:.4f}\nStd: {std_residual:.4f}',
                transform=ax.transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            )
        
        # Hide unused subplots
        for idx in range(n_models, rows * cols):
            row = idx // cols
            col = idx % cols
            if rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.suptitle('Residual Analysis for All Models', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = self.figures_dir / "residual_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        # Convert to LaTeX (commented out due to tikzplotlib error)
        # latex_path = self.figures_dir / "residual_analysis.tex"
        # tikzplotlib.save(latex_path)
        
        plt.close()
        
        self.logger.info(f"Residual analysis plot saved: {plot_path}")
        return str(plot_path)
    
    def plot_feature_importance(
        self, 
        importance_data: Dict[str, np.ndarray],
        feature_names: List[str]
    ) -> str:
        """
        Create feature importance plots.
        
        Args:
            importance_data: Dictionary of feature importance arrays
            feature_names: List of feature names
            
        Returns:
            Path to saved plot
        """
        if not importance_data:
            self.logger.warning("No feature importance data available")
            return ""
        
        n_models = len(importance_data)
        fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 10))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, importance) in enumerate(importance_data.items()):
            ax = axes[idx]
            
            # Sort features by importance
            sorted_idx = np.argsort(importance)[::-1]
            sorted_features = [feature_names[i] for i in sorted_idx]
            sorted_importance = importance[sorted_idx]
            
            # Create horizontal bar plot
            y_pos = np.arange(len(sorted_features))
            bars = ax.barh(y_pos, sorted_importance)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_features)
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'{model_name.title()} Feature Importance', fontweight='bold')
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(
                    width, 
                    bar.get_y() + bar.get_height()/2.,
                    f'{width:.3f}', 
                    ha='left', 
                    va='center', 
                    fontsize=9
                )
            
            ax.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('Feature Importance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = self.figures_dir / "feature_importance.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        # Convert to LaTeX (commented out due to tikzplotlib error)
        # latex_path = self.figures_dir / "feature_importance.tex"
        # tikzplotlib.save(latex_path)
        
        plt.close()
        
        self.logger.info(f"Feature importance plot saved: {plot_path}")
        return str(plot_path)
    
    def plot_correlation_matrix(
        self, 
        df: pd.DataFrame
    ) -> str:
        """
        Create correlation matrix heatmap.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Path to saved plot
        """
        # Select numeric columns for correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Create heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(
            correlation_matrix, 
            mask=mask, 
            annot=True, 
            cmap='coolwarm', 
            center=0, 
            square=True,
            fmt='.2f', 
            cbar_kws={"shrink": .8}
        )
        
        ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = self.figures_dir / "correlation_matrix.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        # Convert to LaTeX (commented out due to tikzplotlib error)
        # latex_path = self.figures_dir / "correlation_matrix.tex"
        # tikzplotlib.save(latex_path)
        
        plt.close()
        
        self.logger.info(f"Correlation matrix plot saved: {plot_path}")
        return str(plot_path)
    
    def generate_all_plots(
        self, 
        df_processed: pd.DataFrame, 
        comparison_df: pd.DataFrame,
        predictions: Dict[str, Any],
        importance_data: Dict[str, np.ndarray] = None
    ) -> List[str]:
        """
        Generate all visualization plots.
        
        Args:
            df_processed: Processed DataFrame
            comparison_df: Model comparison results
            predictions: Model predictions
            importance_data: Feature importance data
            
        Returns:
            List of paths to generated plots
        """
        self.logger.info("Generating all visualization plots...")
        
        plot_paths = []
        
        # Time series comparison (best model)
        best_model = comparison_df.index[0]
        if best_model in predictions:
            path = self.plot_time_series_comparison(df_processed, predictions, best_model)
            if path:
                plot_paths.append(path)
        
        # Scatter plots
        path = self.plot_scatter_actual_vs_predicted(predictions)
        if path:
            plot_paths.append(path)
        
        # Model performance comparison
        path = self.plot_model_performance_comparison(comparison_df)
        if path:
            plot_paths.append(path)
        
        # Residual analysis
        path = self.plot_residual_analysis(predictions)
        if path:
            plot_paths.append(path)
        
        # Feature importance (if available)
        if importance_data:
            feature_names = df_processed.columns.drop(
                ['Date', 'Precipitation_mm', 'Year']
            ).tolist()
            path = self.plot_feature_importance(importance_data, feature_names)
            if path:
                plot_paths.append(path)
        
        # Correlation matrix
        path = self.plot_correlation_matrix(df_processed)
        if path:
            plot_paths.append(path)
        
        self.logger.info(f"Generated {len(plot_paths)} visualization plots")
        return plot_paths
