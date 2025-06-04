"""
Model evaluation module for rainfall forecasting.
Provides comprehensive evaluation metrics and comparison functions.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Tuple, Any
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Handles model evaluation and comparison."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.results = {}
        self.predictions = {}
        
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      model_name: str) -> Dict[str, float]:
        """
        Evaluate a single model with comprehensive metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values  
            model_name: Name of the model
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Calculate regression metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate additional metrics
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        # Calculate residuals
        residuals = y_true - y_pred
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2,
            'MAPE': mape,
            'Mean_Residual': mean_residual,
            'Std_Residual': std_residual
        }
        
        # Store results
        self.results[model_name] = metrics
        self.predictions[model_name] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'residuals': residuals
        }
        
        logger.info(f"Evaluated {model_name}: RMSE={rmse:.4f}, R²={r2:.4f}")
        
        return metrics
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all evaluated models.
        
        Returns:
            DataFrame with model comparison results
        """
        if not self.results:
            logger.warning("No models have been evaluated yet.")
            return pd.DataFrame()
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(self.results).T
        
        # Sort by RMSE (ascending - lower is better)
        comparison_df = comparison_df.sort_values('RMSE')
        
        # Add ranking
        comparison_df['Rank'] = range(1, len(comparison_df) + 1)
        
        return comparison_df
    
    def perform_statistical_tests(self) -> Dict[str, Any]:
        """
        Perform statistical significance tests between models.
        
        Returns:
            Dictionary with test results
        """
        if len(self.results) < 2:
            logger.warning("Need at least 2 models for statistical testing.")
            return {}
        
        test_results = {}
        model_names = list(self.results.keys())
        
        # Paired t-tests between models
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                residuals1 = self.predictions[model1]['residuals']
                residuals2 = self.predictions[model2]['residuals']
                
                # Paired t-test on squared residuals
                squared_res1 = residuals1 ** 2
                squared_res2 = residuals2 ** 2
                
                t_stat, p_value = stats.ttest_rel(squared_res1, squared_res2)
                
                test_results[f"{model1}_vs_{model2}"] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        return test_results
    
    def generate_summary_report(self) -> str:
        """
        Generate a comprehensive summary report.
        
        Returns:
            String containing the summary report
        """
        if not self.results:
            return "No models have been evaluated."
        
        comparison_df = self.compare_models()
        statistical_tests = self.perform_statistical_tests()
        
        report = []
        report.append("="*60)
        report.append("MODEL EVALUATION SUMMARY REPORT")
        report.append("="*60)
        report.append("")
        
        # Best model
        best_model = comparison_df.index[0]
        best_rmse = comparison_df.loc[best_model, 'RMSE']
        best_r2 = comparison_df.loc[best_model, 'R²']
        
        report.append(f"BEST PERFORMING MODEL: {best_model}")
        report.append(f"  - RMSE: {best_rmse:.4f}")
        report.append(f"  - R²: {best_r2:.4f}")
        report.append("")
        
        # All models ranking
        report.append("MODEL RANKINGS (by RMSE):")
        for idx, (model, row) in enumerate(comparison_df.iterrows(), 1):
            report.append(f"  {idx}. {model}: RMSE={row['RMSE']:.4f}, R²={row['R²']:.4f}")
        report.append("")
        
        # Statistical significance
        if statistical_tests:
            report.append("STATISTICAL SIGNIFICANCE TESTS:")
            for comparison, results in statistical_tests.items():
                significance = "Significant" if results['significant'] else "Not significant"
                report.append(f"  {comparison}: p={results['p_value']:.4f} ({significance})")
        
        report.append("="*60)
        
        return "\n".join(report)
    
    def save_results(self, output_dir: str = "results"):
        """
        Save evaluation results to files.
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save comparison DataFrame
        comparison_df = self.compare_models()
        comparison_df.to_csv(output_path / "model_comparison.csv")
        
        # Save detailed results
        joblib.dump(self.results, output_path / "evaluation_results.pkl")
        joblib.dump(self.predictions, output_path / "predictions.pkl")
        
        # Save summary report
        summary = self.generate_summary_report()
        with open(output_path / "evaluation_summary.txt", 'w') as f:
            f.write(summary)
        
        logger.info(f"Saved evaluation results to {output_path}")
