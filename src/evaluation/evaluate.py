"""
Model Evaluation Framework
Standardized evaluation and comparison of all models.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from pathlib import Path
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """
    Comprehensive model evaluation and comparison framework.
    """
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize model evaluator.
        
        Args:
            results_dir: Directory to save evaluation results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Store evaluation results
        self.evaluation_results = {}
        self.predictions = {}
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      model_name: str) -> Dict[str, float]:
        """
        Evaluate a single model using standard metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Ensure arrays are 1D
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Additional metrics
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        residuals = y_true - y_pred
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'Mean_Residual': np.mean(residuals),
            'Std_Residual': np.std(residuals)
        }
        
        # Store results
        self.evaluation_results[model_name] = metrics
        self.predictions[model_name] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'residuals': residuals
        }
        
        self.logger.info(f"Evaluated {model_name} - RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        return metrics
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all evaluated models and rank them.
        
        Returns:
            DataFrame with model comparison results
        """
        if not self.evaluation_results:
            raise ValueError("No models have been evaluated yet")
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(self.evaluation_results).T
        
        # Sort by RMSE (ascending - lower is better)
        comparison_df = comparison_df.sort_values('RMSE')
        
        # Add ranking
        comparison_df['Rank'] = range(1, len(comparison_df) + 1)
        
        # Reorder columns
        columns_order = ['Rank', 'RMSE', 'MAE', 'R2', 'MSE', 'MAPE', 
                        'Mean_Residual', 'Std_Residual']
        comparison_df = comparison_df[columns_order]
        
        self.logger.info("Model comparison completed")
        self.logger.info(f"Best model: {comparison_df.index[0]} (RMSE: {comparison_df.iloc[0]['RMSE']:.4f})")
        
        return comparison_df
    
    def perform_statistical_tests(self) -> Dict[str, Any]:
        """
        Perform statistical significance tests between models.
        
        Returns:
            Dictionary with statistical test results
        """
        if len(self.predictions) < 2:
            self.logger.warning("Need at least 2 models for statistical testing")
            return {}
        
        statistical_results = {}
        model_names = list(self.predictions.keys())
        
        # Perform paired t-tests between all model pairs
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                residuals1 = self.predictions[model1]['residuals']
                residuals2 = self.predictions[model2]['residuals']
                
                # Paired t-test
                t_stat, p_value = stats.ttest_rel(
                    np.abs(residuals1), 
                    np.abs(residuals2)
                )
                
                test_key = f"{model1}_vs_{model2}"
                statistical_results[test_key] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        self.logger.info(f"Performed {len(statistical_results)} paired t-tests")
        return statistical_results
    
    def generate_summary_report(self) -> str:
        """
        Generate a comprehensive summary report.
        
        Returns:
            Formatted summary report string
        """
        if not self.evaluation_results:
            return "No evaluation results available"
        
        comparison_df = self.compare_models()
        statistical_tests = self.perform_statistical_tests()
        
        report = f"""
{'='*80}
RAINFALL FORECASTING MODEL EVALUATION SUMMARY
{'='*80}

MODEL PERFORMANCE RANKING:
{'-'*50}
"""
        
        for idx, (model_name, row) in enumerate(comparison_df.iterrows(), 1):
            report += f"{idx}. {model_name.upper()}\n"
            report += f"   RMSE: {row['RMSE']:.4f} | MAE: {row['MAE']:.4f} | R²: {row['R2']:.4f}\n"
            report += f"   MAPE: {row['MAPE']:.2f}% | Mean Residual: {row['Mean_Residual']:.4f}\n\n"
        
        # Best model details
        best_model = comparison_df.index[0]
        report += f"🏆 BEST PERFORMING MODEL: {best_model.upper()}\n"
        report += f"   Root Mean Square Error: {comparison_df.iloc[0]['RMSE']:.4f}\n"
        report += f"   Coefficient of Determination: {comparison_df.iloc[0]['R2']:.4f}\n"
        report += f"   Mean Absolute Percentage Error: {comparison_df.iloc[0]['MAPE']:.2f}%\n\n"
        
        # Statistical significance
        if statistical_tests:
            report += f"STATISTICAL SIGNIFICANCE TESTS:\n{'-'*50}\n"
            for test_name, results in statistical_tests.items():
                significance = "✓ SIGNIFICANT" if results['significant'] else "✗ NOT SIGNIFICANT"
                report += f"{test_name}: p-value = {results['p_value']:.4f} ({significance})\n"
        
        report += f"\n{'='*80}"
        
        return report
    
    def save_results(self) -> None:
        """Save evaluation results to files."""
        # Save evaluation metrics
        metrics_df = pd.DataFrame(self.evaluation_results).T
        metrics_df.to_csv(self.results_dir / "evaluation_metrics.csv")
        
        # Save comparison results
        comparison_df = self.compare_models()
        comparison_df.to_csv(self.results_dir / "model_comparison.csv")
        
        # Save predictions
        for model_name, pred_data in self.predictions.items():
            pred_df = pd.DataFrame({
                'y_true': pred_data['y_true'],
                'y_pred': pred_data['y_pred'],
                'residuals': pred_data['residuals']
            })
            pred_df.to_csv(self.results_dir / f"{model_name}_predictions.csv", index=False)
        
        # Save statistical tests
        statistical_tests = self.perform_statistical_tests()
        with open(self.results_dir / "statistical_tests.json", 'w') as f:
            json.dump(statistical_tests, f, indent=2)
        
        # Save summary report
        summary_report = self.generate_summary_report()
        with open(self.results_dir / "summary_report.txt", 'w') as f:
            f.write(summary_report)
        
        self.logger.info(f"Evaluation results saved to {self.results_dir}")
    
    def calculate_feature_importance(self, models: Dict[str, Any], 
                                   feature_names: List[str]) -> Dict[str, np.ndarray]:
        """
        Calculate feature importance for applicable models.
        
        Args:
            models: Dictionary of trained models
            feature_names: List of feature names
            
        Returns:
            Dictionary of feature importance arrays
        """
        importance_results = {}
        
        # Random Forest feature importance
        if 'random_forest' in models:
            rf_importance = models['random_forest'].feature_importances_
            importance_results['random_forest'] = rf_importance
            self.logger.info("Calculated Random Forest feature importance")
        
        # XGBoost feature importance
        if 'xgboost' in models:
            xgb_importance = models['xgboost'].feature_importances_
            importance_results['xgboost'] = xgb_importance
            self.logger.info("Calculated XGBoost feature importance")
        
        # Save feature importance
        if importance_results:
            importance_df = pd.DataFrame(importance_results, index=feature_names)
            importance_df.to_csv(self.results_dir / "feature_importance.csv")
            self.logger.info("Feature importance results saved")
        
        return importance_results
    
    def load_results(self) -> None:
        """Load previously saved evaluation results."""
        try:
            # Load evaluation metrics
            metrics_file = self.results_dir / "evaluation_metrics.csv"
            if metrics_file.exists():
                metrics_df = pd.read_csv(metrics_file, index_col=0)
                self.evaluation_results = metrics_df.to_dict('index')
            
            # Load predictions
            for pred_file in self.results_dir.glob("*_predictions.csv"):
                model_name = pred_file.stem.replace("_predictions", "")
                pred_df = pd.read_csv(pred_file)
                self.predictions[model_name] = {
                    'y_true': pred_df['y_true'].values,
                    'y_pred': pred_df['y_pred'].values,
                    'residuals': pred_df['residuals'].values
                }
            
            self.logger.info("Evaluation results loaded successfully")
            
        except Exception as e:
            self.logger.warning(f"Could not load previous results: {e}")
