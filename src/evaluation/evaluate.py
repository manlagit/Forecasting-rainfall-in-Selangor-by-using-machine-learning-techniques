"""
Model Evaluation Module
Evaluates machine learning models for rainfall forecasting.
"""

import logging
import pandas as pd
import os
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ModelEvaluator:
    """
    Evaluates machine learning models for rainfall forecasting.
    """
    
    def __init__(self):
        """
        Initialize the evaluator.
        """
        self.logger = logging.getLogger(__name__)
        self.results = {}
        self.predictions = {}
        self.logger.info("Initialized ModelEvaluator")
        
    def evaluate_model(self, y_true: pd.Series, y_pred: pd.Series, model_name: str):
        """
        Evaluate a model and store the results.
        """
        self.logger.info(f"Evaluating {model_name} model...")
        
        # Calculate metrics
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Store results
        self.results[model_name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        # Store predictions
        self.predictions[model_name] = {
            'true': y_true,
            'pred': y_pred
        }
        
        return self.results[model_name]
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all evaluated models and return a DataFrame.
        """
        if not self.results:
            self.logger.warning("No models evaluated for comparison")
            return pd.DataFrame()
            
        # Create DataFrame from results
        comparison_df = pd.DataFrame.from_dict(
            self.results, 
            orient='index',
            columns=['RMSE', 'MAE', 'R2']
        )
        
        # Sort by RMSE (ascending is better)
        comparison_df = comparison_df.sort_values('RMSE')
        
        return comparison_df
    
    def save_results(self, output_dir: str = "results"):
        """
        Save evaluation results to CSV files.
        """
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save comparison results
        comparison_df = self.compare_models()
        if not comparison_df.empty:
            comparison_path = os.path.join(output_dir, "model_comparison.csv")
            comparison_df.to_csv(comparison_path)
            self.logger.info(f"Saved model comparison to {comparison_path}")
        
        # Save individual model results
        for model_name, metrics in self.results.items():
            model_path = os.path.join(output_dir, f"{model_name}_metrics.csv")
            pd.DataFrame([metrics]).to_csv(model_path, index=False)
            self.logger.info(f"Saved {model_name} metrics to {model_path}")
            
        return True
    
    def generate_summary_report(self) -> str:
        """
        Generate a summary report of model performance.
        """
        if not self.results:
            return "No models evaluated yet."
            
        comparison_df = self.compare_models()
        report = "MODEL PERFORMANCE SUMMARY\n"
        report += "=" * 40 + "\n"
        report += comparison_df.to_string()
        report += "\n" + "=" * 40
        report += f"\nBest model: {comparison_df.index[0]} (RMSE: {comparison_df.iloc[0]['RMSE']:.4f})"
        
        return report
