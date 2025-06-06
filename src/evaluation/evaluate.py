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
        
    def evaluate_classification(self, y_true: pd.Series, y_pred: pd.Series, model_name: str):
        """
        Evaluate a classification model and store the results.
        """
        self.logger.info(f"Evaluating {model_name} classification model...")
        
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score, confusion_matrix
        )
        
        # Calculate metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        try:
            # Try to calculate AUC if we have probabilities
            auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auc = 0.5  # Default to random chance if only one class present
        
        # Store results
        self.results[model_name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'AUC': auc
        }
        
        # Store predictions
        self.predictions[model_name] = {
            'y_true': y_true,
            'y_pred': y_pred
        }
        
        return self.results[model_name]
    
    def evaluate_regression(self, y_true: pd.Series, y_pred: pd.Series, model_name: str):
        """
        Evaluate a regression model and store the results.
        """
        self.logger.info(f"Evaluating {model_name} regression model...")
        
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
            'y_true': y_true,
            'y_pred': y_pred
        }
        
        return self.results[model_name]
    
    def compare_classification_models(self) -> pd.DataFrame:
        """
        Compare all evaluated classification models and return a DataFrame.
        """
        if not self.results:
            self.logger.warning("No models evaluated for classification comparison")
            return pd.DataFrame()
            
        # Create DataFrame from results
        comparison_df = pd.DataFrame.from_dict(
            self.results, 
            orient='index',
            columns=['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
        )
        
        # Sort by AUC (descending is better)
        comparison_df = comparison_df.sort_values('AUC', ascending=False)
        
        return comparison_df
    
    def compare_regression_models(self) -> pd.DataFrame:
        """
        Compare all evaluated regression models and return a DataFrame.
        """
        if not self.results:
            self.logger.warning("No models evaluated for regression comparison")
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
        
        # Save individual model results
        for model_name, metrics in self.results.items():
            model_path = os.path.join(output_dir, f"{model_name}_metrics.csv")
            pd.DataFrame([metrics]).to_csv(model_path, index=False)
            self.logger.info(f"Saved {model_name} metrics to {model_path}")
            
        return True
    
    def generate_classification_summary(self) -> str:
        """
        Generate a classification summary report of model performance.
        """
        if not self.results:
            return "No classification models evaluated yet."
            
        comparison_df = self.compare_classification_models()
        report = "CLASSIFICATION MODEL PERFORMANCE SUMMARY\n"
        report += "=" * 60 + "\n"
        report += comparison_df.to_string()
        report += "\n" + "=" * 60
        report += f"\nBest model: {comparison_df.index[0]} (AUC: {comparison_df.iloc[0]['AUC']:.4f})"
        
        return report
    
    def generate_regression_summary(self) -> str:
        """
        Generate a regression summary report of model performance.
        """
        if not self.results:
            return "No regression models evaluated yet."
            
        comparison_df = self.compare_regression_models()
        report = "REGRESSION MODEL PERFORMANCE SUMMARY\n"
        report += "=" * 60 + "\n"
        report += comparison_df.to_string()
        report += "\n" + "=" * 60
        report += f"\nBest model: {comparison_df.index[0]} (RMSE: {comparison_df.iloc[0]['RMSE']:.4f})"
        
        return report
