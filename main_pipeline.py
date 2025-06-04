"""
Main pipeline script for rainfall forecasting project.
Orchestrates the complete workflow from data loading to report generation.
"""

import logging
import sys
import traceback
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

# Import project modules
from src.data.data_loader import DataLoader
from src.features.preprocessing import DataPreprocessor
from src.models.model_trainer import ModelTrainer
from src.evaluation.evaluate import ModelEvaluator
from src.visualization.visualize import RainfallVisualizer
from src.utils.latex_generator import LaTeXReportGenerator

# Configure logging
def setup_logging():
    """Set up logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def main():
    """Main pipeline execution function."""
    logger = setup_logging()
    logger.info("="*60)
    logger.info("RAINFALL FORECASTING PIPELINE STARTED")
    logger.info("="*60)
    
    try:
        # Step 1: Data Loading and Validation
        logger.info("Step 1: Loading and validating data...")
        data_loader = DataLoader()
        df_raw = data_loader.load_and_validate_data()
        logger.info(f"Loaded dataset with {len(df_raw)} records")
        
        # Step 2: Data Preprocessing
        logger.info("Step 2: Preprocessing data...")
        preprocessor = DataPreprocessor()
        X_normalized, y_normalized, df_processed = preprocessor.preprocess_data(df_raw)
        logger.info(f"Processed data: {X_normalized.shape[1]} features, {len(X_normalized)} samples")
        
        # Step 3: Model Training
        logger.info("Step 3: Training models...")
        trainer = ModelTrainer()
        X_train, X_test, y_train, y_test = trainer.split_data(X_normalized, y_normalized)
        
        # Extract dates for ARIMA
        dates_train = df_processed['Date'].iloc[:len(X_train)]
        
        # Train all models
        models = trainer.train_all_models(X_train, y_train, dates_train)
        trainer.save_models()
        logger.info(f"Trained and saved {len(models)} models")
        
        # Step 4: Model Evaluation
        logger.info("Step 4: Evaluating models...")
        evaluator = ModelEvaluator()
        
        # Evaluate each model
        for model_name, model in models.items():
            if model_name == 'linear_regression':
                selected_features = trainer.best_params['linear_regression']['selected_features']
                y_pred = model.predict(X_test[selected_features])
            elif model_name == 'ann':
                y_pred = model.predict(X_test).flatten()
            elif model_name == 'arima':
                # Skip ARIMA for now due to complexity
                continue
            else:
                y_pred = model.predict(X_test)
            
            # Evaluate model
            evaluator.evaluate_model(y_test.values, y_pred, model_name)
        
        # Generate comparison and save results
        comparison_df = evaluator.compare_models()
        evaluator.save_results()
        
        # Print summary
        print("\n" + evaluator.generate_summary_report())
        
        # Step 5: Generate Visualizations
        logger.info("Step 5: Generating visualizations...")
        visualizer = RainfallVisualizer()
        
        # Generate all plots
        plot_paths = visualizer.generate_all_plots(
            df_processed, 
            comparison_df, 
            evaluator.predictions
        )
        logger.info(f"Generated {len(plot_paths)} visualization plots")
        
        # Step 6: Generate LaTeX Report
        logger.info("Step 6: Generating LaTeX report...")
        report_generator = LaTeXReportGenerator()
        
        # Generate report
        latex_file = report_generator.generate_complete_report(
            comparison_df, 
            plot_paths
        )
        
        # Compile to PDF
        pdf_file = report_generator.compile_pdf(latex_file)
        
        if pdf_file:
            logger.info(f"Successfully generated PDF report: {pdf_file}")
        else:
            logger.warning("PDF compilation failed, but LaTeX file is available")
        
        # Pipeline completion
        logger.info("="*60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        
        # Print final summary
        print(f"\n{'='*60}")
        print("PIPELINE EXECUTION SUMMARY")
        print(f"{'='*60}")
        print(f"✓ Data loaded: {len(df_raw)} records")
        print(f"✓ Features engineered: {X_normalized.shape[1]} features")
        print(f"✓ Models trained: {len(models)}")
        print(f"✓ Best model: {comparison_df.index[0]} (RMSE: {comparison_df.iloc[0]['RMSE']:.4f})")
        print(f"✓ Plots generated: {len(plot_paths)}")
        print(f"✓ Report generated: {latex_file}")
        if pdf_file:
            print(f"✓ PDF compiled: {pdf_file}")
        print(f"{'='*60}")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Pipeline completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Pipeline failed. Check logs for details.")
        sys.exit(1)
