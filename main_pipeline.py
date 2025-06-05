"""
Main pipeline script for rainfall forecasting project.
Orchestrates the complete workflow from data loading to report generation.
"""

import sys
import logging
import traceback
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

# Import project modules
from src.data.data_loader import DataLoader  # noqa: E402
from src.features.build_features import FeatureBuilder  # noqa: E402
from src.features.preprocessing import DataPreprocessor  # noqa: E402
from src.models.model_trainer import ModelTrainer  # noqa: E402
from src.evaluation.evaluate import ModelEvaluator  # noqa: E402
from src.visualization.visualize import RainfallVisualizer  # noqa: E402
from src.utils.latex_generator import generate_latex_report  # noqa: E402


# Configure logging
def setup_logging():
    """Set up logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(
                log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d')}"
                          f"_{datetime.now().strftime('%H%M%S')}.log"
            ),
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
        
        # Step 2: Data Preprocessing (Cleaning)
        logger.info("Step 2: Preprocessing data (cleaning)...")
        preprocessor = DataPreprocessor()
        X, y = preprocessor.preprocess(df_raw)
        logger.info(
            f"Preprocessed data: {X.shape[1]} features, {len(X)} samples"
        )
        
        # Step 3: Feature Engineering
        logger.info("Step 3: Building features...")
        feature_builder = FeatureBuilder()
        # Combine features and target for feature engineering
        df_features = pd.concat([X, y], axis=1)
        df_features = feature_builder.build_features(df_features)
        
        # Separate features and target again
        X = df_features.drop(columns=['Precipitation_mm'])
        y = df_features['Precipitation_mm']
        
        # Remove non-numeric columns for model training
        X = X.select_dtypes(include=['number'])
        logger.info(f"Built {X.shape[1]} numeric features after engineering")
        
        # Step 4: Time-aware data split
        logger.info("Step 4: Splitting data...")
        test_size = 0.2
        split_index = int(len(X) * (1 - test_size))
        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]
        logger.info(f"Split data: Train={len(X_train)}, Test={len(X_test)}")
        
        # Step 5: Scaling
        logger.info("Step 5: Scaling data...")
        preprocessor.fit_scalers(X_train, y_train)
        X_train_scaled, y_train_scaled = preprocessor.transform(
            X_train, y_train
        )
        X_test_scaled, y_test_scaled = preprocessor.transform(
            X_test, y_test
        )
        preprocessor.save_scalers("models/scalers")
        
        # Step 6: Model Training
        logger.info("Step 6: Training models...")
        trainer = ModelTrainer()
        models = trainer.train_all_models(
            X_train_scaled, y_train_scaled
        )
        trainer.save_models()
        logger.info(f"Trained and saved {len(models)} models")
        
        # Prepare processed dataframe for visualization
        df_processed = df_features.copy()
        
        # Step 7: Model Evaluation
        logger.info("Step 7: Evaluating models...")
        evaluator = ModelEvaluator()
        
        # Evaluate each model
        for model_name, model in models.items():
            if model_name == 'ann':
                y_pred = model.predict(X_test_scaled).flatten()
            else:
                y_pred = model.predict(X_test_scaled)
            
            # Inverse transform predictions and actual values
            # Convert to numpy array before reshaping
            y_pred_array = y_pred.values if isinstance(y_pred, pd.Series) else y_pred
            y_test_array = y_test_scaled.values if isinstance(y_test_scaled, pd.Series) else y_test_scaled
            
            y_pred_inverse = preprocessor.target_scaler.inverse_transform(
                y_pred_array.reshape(-1, 1)
            ).flatten()
            y_test_inverse = preprocessor.target_scaler.inverse_transform(
                y_test_array.reshape(-1, 1)
            ).flatten()
            
            evaluator.evaluate_model(y_test_inverse, y_pred_inverse, model_name)
        
        # Generate comparison and save results
        comparison_df = evaluator.compare_models()
        evaluator.save_results("results")
        
        # Print summary
        print("\n" + evaluator.generate_summary_report())
        
        # Step 8: Generate Visualizations
        logger.info("Step 8: Generating visualizations...")
        visualizer = RainfallVisualizer()
        
        # Prepare data for visualization (include date and actual values)
        df_processed = pd.concat([
            df_raw.iloc[split_index:].reset_index(drop=True),
            pd.Series(y_test_inverse, name='Precipitation_actual'),
            pd.DataFrame({f"{model_name}_pred": evaluator.predictions[model_name]['pred'] 
                         for model_name in models.keys()}, index=X_test.index)
        ], axis=1)
        
        # Generate all plots
        plot_paths = visualizer.generate_all_plots(
            df_processed, 
            comparison_df, 
            evaluator.predictions,
            "reports/figures"
        )
        logger.info(f"Generated {len(plot_paths)} visualization plots")
        
        # Step 9: Generate LaTeX Report
        logger.info("Step 9: Generating LaTeX report...")
        
        # Get predictions for the best model
        best_model_name = comparison_df.index[0]
        best_model = models[best_model_name]
        y_pred_best = evaluator.predictions[best_model_name]['pred']
        
        # Generate additional visualizations for report
        residual_plot_path = os.path.join("reports/figures", f"{best_model_name}_residuals.png")
        visualizer.plot_residuals(
            y_test_inverse, 
            y_pred_best, 
            best_model_name,
            residual_plot_path
        )
        
        feature_importance_path = os.path.join("reports/figures", f"{best_model_name}_feature_importance.png")
        if hasattr(best_model, 'feature_importances_'):
            visualizer.plot_feature_importance(
                best_model, 
                X_train.columns, 
                best_model_name,
                feature_importance_path
            )
        else:
            logger.warning(f"Model {best_model_name} does not support feature importance visualization")
            feature_importance_path = "reports/figures/feature_importance_placeholder.png"
        
        # Generate the LaTeX report file
        report_path = generate_latex_report(
            comparison_df, 
            y_test_inverse, 
            y_pred_best, 
            best_model_name,
            feature_importance_path,
            residual_plot_path,
            "reports/latex"
        )
        logger.info(f"Generated LaTeX report at: {report_path}")
        
        # Pipeline completion
        logger.info("="*60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        
        # Print final summary
        print(f"\n{'='*60}")
        print("PIPELINE EXECUTION SUMMARY")
        print(f"{'='*60}")
        print(f"✓ Data loaded: {len(df_raw)} records")
        print(f"✓ Features engineered: {X.shape[1]} features")
        print(f"✓ Models trained: {len(models)}")
        print(
            f"✓ Best model: {comparison_df.index[0]} "
            f"(RMSE: {comparison_df.iloc[0]['RMSE']:.4f})"
        )
        print(f"✓ Plots generated: {len(plot_paths)}")
        print(f"✓ Report generated: {report_path}")
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
