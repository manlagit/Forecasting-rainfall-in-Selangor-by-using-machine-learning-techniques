"""
Main pipeline script for rainfall forecasting project.
Orchestrates the complete workflow from data loading to report generation.
"""

import sys
import logging
import traceback
import os
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

        # Create binary classification target (rain/no-rain)
        rain_threshold = 0.1  # 0.1mm precipitation threshold
        y_test_binary = (y_test > rain_threshold).astype(int)

        # Evaluate each model
        for model_name, model in models.items():
            if model_name == 'ann':
                y_pred = model.predict(X_test_scaled).flatten()
            else:
                y_pred = model.predict(X_test_scaled)
            
            # Create binary predictions
            y_pred_binary = (y_pred > rain_threshold).astype(int)
            
            # Evaluate as classification problem
            evaluator.evaluate_classification(
                y_test_binary, 
                y_pred_binary, 
                model_name
            )

        # Generate comparison and save results
        comparison_df = evaluator.compare_classification_models()
        evaluator.save_results("results")

        # Print summary
        print("\n" + evaluator.generate_classification_summary())
        
        # Step 8: Generate Classification Visualizations
        logger.info("Step 8: Generating classification visualizations...")
        visualizer = RainfallVisualizer()
        
        # Get the best model name
        best_model_name = comparison_df.index[0]
        best_model = models[best_model_name]
        
        # Generate ROC curve for all models
        roc_curve_path = os.path.join("reports/figures", "roc_curve_comparison.png")
        visualizer.plot_roc_curve(
            evaluator.classification_results, 
            roc_curve_path
        )
        logger.info(f"Generated ROC curve at: {roc_curve_path}")

        # Generate confusion matrix for best model
        confusion_matrix_path = os.path.join(
            "reports/figures", 
            f"{best_model_name}_confusion_matrix.png"
        )
        visualizer.plot_confusion_matrix(
            evaluator.classification_results[best_model_name]['y_true'],
            evaluator.classification_results[best_model_name]['y_pred'],
            best_model_name,
            confusion_matrix_path
        )
        logger.info(f"Generated confusion matrix at: {confusion_matrix_path}")

        # Generate feature importance for best model
        feature_importance_path = os.path.join(
            "reports/figures", 
            f"{best_model_name}_feature_importance.png"
        )
        if hasattr(best_model, 'feature_importances_'):
            visualizer.plot_feature_importance(
                best_model, 
                X_train.columns, 
                best_model_name,
                feature_importance_path
            )
            logger.info(f"Generated feature importance plot at: {feature_importance_path}")
        else:
            logger.warning(f"Model {best_model_name} does not support feature importance visualization")
            feature_importance_path = "reports/figures/feature_importance_placeholder.png"
        
        # Step 9: Generate Classification Report
        logger.info("Step 9: Generating classification report...")
        report_path = generate_latex_report(
            comparison_df, 
            evaluator.classification_results[best_model_name]['y_true'],
            evaluator.classification_results[best_model_name]['y_pred'],
            best_model_name,
            feature_importance_path,
            roc_curve_path,
            confusion_matrix_path,
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
            f"✓ Best model: {best_model_name} "
            f"(AUC: {comparison_df.loc[best_model_name, 'AUC']:.4f})"
        )
        print(f"✓ Visualizations generated: ROC curve, confusion matrix, feature importance")
        print(f"✓ Classification report generated: {report_path}")
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
