"""
System Test Script
Verifies that all components of the rainfall forecasting pipeline are properly set up.
"""

import sys
import logging
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

# Import project modules
from src.data.data_loader import DataLoader
from src.features.build_features import FeatureBuilder
from src.features.preprocessing import DataPreprocessor
from src.models.model_trainer import ModelTrainer
from src.evaluation.evaluate import ModelEvaluator
from src.visualization.visualize import RainfallVisualizer
from src.utils.latex_generator import generate_latex_report

def test_system():
    """Run system tests to verify pipeline components"""
    print("="*60)
    print("RAINFALL FORECASTING SYSTEM TEST")
    print("="*60)
    
    # Test data loading
    try:
        print("\nTesting data loading...")
        loader = DataLoader()
        sample_data = pd.DataFrame({
            'Date': pd.date_range(start='2020-01-01', periods=10, freq='W'),
            'Temp_avg': [28.5, 29.0, 29.5, 30.0, 29.8, 29.5, 29.0, 28.5, 28.0, 27.5],
            'Relative_Humidity': [80, 82, 85, 83, 81, 79, 78, 77, 76, 75],
            'Wind_kmh': [10, 12, 15, 14, 13, 11, 10, 9, 8, 7],
            'Precipitation_mm': [5.2, 6.1, 7.5, 8.2, 4.8, 3.5, 2.1, 1.5, 0.8, 0.2]
        })
        loader.save_sample_data(sample_data)
        loaded_data = loader.load_and_validate_data()
        print("✓ Data loading test passed")
    except Exception as e:
        print(f"❌ Data loading test failed: {str(e)}")
        return False
    
    # Test preprocessing
    try:
        print("\nTesting data preprocessing...")
        preprocessor = DataPreprocessor()
        X, y = preprocessor.preprocess(loaded_data.copy())
        print(f"Preprocessed data: {X.shape[0]} samples, {X.shape[1]} features")
        print("✓ Data preprocessing test passed")
    except Exception as e:
        print(f"❌ Data preprocessing test failed: {str(e)}")
        return False
    
    # Test feature engineering
    try:
        print("\nTesting feature engineering...")
        feature_builder = FeatureBuilder()
        df_features = feature_builder.build_features(pd.concat([X, y], axis=1))
        print(f"Engineered features: {df_features.shape[1]} total features")
        print("✓ Feature engineering test passed")
    except Exception as e:
        print(f"❌ Feature engineering test failed: {str(e)}")
        return False
    
    # Test model training
    try:
        print("\nTesting model training...")
        trainer = ModelTrainer()
        X_train = df_features.drop(columns=['Precipitation_mm']).select_dtypes(include=['number'])
        y_train = df_features['Precipitation_mm']
        models = trainer.train_all_models(X_train, y_train)
        print(f"Trained {len(models)} models")
        trainer.save_models("models/test_models")
        print("✓ Model training test passed")
    except Exception as e:
        print(f"❌ Model training test failed: {str(e)}")
        return False
    
    # Test model evaluation
    try:
        print("\nTesting model evaluation...")
        evaluator = ModelEvaluator()
        for model_name, model in models.items():
            y_pred = model.predict(X_train)
            evaluator.evaluate_model(y_train, y_pred, model_name)
        comparison_df = evaluator.compare_models()
        evaluator.save_results("results/test_results")
        print(comparison_df)
        print("✓ Model evaluation test passed")
    except Exception as e:
        print(f"❌ Model evaluation test failed: {str(e)}")
        return False
    
    # Test visualization
    try:
        print("\nTesting visualization...")
        visualizer = RainfallVisualizer()
        plot_paths = visualizer.generate_all_plots(
            df_features, 
            comparison_df, 
            evaluator.predictions,
            "reports/test_figures"
        )
        print(f"Generated {len(plot_paths)} visualizations")
        print("✓ Visualization test passed")
    except Exception as e:
        print(f"❌ Visualization test failed: {str(e)}")
        return False
    
    # Test report generation
    try:
        print("\nTesting report generation...")
        if models:
            model_name = list(models.keys())[0]
            # Use actual values as predictions for testing
            y_pred = y_train.values
        else:
            model_name = "dummy_model"
            y_pred = np.zeros_like(y_train.values)
            
        report_path = generate_latex_report(
            comparison_df, 
            y_train, 
            y_pred, 
            model_name,
            "reports/test_latex"
        )
        print(f"Generated LaTeX report at: {report_path}")
        print("✓ Report generation test passed")
    except Exception as e:
        print(f"❌ Report generation test failed: {str(e)}")
        return False
    
    print("\n" + "="*60)
    print("SYSTEM TEST COMPLETED SUCCESSFULLY!")
    print("="*60)
    return True

if __name__ == "__main__":
    success = test_system()
    if not success:
        print("\n❌ System test failed. Check logs for details.")
        sys.exit(1)
    else:
        print("\n✅ System test passed successfully!")
        sys.exit(0)
