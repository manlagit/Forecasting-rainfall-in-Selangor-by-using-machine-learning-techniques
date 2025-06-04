"""
Quick test script to verify the pipeline components.
Run this before executing the full pipeline.
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.data.data_loader import DataLoader
        print("✓ DataLoader imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import DataLoader: {e}")
        return False
    
    try:
        from src.features.preprocessing import DataPreprocessor
        print("✓ DataPreprocessor imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import DataPreprocessor: {e}")
        return False
    
    try:
        from src.models.model_trainer import ModelTrainer
        print("✓ ModelTrainer imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ModelTrainer: {e}")
        return False
    
    try:
        from src.evaluation.evaluate import ModelEvaluator
        print("✓ ModelEvaluator imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ModelEvaluator: {e}")
        return False
    
    try:
        from src.visualization.visualize import RainfallVisualizer
        print("✓ RainfallVisualizer imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import RainfallVisualizer: {e}")
        return False
    
    try:
        from src.utils.latex_generator import LaTeXReportGenerator
        print("✓ LaTeXReportGenerator imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import LaTeXReportGenerator: {e}")
        return False
    
    return True


def test_data_files():
    """Test if data files exist."""
    print("\nTesting data files...")
    
    data_files = [
        "data/raw/230731665812CCD_weekly1.csv",
        "data/raw/230731450378CCD_weekly2.csv"
    ]
    
    all_exist = True
    for file_path in data_files:
        if Path(file_path).exists():
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} missing")
            all_exist = False
    
    return all_exist


def test_config_files():
    """Test if configuration files exist."""
    print("\nTesting configuration files...")
    
    config_files = [
        "config/config.yaml",
        "config/hyperparameters.yaml"
    ]
    
    all_exist = True
    for file_path in config_files:
        if Path(file_path).exists():
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} missing")
            all_exist = False
    
    return all_exist


def test_directories():
    """Test if required directories exist."""
    print("\nTesting directories...")
    
    required_dirs = [
        "data/raw",
        "data/interim", 
        "data/processed",
        "models/saved_models",
        "models/scalers",
        "reports/figures",
        "src/data",
        "src/features",
        "src/models",
        "src/evaluation",
        "src/visualization",
        "src/utils",
        "logs"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ {dir_path} exists")
        else:
            print(f"✗ {dir_path} missing")
            all_exist = False
    
    return all_exist


def main():
    """Run all tests."""
    print("="*50)
    print("RAINFALL FORECASTING PROJECT - SYSTEM TEST")
    print("="*50)
    
    # Run tests
    imports_ok = test_imports()
    data_ok = test_data_files()
    config_ok = test_config_files()
    dirs_ok = test_directories()
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    print(f"Imports: {'✓ PASS' if imports_ok else '✗ FAIL'}")
    print(f"Data files: {'✓ PASS' if data_ok else '✗ FAIL'}")
    print(f"Config files: {'✓ PASS' if config_ok else '✗ FAIL'}")
    print(f"Directories: {'✓ PASS' if dirs_ok else '✗ FAIL'}")
    
    overall_success = all([imports_ok, data_ok, config_ok, dirs_ok])
    
    if overall_success:
        print("\n🎉 All tests passed! Ready to run the pipeline.")
        print("Execute: python main_pipeline.py")
        return True
    else:
        print("\n❌ Some tests failed. Please fix the issues before running the pipeline.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
