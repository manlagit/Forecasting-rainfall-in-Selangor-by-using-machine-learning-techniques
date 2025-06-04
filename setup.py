"""
Setup script for the rainfall forecasting project.
Ensures all required directories exist and dependencies are ready.
"""

import subprocess
import sys
from pathlib import Path


def create_directories():
    """Create all required project directories."""
    directories = [
        "data/raw",
        "data/interim", 
        "data/processed",
        "models/saved_models",
        "models/scalers",
        "reports/figures",
        "reports/latex",
        "results",
        "logs"
    ]
    
    print("Creating project directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {directory}")


def install_dependencies():
    """Install required Python packages."""
    print("\nInstalling Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False


def main():
    """Run setup process."""
    print("="*50)
    print("RAINFALL FORECASTING PROJECT SETUP")
    print("="*50)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    deps_ok = install_dependencies()
    
    print("\n" + "="*50)
    print("SETUP SUMMARY")
    print("="*50)
    
    if deps_ok:
        print("✓ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run system test: python test_system.py")
        print("2. Execute pipeline: python main_pipeline.py")
    else:
        print("✗ Setup failed. Please check error messages above.")
    
    return deps_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
