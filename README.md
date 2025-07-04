# Rainfall Forecasting in Selangor using Machine Learning Techniques

## Project Overview
This project implements multiple machine learning models to forecast rainfall (precipitation in mm) in Selangor, Malaysia. The system compares various algorithms including Artificial Neural Networks (ANN), Multiple Linear Regression (MLR), K-Nearest Neighbors (KNN), Random Forest (RF), Gradient Boosting (XGBoost), and ARIMA models.

## Features
- Automated data preprocessing pipeline with outlier detection and feature engineering
- Implementation of 6 different machine learning models
- Hyperparameter tuning using GridSearchCV and Optuna
- Automated LaTeX report generation with PGFPlots integration
- Comprehensive model evaluation and comparison framework

## Installation

### Prerequisites
- Python 3.8 or higher
- LaTeX distribution (for report generation)
- Git

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/Forecasting-rainfall-in-Selangor.git
cd Forecasting-rainfall-in-Selangor-by-using-machine-learning-techniques
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Note on LaTeX Installation
For report generation, you'll need a LaTeX distribution installed:
- Windows: [MiKTeX](https://miktex.org/)
- macOS: [MacTeX](https://www.tug.org/mactex/)
- Linux: `sudo apt-get install texlive-full` (Ubuntu/Debian)


## Usage

### Quick System Test
Before running the pipeline, test if everything is properly set up:
```bash
python test_system.py
```

### Running the Complete Pipeline
Execute the main pipeline script:
```bash
python main_pipeline.py
```

This will:
1. Load and validate the data
2. Perform preprocessing and feature engineering
3. Train all models with hyperparameter tuning
4. Evaluate model performance
5. Generate visualizations
6. Create a LaTeX report
7. Compile the PDF report

### Running Individual Components
```python
# Test system setup
python test_system.py

# Data preprocessing only
from src.data.data_loader import DataLoader
loader = DataLoader()
data = loader.load_and_validate_data()

# Train specific models
from src.models.model_trainer import ModelTrainer
trainer = ModelTrainer()

# Generate visualizations
from src.visualization.visualize import RainfallVisualizer
visualizer = RainfallVisualizer()
```

### Expected Output
After successful execution, you will find:
- **Trained models**: `models/saved_models/`
- **Evaluation results**: `results/`
- **Visualizations**: `reports/figures/`
- **LaTeX report**: `reports/latex/rainfall_forecasting_report.tex`
- **PDF report**: `reports/latex/rainfall_forecasting_report.pdf`
- **Logs**: `logs/`

## Project Structure
See `PROJECT_STRUCTURE.md` for detailed directory layout.

## Data
- **Input**: Two CSV files containing weekly weather data (2012-2021)
  - `230731665812CCD_weekly1.csv` (470 records)
  - `230731450378CCD_weekly2.csv` (validation duplicate)
- **Features**: Temperature, Relative Humidity, Wind Speed
- **Target**: Precipitation (mm)

## Models Implemented
1. **Artificial Neural Networks (ANN)**: Deep learning approach with configurable architecture
2. **Multiple Linear Regression (MLR)**: Baseline statistical model
3. **K-Nearest Neighbors (KNN)**: Instance-based learning
4. **Random Forest (RF)**: Ensemble tree-based method
5. **Gradient Boosting (XGBoost)**: Advanced boosting algorithm
6. **ARIMA**: Time series forecasting model

## Results
Model performance metrics and comparisons are automatically generated in the reports folder.

## Contributing
Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Authors
- Your Name - Initial work

## Acknowledgments
- Weather data provided by [Data Source]
- Academic supervision by [Supervisor Name]
