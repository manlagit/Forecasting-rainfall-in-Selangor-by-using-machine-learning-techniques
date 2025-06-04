# Rainfall Forecasting Project Structure

```
Forecasting-rainfall-in-Selangor-by-using-machine-learning-techniques/
│
├── data/                        # Data directory
│   ├── raw/                    # Original, immutable data
│   │   ├── 230731665812CCD_weekly1.csv
│   │   └── 230731450378CCD_weekly2.csv
│   ├── interim/                # Intermediate data that has been transformed
│   └── processed/              # Final, canonical data sets for modeling
│       ├── train_data.csv
│       ├── test_data.csv
│       └── scaled_data.pkl
│
├── models/                     # Trained and serialized models
│   ├── saved_models/          # Serialized model files
│   │   ├── ann_model.h5
│   │   ├── mlr_model.pkl
│   │   ├── knn_model.pkl
│   │   ├── rf_model.pkl
│   │   ├── xgb_model.pkl
│   │   └── arima_model.pkl
│   └── scalers/               # Saved preprocessing scalers
│       ├── feature_scaler.pkl
│       └── target_scaler.pkl
│
├── src/                       # Source code for the project
│   ├── __init__.py
│   ├── data/                  # Scripts to download or generate data
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── data_validator.py
│   ├── features/              # Scripts to turn raw data into features
│   │   ├── __init__.py
│   │   ├── build_features.py
│   │   └── feature_engineering.py
│   ├── models/                # Scripts to train models and make predictions
│   │   ├── __init__.py
│   │   ├── train_models.py
│   │   ├── predict_models.py
│   │   ├── ann_model.py
│   │   ├── mlr_model.py
│   │   ├── knn_model.py
│   │   ├── rf_model.py
│   │   ├── xgb_model.py
│   │   └── arima_model.py
│   ├── evaluation/            # Model evaluation scripts
│   │   ├── __init__.py
│   │   └── evaluate.py
│   ├── visualization/         # Scripts to create visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   └── utils/                 # Utility functions
│       ├── __init__.py
│       ├── logger.py
│       └── helpers.py
│
├── notebooks/                 # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_results_analysis.ipynb
│
├── reports/                   # Generated analysis and reports
│   ├── figures/              # Generated graphics and figures
│   │   ├── model_comparison.png
│   │   ├── time_series_plot.png
│   │   ├── residual_plots.png
│   │   └── feature_importance.png
│   └── latex/                # LaTeX report files
│       ├── main_report.tex
│       ├── bibliography.bib
│       └── main_report.pdf
│
├── tests/                     # Unit tests
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_features.py
│   └── test_models.py
│
├── config/                    # Configuration files
│   ├── config.yaml
│   └── hyperparameters.yaml
│
├── logs/                      # Log files
│   └── pipeline.log
│
├── main_pipeline.py          # Master execution script
├── requirements.txt          # Project dependencies
├── setup.py                  # Setup script
├── README.md                 # Project documentation
├── .gitignore               # Git ignore file
└── LICENSE                  # License file
```

## Directory Descriptions

### `data/`
- **raw/**: Original dataset files (CSV)
- **interim/**: Intermediate processed data
- **processed/**: Final data ready for modeling

### `models/`
- **saved_models/**: Serialized trained models (.pkl, .h5)
- **scalers/**: Saved preprocessing scalers

### `src/`
- **data/**: Data loading and validation modules
- **features/**: Feature engineering and preprocessing
- **models/**: Model implementations and training scripts
- **evaluation/**: Model evaluation metrics and comparison
- **visualization/**: Plotting and visualization functions
- **utils/**: Helper functions and utilities

### `notebooks/`
Jupyter notebooks for exploratory data analysis and experimentation

### `reports/`
- **figures/**: Generated plots and visualizations
- **latex/**: LaTeX report source files and compiled PDF

### `tests/`
Unit tests for all modules

### `config/`
Configuration files for the project

### `logs/`
Execution and error logs
