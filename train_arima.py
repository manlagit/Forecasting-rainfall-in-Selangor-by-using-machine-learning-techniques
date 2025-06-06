import pandas as pd
from src.data.data_loader import DataLoader
from src.models.arima_model import ARIMAModel
from src.evaluation.evaluate import ModelEvaluator
import yaml

# Load configuration
with open("config/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Load data
data_loader = DataLoader()
df = data_loader.load_and_validate_data()

# Prepare time series data
time_series = df.set_index('Date')['Precipitation_mm']

# Train ARIMA model
arima_model = ARIMAModel(p=2, d=1, q=1)
arima_model.train(time_series)

# Make predictions (example: predict 30 steps)
predictions = arima_model.predict(steps=30)

# Evaluate model
model_evaluator = ModelEvaluator()
eval_metrics = model_evaluator.evaluate_regression(time_series[-len(predictions):], predictions, "arima")

# Load existing model comparison data
try:
    model_comparison = pd.read_csv("results/model_comparison.csv")
except FileNotFoundError:
    model_comparison = pd.DataFrame(columns=[',RMSE,MAE,R2'])

# Add ARIMA results
arima_results = pd.DataFrame({
    'RMSE': [eval_metrics['RMSE']],
    'MAE': [eval_metrics['MAE']],
    'R2': [eval_metrics['R2']]
}, index=['arima'])

model_comparison = pd.concat([model_comparison, arima_results])
model_comparison.to_csv("results/model_comparison.csv", index=True)

print("ARIMA model trained and evaluated. Results added to model_comparison.csv")
