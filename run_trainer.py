import os
os.chdir("d:/Forecasting-rainfall-in-Selangor-by-using-machine-learning-techniques")

from src.models.train_models import ModelTrainer

trainer = ModelTrainer()

# Load data (replace with actual data loading if needed)
import pandas as pd
import numpy as np
X = pd.DataFrame(np.random.rand(100, 5))
y = pd.Series(np.random.rand(100))

# Train models
trainer.train_model('ann', X, y)
trainer.train_model('knn', X, y)
trainer.train_model('rf', X, y)
trainer.train_model('xgb', X, y)
trainer.train_model('mlr', X, y)

print("Model training completed.")
