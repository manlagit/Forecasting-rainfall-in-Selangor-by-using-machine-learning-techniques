from statsmodels.tsa.arima.model import ARIMA
import pickle
import yaml
from pathlib import Path
import pandas as pd


class ARIMAModel:
    def __init__(self, p: int = 1, d: int = 1, q: int = 1):
        """Initialize ARIMA model with order parameters.
        
        Args:
            p: Autoregressive order
            d: Differencing order
            q: Moving average order
        """
        self.p = p
        self.d = d
        self.q = q
        self.model_fit = None
        self._load_hyperparameters()

    def _load_hyperparameters(self):
        """Load hyperparameters from config file."""
        try:
            with open('config/hyperparameters.yaml') as f:
                config = yaml.safe_load(f)
                arima_config = config.get('arima', {})
                self.p_range = arima_config.get('p_range', [0, 1, 2, 3, 4, 5])
                self.d_range = arima_config.get('d_range', [0, 1, 2])
                self.q_range = arima_config.get('q_range', [0, 1, 2, 3, 4, 5])
        except Exception as e:
            print(f"Error loading hyperparameters: {e}")
            self.p_range = [0, 1, 2, 3, 4, 5]
            self.d_range = [0, 1, 2]
            self.q_range = [0, 1, 2, 3, 4, 5]

    def train(self, time_series: pd.Series) -> 'ARIMAModel':
        """Train ARIMA model on time series data.
        
        Args:
            time_series: Pandas Series with datetime index
            
        Returns:
            self: Trained model instance
        """
        try:
            self.model_fit = ARIMA(
                time_series, 
                order=(self.p, self.d, self.q)
            ).fit()
            return self
        except Exception as e:
            print(f"Error training ARIMA model: {e}")
            raise

    def predict(self, steps: int) -> pd.Series:
        """Generate predictions from trained model.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Forecasted values as pandas Series
        """
        if not self.model_fit:
            raise ValueError("Model not trained - call train() first")
        return self.model_fit.forecast(steps=steps)

    def save(self, path: str) -> None:
        """Save trained model to file.
        
        Args:
            path: File path to save model
        """
        if not self.model_fit:
            raise ValueError("Model not trained - call train() first")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model_fit, f)

    @staticmethod
    def load(path: str) -> 'ARIMAModel':
        """Load trained model from file.
        
        Args:
            path: File path to load model from
            
        Returns:
            ARIMAModel instance with loaded model
        """
        with open(path, 'rb') as f:
            model_fit = pickle.load(f)
        # Extract order parameters from model
        order = model_fit.model.order
        model = ARIMAModel(p=order[0], d=order[1], q=order[2])
        model.model_fit = model_fit
        return model
