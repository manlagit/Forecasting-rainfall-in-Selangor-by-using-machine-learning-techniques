import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import itertools
import warnings

def train_arima(y_train, p_range=[0,1,2], d_range=[0,1], q_range=[0,1]):
    """
    Trains an ARIMA model by searching for the best (p,d,q) order
    based on AIC.

    Args:
        y_train (pd.Series or np.array): Training time series data.
        p_range (list): Range of p values to test.
        d_range (list): Range of d values to test.
        q_range (list): Range of q values to test.

    Returns:
        statsmodels.tsa.arima.ARIMAResultsWrapper: The fitted ARIMA model.
                                                   Returns None if no model could be fit.
    """
    warnings.filterwarnings("ignore") # Suppress convergence warnings during grid search
    best_aic = float('inf')
    best_order = None
    best_model_results = None

    # Ensure y_train is a pandas Series with a DatetimeIndex if it's not already
    # This is often helpful for time series models in statsmodels
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train)
    
    # If y_train does not have a proper time index, create one
    if not isinstance(y_train.index, pd.DatetimeIndex):
        y_train.index = pd.date_range(start='1/1/2000', periods=len(y_train), freq='W') # Example frequency

    print(f"Searching for best ARIMA order (p,d,q) for p in {p_range}, d in {d_range}, q in {q_range}...")
    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    model = ARIMA(y_train, order=(p,d,q), enforce_stationarity=False, enforce_invertibility=False)
                    results = model.fit()
                    print(f"  ARIMA{p,d,q} - AIC: {results.aic:.2f}")
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_order = (p,d,q)
                        best_model_results = results
                except Exception as e:
                    # print(f"   Error fitting ARIMA{p,d,q}: {e}")
                    continue
    
    if best_model_results:
        print(f"\nBest ARIMA model found: Order {best_order} with AIC: {best_aic:.2f}")
    else:
        print("\nCould not find a suitable ARIMA model with the given parameters.")
        
    return best_model_results

if __name__ == '__main__':
    # Example Usage
    print("Testing ARIMA model training...")
    # Create some sample time series data
    data = [x + np.random.random()*5 for x in range(1, 100)]
    sample_y_train = pd.Series(data)
    
    # Train the ARIMA model
    # Using smaller ranges for quick testing
    trained_model = train_arima(sample_y_train, p_range=[0,1], d_range=[0,1], q_range=[0,1])
    
    if trained_model:
        print("\nARIMA model trained successfully.")
        # Forecast next 10 steps
        forecast = trained_model.forecast(steps=10)
        print("Forecast for next 10 steps:", forecast)
    else:
        print("\nARIMA model training failed.")
