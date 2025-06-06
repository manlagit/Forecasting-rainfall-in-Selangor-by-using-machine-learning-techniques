import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_knn(X_train, y_train, n_splits=5):
    """
    Trains a K-Nearest Neighbors (KNN) regressor model with hyperparameter tuning
    using GridSearchCV and TimeSeriesSplit. Includes scaling of features.

    Args:
        X_train (pd.DataFrame or np.array): Training feature data.
        y_train (pd.Series or np.array): Training target data.
        n_splits (int): Number of splits for TimeSeriesSplit.

    Returns:
        sklearn.neighbors.KNeighborsRegressor: The best KNN estimator found by GridSearchCV.
                                              Returns None if training fails.
    """
    print("Training KNN model with GridSearchCV and TimeSeriesSplit...")

    # Define the parameter grid for KNN
    param_grid = {
        'model__n_neighbors': [3, 5, 7, 9, 11, 15],
        'model__weights': ['uniform', 'distance'],
        'model__p': [1, 2]  # 1 for Manhattan distance, 2 for Euclidean distance
    }

    # Create a pipeline with StandardScaler and KNeighborsRegressor
    # Scaling is important for distance-based algorithms like KNN
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', KNeighborsRegressor())
    ])

    # Use TimeSeriesSplit for cross-validation as data has a temporal order
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Setup GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',  # Use a score that reflects error (lower is better)
        n_jobs=-1,  # Use all available cores
        verbose=1   # Print progress
    )

    try:
        grid_search.fit(X_train, y_train)
        print(f"\nBest KNN parameters found: {grid_search.best_params_}")
        print(f"Best KNN score (Negative MSE): {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_
    except Exception as e:
        print(f"Error during KNN model training: {e}")
        return None

if __name__ == '__main__':
    # Example Usage
    print("Testing KNN model training...")
    
    # Create sample data (time-series like)
    X_sample = pd.DataFrame({
        'feature1': np.arange(100) + np.random.rand(100) * 10,
        'feature2': np.arange(100, 200) + np.random.rand(100) * 5
    })
    y_sample = pd.Series(np.arange(50, 150) + np.random.rand(100) * 20)
    
    # Train the KNN model
    trained_knn_model = train_knn(X_sample, y_sample, n_splits=3) # Using fewer splits for faster example
    
    if trained_knn_model:
        print("\nKNN model trained successfully.")
        # Example prediction
        if len(X_sample) > 0:
            sample_prediction = trained_knn_model.predict(X_sample.head(5))
            print("Sample predictions for the first 5 data points:", sample_prediction)
    else:
        print("\nKNN model training failed.")
