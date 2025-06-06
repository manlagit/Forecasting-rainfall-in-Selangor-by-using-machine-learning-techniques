import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# Adjust import paths based on project structure
# Assuming train_models.py is in src/models/
# and other modules are in src/evaluation/ and src/models/
try:
    from src.evaluation.evaluate import evaluate_regression_model, evaluate_classification_model_auc_roc
    from src.models.arima_model import train_arima
    from src.models.knn_model import train_knn
except ModuleNotFoundError:
    # Fallback for direct execution if PYTHONPATH is not set, or for easier notebook import
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..')) # Add project root to path
    from src.evaluation.evaluate import evaluate_regression_model, evaluate_classification_model_auc_roc
    from src.models.arima_model import train_arima
    from src.models.knn_model import train_knn


# Define base directory for relative paths
BASE_DIR = os.path.join(os.path.dirname(__file__), '..', '..') # Project root

# Define file paths
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'engineered_features.csv')
RESULTS_PATH = os.path.join(BASE_DIR, 'results')
MODELS_PATH = os.path.join(BASE_DIR, 'models', 'saved_models')
FIGURES_PATH = os.path.join(BASE_DIR, 'reports', 'figures')

# Ensure output directories exist
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(FIGURES_PATH, exist_ok=True)

def main():
    print("Starting model training and evaluation script...")

    # Load engineered dataset
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        df['Date'] = pd.to_datetime(df['Date'])
        print(f"Engineered data loaded successfully from {PROCESSED_DATA_PATH}")
    except FileNotFoundError:
        print(f"Error: Engineered data file not found at {PROCESSED_DATA_PATH}")
        return
    except Exception as e:
        print(f"Error loading engineered data: {e}")
        return

    # Drop rows with NaN values that might persist (e.g. from feature selection if original cols had NaNs)
    df.dropna(inplace=True)
    if df.empty:
        print("DataFrame is empty after dropping NaNs. Cannot proceed.")
        return

    # Split Features and Target
    # Ensure 'Precipitation_mm' is the target and 'Date' is handled
    if 'Precipitation_mm' not in df.columns:
        print("Error: Target column 'Precipitation_mm' not found.")
        return
    
    # Select features: exclude Date, Year, Week_Number for X if they exist, and the target
    # This assumes all other columns are features. Adjust if specific features were pre-selected.
    feature_cols = [col for col in df.columns if col not in ['Date', 'Year', 'Week_Number', 'Precipitation_mm']]
    
    # If 'Year' and 'Week_Number' were part of selected_features in feature_engineering, they might be here.
    # For many models, they are useful. For ARIMA, only y_train is used.
    # For KNN, ensure X_train, X_test are purely numeric.
    
    X = df[feature_cols]
    y = df['Precipitation_mm']
    
    if X.empty or y.empty:
        print("Feature set X or target y is empty. Cannot proceed.")
        return

    # Time-Based Train-Test Split (80-20)
    split_idx = int(len(df) * 0.8)
    if split_idx == 0 or split_idx == len(df):
        print("Not enough data for a meaningful train-test split.")
        return
        
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_test = df['Date'].iloc[split_idx:]

    print(f"Training data shape: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Testing data shape: X_test {X_test.shape}, y_test {y_test.shape}")

    # Initialize results storage
    all_model_results = []
    model_predictions = {'Date': dates_test.values, 'Actual': y_test.values}


    # --- Train and Evaluate ARIMA Model ---
    print("\n--- Training ARIMA Model ---")
    arima_model_fitted = train_arima(y_train.copy(), p_range=[0,1,2], d_range=[0,1], q_range=[0,1,2]) # Use a copy for y_train
    if arima_model_fitted:
        arima_pred = arima_model_fitted.forecast(steps=len(y_test))
        model_predictions['ARIMA_pred'] = arima_pred.values
        
        reg_metrics_arima = evaluate_regression_model(y_test, arima_pred, "ARIMA")
        all_model_results.append(reg_metrics_arima)
        
        # Example AUC-ROC for ARIMA (treating predictions as scores for rain/no-rain)
        y_test_binary = (y_test > 0.1).astype(int) # Define rain if > 0.1mm
        if len(np.unique(y_test_binary)) > 1:
             # ARIMA predictions can be negative, clip at 0 or scale if necessary for probability-like scores
            arima_pred_scores = np.maximum(0, arima_pred) 
            auc_roc_arima = evaluate_classification_model_auc_roc(y_test_binary, arima_pred_scores, "ARIMA_classification")
            if auc_roc_arima and 'AUC_ROC' in auc_roc_arima:
                 all_model_results[-1]['AUC_ROC'] = auc_roc_arima['AUC_ROC']
        else:
            print("Skipping AUC-ROC for ARIMA as y_test_binary has only one class.")
        
        # Save ARIMA model
        try:
            arima_model_fitted.save(os.path.join(MODELS_PATH, "arima_model.pkl"))
            print("ARIMA model saved.")
        except Exception as e:
            print(f"Error saving ARIMA model: {e}")
    else:
        print("ARIMA model training failed.")

    # --- Train and Evaluate KNN Model ---
    print("\n--- Training KNN Model ---")
    # Ensure X_train and X_test for KNN are purely numeric and do not contain NaNs from feature engineering
    X_train_knn = X_train.select_dtypes(include=np.number).fillna(X_train.mean()) # Simple imputation for safety
    X_test_knn = X_test.select_dtypes(include=np.number).fillna(X_train.mean())   # Use X_train's mean for X_test

    knn_model_fitted = train_knn(X_train_knn, y_train, n_splits=3) # Using 3 splits for faster run
    if knn_model_fitted:
        knn_pred = knn_model_fitted.predict(X_test_knn)
        model_predictions['KNN_pred'] = knn_pred
        
        reg_metrics_knn = evaluate_regression_model(y_test, knn_pred, "KNN")
        all_model_results.append(reg_metrics_knn)

        # Example AUC-ROC for KNN
        if len(np.unique(y_test_binary)) > 1:
            # KNN regression outputs are continuous. Use these as scores.
            # Scale if necessary, but direct use is common.
            knn_pred_scores = knn_pred 
            auc_roc_knn = evaluate_classification_model_auc_roc(y_test_binary, knn_pred_scores, "KNN_classification")
            if auc_roc_knn and 'AUC_ROC' in auc_roc_knn:
                all_model_results[-1]['AUC_ROC'] = auc_roc_knn['AUC_ROC']
        else:
            print("Skipping AUC-ROC for KNN as y_test_binary has only one class.")

        # Save KNN model
        try:
            joblib.dump(knn_model_fitted, os.path.join(MODELS_PATH, "knn_model.pkl"))
            print("KNN model saved.")
        except Exception as e:
            print(f"Error saving KNN model: {e}")
    else:
        print("KNN model training failed.")

    # --- Results Analysis ---
    if not all_model_results:
        print("\nNo models were successfully trained and evaluated.")
        return

    results_df = pd.DataFrame(all_model_results)
    print("\n--- Model Performance Comparison ---")
    print(results_df)
    try:
        results_df.to_csv(os.path.join(RESULTS_PATH, "model_comparison.csv"), index=False)
        print(f"Model comparison results saved to {os.path.join(RESULTS_PATH, 'model_comparison.csv')}")
    except Exception as e:
        print(f"Error saving model comparison results: {e}")
    
    # Save predictions
    predictions_df = pd.DataFrame(model_predictions)
    try:
        predictions_df.to_csv(os.path.join(RESULTS_PATH, "model_predictions.csv"), index=False)
        print(f"Model predictions saved to {os.path.join(RESULTS_PATH, 'model_predictions.csv')}")
    except Exception as e:
        print(f"Error saving predictions: {e}")


    # --- Visualize Predictions vs Actual ---
    print("\n--- Visualizing Predictions ---")
    plt.figure(figsize=(14, 7))
    plt.plot(dates_test, y_test, label='Actual Rainfall', color='blue', marker='o', linestyle='-')
    if 'ARIMA_pred' in model_predictions:
        plt.plot(dates_test, model_predictions['ARIMA_pred'], label='ARIMA Predicted', color='red', linestyle='--')
    if 'KNN_pred' in model_predictions:
        plt.plot(dates_test, model_predictions['KNN_pred'], label='KNN Predicted', color='green', linestyle='-.')
    
    plt.title('Actual vs. Predicted Rainfall Over Time')
    plt.xlabel('Date')
    plt.ylabel('Precipitation (mm)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    try:
        plt.savefig(os.path.join(FIGURES_PATH, "predictions_vs_actual.png"), dpi=300)
        print(f"Predictions vs. actual plot saved to {os.path.join(FIGURES_PATH, 'predictions_vs_actual.png')}")
    except Exception as e:
        print(f"Error saving predictions plot: {e}")
    # plt.show() # Comment out if running in a non-interactive environment

    # --- Save Best Model (Example based on RMSE) ---
    if not results_df.empty and 'RMSE' in results_df.columns:
        best_model_name_reg = results_df.sort_values(by='RMSE').iloc[0]['Model']
        print(f"\nBest regression model based on RMSE: {best_model_name_reg}")
        # Note: Saving logic is already within each model's training block.
        # This part is just to identify the best one based on current results.
    else:
        print("\nCould not determine the best model as results_df is empty or RMSE column is missing.")

    print("\nModel training and evaluation script completed.")

if __name__ == '__main__':
    main()
