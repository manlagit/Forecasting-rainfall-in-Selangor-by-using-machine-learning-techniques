import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

def evaluate_regression_model(y_true, y_pred, model_name):
    """
    Evaluates a regression model using MAE, MSE, RMSE, and R2.

    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.
        model_name (str): Name of the model being evaluated.

    Returns:
        dict: A dictionary containing the performance metrics.
    """
    metrics = {
        'Model': model_name,
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred)
    }
    
    print(f"\n{model_name} Regression Performance:")
    for metric_name, value in metrics.items():
        if metric_name != 'Model':
            print(f"{metric_name}: {value:.4f}")
    return metrics

def evaluate_classification_model_auc_roc(y_true_binary, y_pred_proba, model_name):
    """
    Evaluates a classification model using AUC-ROC.
    Assumes y_pred_proba contains probabilities for the positive class.

    Args:
        y_true_binary (array-like): True binary labels (0 or 1).
        y_pred_proba (array-like): Predicted probabilities for the positive class.
        model_name (str): Name of the model being evaluated.

    Returns:
        dict: A dictionary containing the AUC-ROC score.
    """
    try:
        auc_roc = roc_auc_score(y_true_binary, y_pred_proba)
        metrics = {
            'Model': model_name,
            'AUC_ROC': auc_roc
        }
        print(f"\n{model_name} Classification Performance:")
        print(f"AUC-ROC: {auc_roc:.4f}")
        return metrics
    except ValueError as e:
        print(f"Could not calculate AUC-ROC for {model_name}. Error: {e}")
        print("Ensure y_true_binary contains at least two classes and y_pred_proba are valid probabilities.")
        return {'Model': model_name, 'AUC_ROC': np.nan}

if __name__ == '__main__':
    # Example Usage (can be removed or kept for direct script testing)
    print("Testing evaluation functions...")

    # Regression example
    y_true_reg = np.array([10, 12, 15, 11, 13])
    y_pred_reg = np.array([10.5, 11.5, 14.5, 11.2, 12.8])
    reg_metrics = evaluate_regression_model(y_true_reg, y_pred_reg, "SampleRegressionModel")
    print("Regression Metrics:", reg_metrics)

    # Classification example
    y_true_clf = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    # Example: probabilities for the positive class
    y_pred_clf_proba = np.array([0.1, 0.8, 0.6, 0.2, 0.9, 0.3, 0.4, 0.7]) 
    
    # Example: binary predictions (if model outputs binary predictions directly)
    # If your model outputs binary predictions (0 or 1) instead of probabilities,
    # you might need to adjust how you get y_pred_proba or use a different metric.
    # For AUC-ROC, probabilities are generally preferred.
    # If you only have binary predictions, you can still calculate AUC-ROC,
    # but it might not be as informative as when using probabilities.
    # y_pred_clf_binary = np.array([0, 1, 1, 0, 1, 0, 1, 1]) # Example binary predictions
    # For AUC-ROC, it's better to use probabilities if available.
    # If using binary predictions directly with roc_auc_score, it's fine,
    # but the curve will be less smooth.

    clf_metrics = evaluate_classification_model_auc_roc(y_true_clf, y_pred_clf_proba, "SampleClassificationModel")
    print("Classification Metrics:", clf_metrics)

    # Example of handling a binary target for AUC-ROC from continuous predictions
    # This is how you might adapt it in your main training script
    y_test_continuous = np.array([0.05, 10.2, 0.0, 5.5, 0.02]) # Example continuous precipitation
    y_test_binary_target = (y_test_continuous > 0.1).astype(int) # Threshold to define "rain"

    # Assuming knn_pred_continuous are the continuous predictions from a regression model
    knn_pred_continuous = np.array([0.1, 8.0, 0.5, 3.0, 0.0]) 
    
    # For AUC-ROC, if your model predicts continuous values (like rainfall amount)
    # and you want to evaluate its ability to classify rain/no-rain,
    # the continuous predictions themselves can sometimes be used as "scores" or "probabilities".
    # However, it's often better if the model is specifically trained for classification
    # or if its output can be meaningfully interpreted as a probability of rain.
    # If using raw regression outputs as scores:
    # Note: This assumes higher predicted rainfall amount means higher likelihood of "rain" class.
    # Scaling predictions to [0,1] might be needed if they are not already in a probability-like range.
    # For simplicity, using them directly here:
    if len(np.unique(y_test_binary_target)) > 1: # Check if there are at least two classes
         auc_roc_from_regression = evaluate_classification_model_auc_roc(y_test_binary_target, knn_pred_continuous, "KNN_as_Classifier")
         print("AUC-ROC for KNN (used as classifier):", auc_roc_from_regression)
    else:
        print("Skipping AUC-ROC for KNN (used as classifier) as y_test_binary_target has only one class.")
