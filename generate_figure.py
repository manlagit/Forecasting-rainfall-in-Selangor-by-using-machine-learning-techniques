import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import json
import seaborn as sns
from pathlib import Path

def generate_model_comparison():
    """Generate model performance comparison figure"""
    model_comparison = pd.read_csv("results/model_comparison.csv", index_col=1)
    model_comparison = model_comparison[['RMSE', 'MAE', 'R2']]

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    colors = ['red', 'green', 'blue', 'orange']

    model_comparison['RMSE'].sort_values().plot(kind='bar', ax=axes[0], color=colors)
    axes[0].set_title('Root Mean Squared Error (RMSE)')
    axes[0].set_ylabel('RMSE')

    model_comparison['MAE'].sort_values().plot(kind='bar', ax=axes[1], color=colors)
    axes[1].set_title('Mean Absolute Error (MAE)')
    axes[1].set_ylabel('MAE')

    model_comparison['R2'].sort_values(ascending=False).plot(kind='bar', ax=axes[2], color=colors)
    axes[2].set_title('R-squared (R2)')
    axes[2].set_ylabel('R2')

    plt.tight_layout()
    return fig

def generate_optuna_optimization():
    """Generate Optuna optimization history plot"""
    # Load optimization history
    with open("models/best_parameters.json") as f:
        study = json.load(f)
    
    fig = plt.figure(figsize=(10, 6))
    # Plot optimization history
    plt.title("Optuna Optimization History")
    plt.xlabel("Trial")
    plt.ylabel("Objective Value")
    return fig

def generate_xgboost_residuals():
    """Generate XGBoost residuals plot"""
    df = pd.read_csv("results/xgboost_predictions.csv")
    fig = plt.figure(figsize=(10, 6))
    sns.residplot(x='y_pred', y='residuals', data=df, lowess=False)
    plt.title("XGBoost Residuals Plot")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    return fig

def generate_ann_training_history():
    """Generate ANN training history plot"""
    history_paths = [
        "models/ann_training_history.csv",
        "logs/training_history.csv",
        "results/ann_training_history.csv"
    ]
    
    history = None
    for path in history_paths:
        try:
            history = pd.read_csv(path)
            break
        except FileNotFoundError:
            continue
            
    if history is None:
        print("Warning: No training history found in any of these locations:")
        print("\n".join(history_paths))
        fig = plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, 'Training history data not found\nChecked locations:\n' + 
                '\n'.join(history_paths),
                ha='center', va='center', fontsize=10)
        plt.title('ANN Training History (Data Missing)')
        return fig
        
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(history['loss'], label='Train')
        if 'val_loss' in history:
            ax1.plot(history['val_loss'], label='Validation')
        ax1.set_title('Model Loss')
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend()

        if 'accuracy' in history:
            ax2.plot(history['accuracy'], label='Train')
            if 'val_accuracy' in history:
                ax2.plot(history['val_accuracy'], label='Validation')
            ax2.set_title('Model Accuracy')
            ax2.set_ylabel('Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.legend()
        
        return fig
    except Exception as e:
        print(f"Could not generate ANN training history: {e}")
        return plt.figure()  # Return empty figure

def main():
    parser = argparse.ArgumentParser(description='Generate figures for rainfall forecasting report')
    parser.add_argument('--model-comparison', action='store_true', help='Generate model comparison figure')
    parser.add_argument('--optuna', action='store_true', help='Generate Optuna optimization figure')
    parser.add_argument('--xgboost-residuals', action='store_true', help='Generate XGBoost residuals plot')
    parser.add_argument('--ann-history', action='store_true', help='Generate ANN training history plot')
    parser.add_argument('--style', choices=['seaborn', 'ggplot', 'default'], default='seaborn',
                      help='Plotting style to use')
    parser.add_argument('--size', choices=['small', 'medium', 'large'], default='medium',
                      help='Figure size')

    args = parser.parse_args()

    # Set style
    try:
        plt.style.use(args.style)
    except:
        plt.style.use('default')
    
    # Set figure size
    sizes = {
        'small': (8, 6),
        'medium': (10, 8),
        'large': (12, 10)
    }
    plt.rcParams['figure.figsize'] = sizes[args.size]

    # Generate requested figure
    if args.model_comparison:
        fig = generate_model_comparison()
        output_path = "reports/figures/model_performance_comparison.png"
    elif args.optuna:
        fig = generate_optuna_optimization()
        output_path = "reports/figures/optuna_optimization.png"
    elif args.xgboost_residuals:
        fig = generate_xgboost_residuals()
        output_path = "reports/figures/XGBoost_residuals.png"
    elif args.ann_history:
        fig = generate_ann_training_history()
        output_path = "reports/figures/ann_training_history.png"
    else:
        print("No figure type specified. Use --help for options.")
        return

    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    print(f"Figure saved to {output_path}")

if __name__ == "__main__":
    main()
