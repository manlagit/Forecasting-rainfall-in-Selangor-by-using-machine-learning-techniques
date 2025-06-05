import matplotlib.pyplot as plt
from pathlib import Path
import logging


# Set up logging
logger = logging.getLogger(__name__)

def generate_full_report(model_metrics, best_model, best_rmse, output_dir="reports/latex"):
    """
    Generate a full LaTeX report document for the rainfall forecasting project.
    
    Args:
        model_metrics (dict): Dictionary with model names as keys and metrics dictionaries as values
        best_model (str): Name of the best performing model
        best_rmse (float): RMSE score of the best model
        output_dir (str): Output directory for the report file
        
    Returns:
        Path: Path to the generated LaTeX report file
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "rainfall_forecasting_report.tex"
    
    # Create LaTeX document
    latex_content = r"""
\documentclass{article}
\usepackage{graphicx}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepackage{booktabs}
\usepackage{array}
\usepackage{geometry}
\geometry{a4paper, margin=1in}
\usepackage{longtable}
\usepackage{xcolor}
\usepackage{hyperref}

\title{Rainfall Forecasting in Selangor: Machine Learning Approaches}
\author{Data Science Team}
\date{\today}

\begin{document}

\maketitle

\section{Executive Summary}
This report presents the analysis of rainfall patterns in Selangor, Malaysia, 
using various machine learning techniques. 
The project aims to develop accurate forecasting models to predict weekly 
rainfall amounts based on historical weather data.

\section{Data Description}
The dataset contains weekly weather measurements from 2012 to 2021, 
including:
\begin{itemize}
    \item Temperature (average)
    \item Relative Humidity
    \item Wind speed (km/h)
    \item Precipitation (mm)
\end{itemize}

\section{Methodology}
We employed the following machine learning algorithms:
\begin{itemize}
    \item Multiple Linear Regression (MLR)
    \item K-Nearest Neighbors (KNN)
    \item Random Forest (RF)
    \item Extreme Gradient Boosting (XGBoost)
    \item Artificial Neural Network (ANN)
\end{itemize}

Models were evaluated using RMSE, MAE, MAPE, and R-squared metrics. 
A comprehensive comparison was performed to identify the best-performing model.

\section{Results}
"""

    # Add model comparison table
    latex_content += r"""
\subsection{Model Performance Comparison}
\begin{center}
\begin{tabular}{lcccc}
\toprule
Model & RMSE & MAE & R-squared & MAPE (\%) \\
\midrule
"""
    
    for model, metrics in model_metrics.items():
        latex_content += (
            f"{model} & {metrics['RMSE']:.4f} & {metrics['MAE']:.4f} & "
            f"{metrics['R2']:.4f} & {metrics['MAPE']:.2f} \\\\\n"
        )
    
    latex_content += r"""\bottomrule
\end{tabular}
\end{center}
"""

    # Highlight best model
    latex_content += f"""
\\subsection{{Best Performing Model}}
The best-performing model was \\textbf{{{best_model}}}, 
achieving an RMSE of {best_rmse:.4f}. 
This model demonstrated superior predictive accuracy compared to other approaches.
"""

    # Add visualizations section
    latex_content += r"""
\subsection{Visualizations}

\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{actual_vs_predicted.png}
\caption{Actual vs Predicted Rainfall}
\label{fig:actual_pred}
\end{figure}

\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{residuals_distribution.png}
\caption{Residuals Distribution}
\label{fig:residuals}
\end{figure}

\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{feature_importance.png}
\caption{Feature Importance}
\label{fig:feature_imp}
\end{figure}
"""

    # Add conclusion
    latex_content += r"""
\section{Conclusion}
The machine learning models developed in this study demonstrate promising 
results for rainfall forecasting in Selangor. 
The \textbf{""" + best_model + r"""} model achieved the best performance 
with an RMSE of """ + f"{best_rmse:.4f}" + r""".

Key findings include:
\begin{itemize}
    \item Rainfall patterns in Selangor show significant seasonal variation
    \item Feature engineering (lag features, moving averages) improved model performance
    \item Ensemble methods (Random Forest, XGBoost) outperformed linear models
\end{itemize}

\section{Future Work}
Future work could explore:
\begin{itemize}
    \item Incorporating additional weather parameters (pressure, cloud cover)
    \item Using deep learning approaches (LSTM, GRU) for sequence modeling
    \item Ensemble modeling techniques to combine predictions
    \item Real-time forecasting system implementation
\end{itemize}

\end{document}
"""
    
    # Save to file
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(latex_content)
    
    logger.info(f"Generated full LaTeX report at {report_path}")
    return report_path


def generate_latex_report(results_df, actual, predictions, model_name, output_dir="reports/latex"):
    """
    Generate LaTeX report components including results table and visualizations
    
    Args:
        results_df (pd.DataFrame): DataFrame with model performance metrics
        actual (np.array): Actual target values
        predictions (np.array): Predicted values
        model_name (str): Name of the model being evaluated
        output_dir (str): Output directory for LaTeX components
        
    Returns:
        dict: Paths to generated components
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate results table in LaTeX format
    latex_table = results_df.to_latex(
        index=False, 
        float_format="%.4f", 
        caption="Model Performance Metrics",
        label="tab:model_performance",
        position="h"
    )
    
    # Save table to file
    table_path = output_path / "results_table.tex"
    with open(table_path, "w") as f:
        f.write(latex_table)
    
    # Generate actual vs predicted plot
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label="Actual Rainfall")
    plt.plot(predictions, label="Predicted Rainfall")
    plt.title(f"Actual vs Predicted Rainfall ({model_name})")
    plt.xlabel("Sample Index")
    plt.ylabel("Rainfall (mm)")
    plt.legend()
    plot_path = output_path / f"actual_vs_predicted_{model_name}.png"
    plt.savefig(plot_path)
    plt.close()
    
    # Generate error distribution plot
    errors = actual - predictions
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, alpha=0.7)
    plt.title(f"Error Distribution ({model_name})")
    plt.xlabel("Prediction Error (mm)")
    plt.ylabel("Frequency")
    error_path = output_path / f"error_distribution_{model_name}.png"
    plt.savefig(error_path)
    plt.close()
    
    return {
        "results_table": table_path,
        "actual_vs_predicted": plot_path,
        "error_distribution": error_path
    }
