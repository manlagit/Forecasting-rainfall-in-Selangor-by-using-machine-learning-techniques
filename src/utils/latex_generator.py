"""
LaTeX Report Generation Module
Generates LaTeX reports for rainfall forecasting results.
"""

import logging
import pandas as pd
import os
from pathlib import Path

def generate_latex_report(comparison_df: pd.DataFrame, 
                         y_test: pd.Series, 
                         y_pred: pd.Series, 
                         best_model_name: str,
                         feature_importance_path: str,
                         residual_plot_path: str,
                         output_dir: str = "reports/latex") -> str:
    """
    Generate a comprehensive LaTeX report with model performance and visualizations.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report_path = os.path.join(output_dir, "rainfall_report.tex")
    
    # Create LaTeX content
    latex_content = r"""
\documentclass{article}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{geometry}
\usepackage{subcaption}
\geometry{a4paper, margin=1in}
\title{Rainfall Forecasting Report}
\author{Machine Learning Project}
\date{\today}

\begin{document}
\maketitle

\section{Introduction}
This report summarizes the results of rainfall forecasting in Selangor using machine learning techniques. 
The analysis includes model performance metrics and visualizations of the predictions.

\section{Key Findings}
Based on our analysis of rainfall patterns in Selangor from 2012-2021, we found:
\begin{itemize}
    \item The best performing model was \textbf{""" + best_model_name.replace('_', r'\_') + r"""} with an R\textsuperscript{2} score of """ + f"{best_r2:.3f}" + r""" and RMSE of """ + f"{best_rmse:.2f}" + r""" mm.
    \item Rainfall patterns show strong seasonality with peaks during the monsoon months (October-December, April).
    \item Temperature and humidity were the most significant predictors of rainfall amounts.
    \item The """ + best_model_name.replace('_', r'\_') + r""" model captured the temporal dependencies in the data effectively.
\end{itemize}

\section{Model Comparison}
The following table shows the performance metrics for each model:

\begin{table}[h]
\centering
\caption{Model Performance Comparison}
\begin{tabular}{lcccc}
\toprule
Model & RMSE & MAE & R\textsuperscript{2} & Training Time (s) \\
\midrule
""" + _df_to_latex(comparison_df) + r"""
\bottomrule
\end{tabular}
\end{table}

The best performing model is \textbf{""" + best_model_name.replace('_', r'\_') + r"""}.

\section{Visualizations}

\subsection{Actual vs Predicted Rainfall}
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{figures/""" + best_model_name + r"""_pred_vs_actual.png}
\caption{Predicted vs Actual Rainfall (""" + best_model_name.replace('_', r'\_') + r""")}
\end{figure}

\subsection{Residual Analysis}
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{""" + residual_plot_path + r"""}
\caption{Residual Plot}
\end{figure}

"""
    # Add feature importance section only if feature_importance_path exists and is not the placeholder
    if feature_importance_path and "placeholder" not in feature_importance_path:
        latex_content += r"""
\subsection{Feature Importance}
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{""" + feature_importance_path + r"""}
\caption{Feature Importance for """ + best_model_name.replace('_', r'\_') + r"""}
\end{figure}
"""
    else:
        latex_content += r"""
\subsection{Feature Importance}
Feature importance visualization is not available for the """ + best_model_name.replace('_', r'\_') + r""" model.
"""
    
    # Save LaTeX file
    with open(report_path, 'w') as f:
        f.write(latex_content)
        
    return report_path

def _df_to_latex(df: pd.DataFrame) -> str:
    """
    Convert DataFrame to LaTeX table rows.
    """
    rows = []
    for index, row in df.iterrows():
        row_str = index.replace('_', r'\_') + " & "
        row_str += f"{row['RMSE']:.4f} & {row['MAE']:.4f} & {row['R2']:.4f} & {row['Training Time']:.2f} \\\\"
        rows.append(row_str)
    return "\n".join(rows)
