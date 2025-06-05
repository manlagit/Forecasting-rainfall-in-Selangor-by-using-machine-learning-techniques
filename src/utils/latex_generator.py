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
                         output_dir: str = "reports/latex") -> str:
    """
    Generate a LaTeX report with model performance and visualizations.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report_path = os.path.join(output_dir, "rainfall_report.tex")
    
    # Create LaTeX content
    latex_content = r"""
\documentclass{article}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{geometry}
\geometry{a4paper, margin=1in}
\title{Rainfall Forecasting Report}
\author{Machine Learning Project}
\date{\today}

\begin{document}
\maketitle

\section{Introduction}
This report summarizes the results of rainfall forecasting in Selangor using machine learning techniques. 
The analysis includes model performance metrics and visualizations of the predictions.

\section{Model Comparison}
The following table shows the performance metrics for each model:

\begin{table}[h]
\centering
\caption{Model Performance Comparison}
\begin{tabular}{lccc}
\toprule
Model & RMSE & MAE & R\textsuperscript{2} \\
\midrule
""" + _df_to_latex(comparison_df) + r"""
\bottomrule
\end{tabular}
\end{table}

The best performing model is \textbf{""" + best_model_name.replace('_', r'\_') + r"""}.

\section{Visualizations}
\subsection{Time Series of Rainfall}
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{figures/time_series.png}
\caption{Time Series of Actual Rainfall}
\end{figure}

\subsection{Prediction vs Actual}
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{figures/""" + best_model_name + r"""_pred_vs_actual.png}
\caption{Predicted vs Actual Rainfall (""" + best_model_name.replace('_', r'\_') + r""")}
\end{figure}

\subsection{Model Comparison}
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{figures/model_comparison.png}
\caption{Model Performance Comparison}
\end{figure}

\end{document}
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
        row_str += f"{row['RMSE']:.4f} & {row['MAE']:.4f} & {row['R2']:.4f} \\\\"
        rows.append(row_str)
    return "\n".join(rows)
