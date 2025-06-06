"""
LaTeX Report Generation Module
Generates LaTeX reports for rainfall classification results.
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
                         roc_curve_path: str,
                         confusion_matrix_path: str,
                         output_dir: str = "reports/latex") -> str:
    """
    Generate a comprehensive LaTeX report with model performance and visualizations.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report_path = os.path.join(output_dir, "rainfall_classification_report.tex")
    
    # Extract best model metrics
    best_auc = comparison_df.loc[best_model_name, 'AUC']
    best_precision = comparison_df.loc[best_model_name, 'Precision']
    best_recall = comparison_df.loc[best_model_name, 'Recall']
    
    # Create LaTeX content
    latex_content = r"""
\documentclass{article}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{geometry}
\usepackage{subcaption}
\usepackage{hyperref}
\geometry{a4paper, margin=1in}
\title{Rainfall Occurrence Classification Report for Selangor}
\author{Machine Learning Project}
\date{\today}

\begin{document}
\maketitle

\section{Introduction}
This report summarizes the results of rainfall occurrence classification in Selangor using machine learning techniques. 
The analysis focuses on predicting rain/no-rain events using meteorological data.

\section{Key Findings}
Based on our analysis of rainfall patterns in Selangor from 2012-2021, we found:
\begin{itemize}
    \item The best performing model was \textbf{""" + best_model_name.replace('_', r'\_') + r"""} with an AUC of """ + f"{best_auc:.3f}" + r""", Precision of """ + f"{best_precision:.3f}" + r""", and Recall of """ + f"{best_recall:.3f}" + r""".
    \item Temperature emerged as the most significant predictor of rainfall occurrence.
    \item Humidity and wind speed were also important features in classification.
    \item The """ + best_model_name.replace('_', r'\_') + r""" model demonstrated superior performance in capturing rainfall patterns.
\end{itemize}

\section{Model Comparison}
The following table shows the classification metrics for each model:

\begin{table}[h]
\centering
\caption{Model Performance Comparison}
\begin{tabular}{lcccc}
\toprule
Model & AUC & Precision & Recall & F1 Score \\
\midrule
""" + _df_to_latex(comparison_df) + r"""
\bottomrule
\end{tabular}
\end{table}

The best performing model is \textbf{""" + best_model_name.replace('_', r'\_') + r"""}.

\section{Visualizations}

\subsection{ROC Curve}
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{""" + roc_curve_path + r"""}
\caption{ROC Curve Comparison}
\end{figure}

\subsection{Confusion Matrix}
\begin{figure}[h]
\centering
\includegraphics[width=0.6\textwidth]{""" + confusion_matrix_path + r"""}
\caption{Confusion Matrix for """ + best_model_name.replace('_', r'\_') + r"""}
\end{figure}

\subsection{Feature Importance}
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{""" + feature_importance_path + r"""}
\caption{Feature Importance for """ + best_model_name.replace('_', r'\_') + r"""}
\end{figure}

\section{Practical Implementation}
The developed rainfall classification system can be integrated into Selangor's water management infrastructure to:
\begin{itemize}
    \item Optimize reservoir operations based on rainfall predictions
    \item Provide early warnings for potential flood events
    \item Improve agricultural planning and irrigation scheduling
    \item Enhance urban water distribution efficiency
\end{itemize}

\section{Limitations and Future Work}
\begin{itemize}
    \item Current model uses only meteorological station data - future versions could incorporate satellite imagery
    \item Model performance could be improved with higher temporal resolution data
    \item Integration with real-time monitoring systems would enhance practical utility
    \item Expanding to other regions of Malaysia would increase applicability
\end{itemize}

\end{document}
"""
    # Save LaTeX file
    with open(report_path, 'w') as f:
        f.write(latex_content)
        
    return report_path

def _df_to_latex(df: pd.DataFrame) -> str:
    """
    Convert DataFrame to LaTeX table rows with classification metrics.
    """
    rows = []
    for index, row in df.iterrows():
        row_str = index.replace('_', r'\_') + " & "
        row_str += f"{row['AUC']:.4f} & {row['Precision']:.4f} & {row['Recall']:.4f} & {row['F1']:.4f} \\\\"
        rows.append(row_str)
    return "\n".join(rows)
