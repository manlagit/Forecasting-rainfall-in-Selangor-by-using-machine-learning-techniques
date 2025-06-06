import os
import glob

def generate_latex_report(directory="reports/figures"):
    """
    Generates a LaTeX report with figures from a specified directory.
    """

    png_files = glob.glob(os.path.join(directory, "*.png"))
    png_files.sort()  # Ensure consistent ordering

    latex_str = r"""
\documentclass{article}
\usepackage{graphicx}
\usepackage{geometry}
\geometry{a4paper, margin=1in}

\begin{document}
\title{Rainfall Forecasting Report}
\author{AI Assistant}
\date{\today}
\maketitle

\section{Introduction}
This report presents the results of rainfall forecasting models.

\section{Results}
"""

    for png_file in png_files:
        filename = os.path.basename(png_file)
        caption = filename.replace(".png", "").replace("_", " ").title()
        latex_str += rf"""
\begin{{figure}}[h!]
    \centering
    \includegraphics[width=0.8\textwidth]{../figures/{os.path.basename(png_file)}}
    \caption{{{caption}}}
    \label{{fig:{caption.lower().replace(" ", "_")}}}
\end{{figure}}
"""

    latex_str += r"""
\section{Conclusion}
These results demonstrate the performance of various rainfall forecasting models.

\end{document}
"""

    with open("reports/latex/report.tex", "w") as f:
        f.write(latex_str)

if __name__ == "__main__":
    generate_latex_report()
    print("LaTeX report generated successfully at reports/latex/report.tex")
