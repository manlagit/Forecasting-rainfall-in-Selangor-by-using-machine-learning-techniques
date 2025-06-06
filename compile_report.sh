#!/bin/bash

# Generate the LaTeX report (if needed, though for expanded_report.tex it's manual)
# python generate_report.py

# Compile the LaTeX document
cd reports/latex
pdflatex expanded_report.tex
bibtex expanded_report
pdflatex expanded_report.tex
pdflatex expanded_report.tex

# Move the PDF to the reports directory
mv expanded_report.pdf ../

if [ -f "../expanded_report.pdf" ]; then
  echo "Report compiled successfully: reports/expanded_report.pdf"
else
  echo "Report compilation failed!"
  exit 1
fi
