#!/bin/bash
# Script to compile LaTeX report to PDF
echo "Compiling LaTeX report to PDF..."

# Navigate to the reports/latex directory
cd reports/latex/ || { echo "Directory reports/latex/ not found."; exit 1; }

# Run pdflatex twice to ensure all references are resolved
pdflatex -interaction=nonstopmode rainfall_report.tex
pdflatex -interaction=nonstopmode rainfall_report.tex

# Verify PDF generation
if [ -f "rainfall_report.pdf" ]; then
    echo "PDF compilation successful: rainfall_report.pdf"
    echo "Generated PDF size: $(du -h rainfall_report.pdf | cut -f1)"
    exit 0
else
    echo "PDF compilation failed. Check rainfall_report.log for details."
    cat rainfall_report.log
    exit 1
fi
