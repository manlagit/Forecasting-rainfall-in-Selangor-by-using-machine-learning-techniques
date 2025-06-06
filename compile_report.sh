#!/bin/bash

# Compile the LaTeX report
cd reports/latex
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex

# Move the PDF to the reports directory
mv report.pdf ../

echo "Report compiled successfully: reports/report.pdf"
