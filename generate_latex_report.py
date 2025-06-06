"""
Run LaTeX Report Generation with Actual Data
This script runs the LaTeX report generator for the rainfall forecasting project
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_dir = r"D:\Forecasting-rainfall-in-Selangor-by-using-machine-learning-techniques"
sys.path.append(os.path.join(project_dir, 'src'))

from utils.latex_report_generator import LatexReportGenerator

def main():
    """Main function to run LaTeX report generation"""
    
    print("=" * 60)
    print("LATEX REPORT GENERATION FOR RAINFALL FORECASTING")
    print("=" * 60)
    
    # Create generator instance
    generator = LatexReportGenerator(project_dir)
    
    # Generate LaTeX report
    print("\nGenerating LaTeX report...")
    tex_file = generator.generate_latex_report()
    
    # Try to compile to PDF
    print("\nAttempting to compile PDF...")
    success = generator.compile_latex(tex_file)
    
    if success:
        print("\n✓ Report generation completed successfully!")
        print(f"  - LaTeX file: {tex_file}")
        print(f"  - PDF file: {tex_file.with_suffix('.pdf')}")
    else:
        print("\n⚠ LaTeX file generated but PDF compilation failed")
        print("  - Make sure you have LaTeX (pdflatex) installed")
        print("  - You can compile the .tex file manually")
        print(f"  - LaTeX file location: {tex_file}")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
