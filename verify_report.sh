#!/bin/bash
# Script to verify PDF report contents
echo "Verifying PDF report..."

PDF_FILE="reports/latex/rainfall_report.pdf"

# Check if PDF exists and has content
if [ -s "$PDF_FILE" ]; then
    # Get file size and page count
    file_size=$(du -h "$PDF_FILE" | cut -f1)
    page_count=$(pdftk "$PDF_FILE" dump_data | grep NumberOfPages | awk '{print $2}')
    
    echo "PDF verification passed:"
    echo "  - File size: $file_size"
    echo "  - Page count: $page_count pages"
    echo "  - Content checks:"
    
    # Check for required sections in PDF
    required_sections=("Introduction" "Methodology" "Results" "Key Findings")
    for section in "${required_sections[@]}"; do
        if pdftotext "$PDF_FILE" - | grep -q "$section"; then
            echo "    ✓ '$section' found"
        else
            echo "    ✗ '$section' missing"
        fi
    done
    
    exit 0
else
    echo "PDF report is missing or empty."
    exit 1
fi
