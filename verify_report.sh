#!/bin/bash

# Verify the report PDF exists and is non-empty
if [ -f "reports/report.pdf" ]; then
    size=$(du -k "reports/report.pdf" | cut -f1)
    if [ "$size" -gt 0 ]; then
        echo "✅ Report verification passed: PDF file exists and is non-empty"
        exit 0
    else
        echo "❌ Report verification failed: PDF file is empty"
        exit 1
    fi
else
    echo "❌ Report verification failed: PDF file not found"
    exit 1
fi
