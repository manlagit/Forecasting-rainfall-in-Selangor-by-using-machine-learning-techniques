#!/bin/bash
# Script to monitor pipeline execution and handle interruptions
echo "Monitoring pipeline execution..."

MAX_RETRIES=3
retry_count=0

while [ $retry_count -le $MAX_RETRIES ]; do
    # Start the pipeline
    echo "Starting pipeline (attempt $((retry_count+1)) of $((MAX_RETRIES+1)))..."
    
    # For demonstration: Run test script that fails first two times
    if [ $retry_count -lt 2 ]; then
        echo "Simulating pipeline failure (demonstration)..."
        exit 1
    else
        echo "Simulating pipeline success (demonstration)..."
        exit 0
    fi
    
    # Check exit status
    if [ $? -eq 0 ]; then
        echo "Pipeline completed successfully."
        break
    else
        echo "Pipeline failed. Retrying..."
        retry_count=$((retry_count+1))
        sleep 2
    fi
done
if [ $retry_count -gt $MAX_RETRIES ]; then
    echo "Pipeline failed after $MAX_RETRIES retries."
    exit 1
fi

# Check for successful completion
if [ -f "reports/latex/rainfall_report.tex" ]; then
    echo "Pipeline completed successfully. Report generated at reports/latex/rainfall_report.tex"
else
    echo "Pipeline may have failed. Final check..."
    tail -n 20 logs/pipeline_$(date +%Y%m%d)*.log
    exit 1
fi
