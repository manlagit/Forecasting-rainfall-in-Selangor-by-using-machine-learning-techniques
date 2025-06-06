#!/bin/bash

# Test pipeline failure and recovery
echo "Testing pipeline failure and recovery..."

# Step 1: Introduce a deliberate error in the pipeline
sed -i 's/DataLoader()/DataLoader("invalid_config.yaml")/' main_pipeline.py

# Step 2: Run the pipeline and expect failure
echo "Running pipeline with invalid configuration (expected to fail)..."
python main_pipeline.py
if [ $? -eq 0 ]; then
    echo "❌ Test failed: Pipeline succeeded with invalid configuration"
    exit 1
fi

# Step 3: Restore the original file
git checkout -- main_pipeline.py

# Step 4: Run the pipeline and expect success
echo "Running pipeline with valid configuration (expected to succeed)..."
python main_pipeline.py
if [ $? -ne 0 ]; then
    echo "❌ Test failed: Pipeline failed with valid configuration"
    exit 1
fi

echo "✅ Pipeline failure and recovery test passed successfully"
