@echo off
echo === STARTING AUTOMATED FEATURE ENGINEERING ===
echo.

REM Activate virtual environment
call .venv\Scripts\activate

REM Create directories if they don't exist
if not exist "reports\figures" mkdir reports\figures
if not exist "data\processed" mkdir data\processed

REM Run the feature engineering script
python feature_engineering_auto.py

REM Check if the script ran successfully
if %ERRORLEVEL% EQU 0 (
    echo.
    echo === FEATURE ENGINEERING COMPLETED SUCCESSFULLY ===
    echo Check data/processed/engineered_features.csv for results
) else (
    echo.
    echo === ERROR: Feature engineering failed! ===
    echo Check the output above for errors
)

pause
