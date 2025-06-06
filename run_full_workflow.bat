@echo off
REM Master script to run the entire rainfall forecasting workflow
echo Starting Rainfall Forecasting Workflow...

REM Step 1: Run the pipeline with monitoring
echo Running pipeline with monitoring...
call monitor_pipeline.sh

if errorlevel 1 (
    echo Pipeline failed. Exiting workflow.
    exit /b 1
)

REM Step 2: Compile the report
echo Compiling LaTeX report...
call compile_report.sh

if errorlevel 1 (
    echo Report compilation failed. Exiting workflow.
    exit /b 1
)

REM Step 3: Verify the report
echo Verifying report contents...
call verify_report.sh

if errorlevel 1 (
    echo Report verification failed. Exiting workflow.
    exit /b 1
)

REM Step 4: Final success message
echo.
echo =========================================
echo WORKFLOW COMPLETED SUCCESSFULLY!
echo =========================================
echo Final report: reports\latex\rainfall_report.pdf
echo You can open it with: start reports\latex\rainfall_report.pdf
