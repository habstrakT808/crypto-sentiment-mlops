@echo off
echo ==================================
echo CRYPTO SENTIMENT API - LOCAL RUN
echo ==================================

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.api.txt

echo Starting API server...
echo API will be available at: http://localhost:8000
echo Dashboard: http://localhost:8501
echo MLflow: http://localhost:5000
echo.
echo Press Ctrl+C to stop the API
echo.

uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
