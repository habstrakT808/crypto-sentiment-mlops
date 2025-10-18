@echo off
echo ==================================
echo CRYPTO SENTIMENT - SIMPLE MODE
echo ==================================

call venv\Scripts\activate

echo Starting API only...
echo API will be available at: http://localhost:8000
echo Dashboard: Use browser to access http://localhost:8501 (if Docker dashboard is running)

venv\Scripts\python.exe -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
