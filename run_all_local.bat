@echo off
echo ==================================
echo CRYPTO SENTIMENT - LOCAL MODE
echo ==================================
echo Starting all services in virtual environment...

call venv\Scripts\activate

echo.
echo [1/4] Starting PostgreSQL...
start "PostgreSQL" /min cmd /c "venv\Scripts\python.exe -c \"import subprocess; subprocess.run(['pg_ctl', 'start', '-D', 'postgres_data'], shell=True)\""

echo [2/4] Starting Redis...
start "Redis" /min cmd /c "redis-server --port 6379"

echo [3/4] Starting MLflow...
start "MLflow" /min cmd /c "venv\Scripts\mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow_artifacts"

echo [4/4] Starting API...
start "API" /min cmd /c "venv\Scripts\python.exe -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload"

echo.
echo [5/5] Starting Dashboard...
venv\Scripts\streamlit run src/monitoring/dashboard/streamlit_app.py --server.port 8501 --server.address 0.0.0.0

echo.
echo All services started!
echo API: http://localhost:8000
echo Dashboard: http://localhost:8501
echo MLflow: http://localhost:5000
pause
