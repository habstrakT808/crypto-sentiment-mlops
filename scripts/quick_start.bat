@echo off
REM File: scripts/quick_start.bat
REM Quick Start Script for Windows

echo ==================================
echo CRYPTO SENTIMENT INTELLIGENCE
echo ==================================
echo Quick Start Deployment Script
echo ==================================
echo.

echo Checking prerequisites...

REM Check Docker
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not installed!
    exit /b 1
)
echo [OK] Docker found

REM Check Docker Compose
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker Compose is not installed!
    exit /b 1
)
echo [OK] Docker Compose found

echo.
echo Checking environment file...
if not exist .env (
    echo [WARNING] .env file not found. Copying from .env.example...
    copy .env.example .env
    echo Please edit .env file with your credentials
    pause
)

echo.
echo Validating production model...
if not exist models\lightgbm_production.pkl (
    echo [ERROR] Production model not found!
    echo Please train the model first.
    exit /b 1
)
echo [OK] Production model found

echo.
echo Stopping existing services...
docker-compose -f docker-compose.prod.yml down

echo.
echo Building Docker images (this may take 5-10 minutes)...
docker-compose -f docker-compose.prod.yml build --no-cache

echo.
echo Starting production services...
docker-compose -f docker-compose.prod.yml up -d

echo.
echo Waiting for services to start (30 seconds)...
ping 127.0.0.1 -n 31 > nul

echo.
echo Running health checks...
curl -s http://localhost:8000/api/v1/health/live

echo.
echo ==================================
echo DEPLOYMENT COMPLETE!
echo ==================================
echo.
echo Access Points:
echo   API Documentation:  http://localhost:8000/docs
echo   Dashboard:          http://localhost:8501
echo   MLflow:             http://localhost:5000
echo.
echo Default API Key: demo-key-789
echo.
echo To view logs:
echo   docker-compose -f docker-compose.prod.yml logs -f
echo.
echo To stop services:
echo   docker-compose -f docker-compose.prod.yml down
echo.
pause