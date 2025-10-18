# File: src/api/routers/health.py
"""
Health Check Router
System health and status monitoring
"""

from fastapi import APIRouter, Depends, HTTPException
import time
import psutil
import os
from datetime import datetime

from ..models import HealthResponse
from ..dependencies import get_current_predictor, get_current_redis
from src.serving.predictor import SentimentPredictor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()

# Store startup time
startup_time = time.time()

@router.get("/health", response_model=HealthResponse)
async def health_check(
    predictor: SentimentPredictor = Depends(get_current_predictor),
    redis_client = Depends(get_current_redis)
):
    """
    🏥 **Comprehensive health check**
    
    Returns detailed health status of all system components.
    
    **Checks:**
    - API service status
    - Model loading status
    - Redis connection status
    - System resource usage
    - Service uptime
    """
    try:
        # Check model status
        model_status = "healthy" if predictor and predictor.is_loaded else "unhealthy"
        
        # Check Redis status
        redis_status = "healthy"
        if redis_client:
            try:
                redis_client.ping()
            except:
                redis_status = "unhealthy"
        else:
            redis_status = "unavailable"
        
        # System metrics
        memory_usage = psutil.virtual_memory().used / 1024 / 1024  # MB
        cpu_usage = psutil.cpu_percent(interval=1)
        uptime = time.time() - startup_time
        
        # Overall status
        overall_status = "healthy" if model_status == "healthy" else "degraded"
        
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.utcnow(),
            version="2.0.0",
            model_status=model_status,
            redis_status=redis_status,
            uptime_seconds=uptime,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage
        )
        
    except Exception as e:
        logger.error(f"Health check error: {e}", exc_info=True)
        return HealthResponse(
            status="error",
            timestamp=datetime.utcnow(),
            version="2.0.0",
            model_status="unknown",
            redis_status="unknown",
            uptime_seconds=time.time() - startup_time,
            memory_usage_mb=0,
            cpu_usage_percent=0
        )

@router.get("/health/live")
async def liveness_probe():
    """
    💓 **Kubernetes liveness probe**
    
    Simple endpoint for container orchestration health checks.
    """
    return {"status": "alive", "timestamp": datetime.utcnow()}

@router.get("/health/ready")
async def readiness_probe(
    predictor: SentimentPredictor = Depends(get_current_predictor)
):
    """
    ✅ **Kubernetes readiness probe**
    
    Checks if the service is ready to handle requests.
    """
    if not predictor or not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {"status": "ready", "timestamp": datetime.utcnow()}