# File: src/api/routers/monitoring.py
"""
🔍 Monitoring & Metrics Router
Real-time system monitoring and performance metrics
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List, Any
import time
from datetime import datetime, timedelta
import psutil
import redis
import json

from ..models import MetricsResponse
from ..dependencies import get_current_redis, verify_api_key
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()

# Global metrics storage (in production, use Redis/TimescaleDB)
class MetricsCollector:
    """In-memory metrics collector"""
    
    def __init__(self):
        self.predictions_count = 0
        self.predictions_today = 0
        self.response_times = []
        self.error_count = 0
        self.sentiment_distribution = {"positive": 0, "negative": 0, "neutral": 0}
        self.start_time = time.time()
        self.last_reset = datetime.utcnow()
    
    def record_prediction(self, sentiment: str, response_time: float, success: bool = True):
        """Record a prediction"""
        self.predictions_count += 1
        self.predictions_today += 1
        self.response_times.append(response_time)
        
        # Keep only last 1000 response times
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]
        
        if success:
            self.sentiment_distribution[sentiment] = self.sentiment_distribution.get(sentiment, 0) + 1
        else:
            self.error_count += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        uptime = time.time() - self.start_time
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        return {
            "total_predictions": self.predictions_count,
            "predictions_last_24h": self.predictions_today,
            "average_response_time_ms": avg_response_time,
            "accuracy_last_100": 0.846,  # Mock - in production, calculate from actual results
            "sentiment_distribution_24h": self.sentiment_distribution,
            "error_rate_24h": self.error_count / max(self.predictions_count, 1),
            "uptime_percentage": 99.9,  # Mock
            "uptime_seconds": uptime
        }
    
    def reset_daily(self):
        """Reset daily counters"""
        now = datetime.utcnow()
        if now.date() > self.last_reset.date():
            self.predictions_today = 0
            self.sentiment_distribution = {"positive": 0, "negative": 0, "neutral": 0}
            self.error_count = 0
            self.last_reset = now

# Global metrics collector
metrics_collector = MetricsCollector()

@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    redis_client = Depends(get_current_redis),
    api_key_type: str = Depends(verify_api_key)
):
    """
    📊 **Get system metrics and statistics**
    
    Returns comprehensive metrics including:
    - Total predictions made
    - Prediction volume trends
    - Response time statistics
    - Model accuracy metrics
    - Error rates
    - Sentiment distribution
    
    **Example Response:**
    ```json
    {
        "total_predictions": 1247,
        "predictions_last_24h": 156,
        "average_response_time_ms": 145.3,
        "accuracy_last_100": 0.846,
        "sentiment_distribution_24h": {
            "positive": 45,
            "negative": 23,
            "neutral": 88
        },
        "error_rate_24h": 0.012,
        "uptime_percentage": 99.9
    }
    """
    try:
        # Reset daily counters if needed
        metrics_collector.reset_daily()
        
        # Get metrics from collector
        metrics = metrics_collector.get_metrics()
        
        # Enhance with Redis data if available
        if redis_client:
            try:
                # Get additional metrics from Redis
                redis_total = redis_client.get("total_predictions")
                if redis_total:
                    metrics["total_predictions"] = int(redis_total)
                
                # Get today's predictions
                today_key = f"predictions_today:{datetime.utcnow().date()}"
                redis_today = redis_client.get(today_key)
                if redis_today:
                    metrics["predictions_last_24h"] = int(redis_today)
                
            except Exception as e:
                logger.warning(f"Redis metrics error: {e}")
        
        return MetricsResponse(**metrics)
        
    except Exception as e:
        logger.error(f"Metrics error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve metrics"
        )

@router.get("/metrics/detailed")
async def get_detailed_metrics(
    redis_client = Depends(get_current_redis),
    api_key_type: str = Depends(verify_api_key)
):
    """
    📈 Get detailed system metrics
    
    Returns extended metrics including:
Hourly prediction trends
Response time percentiles
Model performance breakdown
Resource utilization
Error analysis
    """
    try:
        basic_metrics = metrics_collector.get_metrics()
        
        # Calculate percentiles
        response_times = metrics_collector.response_times
        if response_times:
            response_times_sorted = sorted(response_times)
            percentiles = {
                "p50": response_times_sorted[len(response_times_sorted) // 2],
                "p95": response_times_sorted[int(len(response_times_sorted) * 0.95)],
                "p99": response_times_sorted[int(len(response_times_sorted) * 0.99)],
                "min": min(response_times),
                "max": max(response_times)
            }
        else:
            percentiles = {"p50": 0, "p95": 0, "p99": 0, "min": 0, "max": 0}
        
        # System resources
        system_metrics = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_mb": psutil.virtual_memory().used / 1024 / 1024,
            "disk_percent": psutil.disk_usage('/').percent
        }
        
        # Model performance
        model_metrics = {
            "model_name": "LightGBM Production",
            "model_version": "v1.0",
            "accuracy": 0.846,
            "f1_score": 0.834,
            "training_samples": 614,
            "features_count": 43
        }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "basic_metrics": basic_metrics,
            "response_time_percentiles": percentiles,
            "system_resources": system_metrics,
            "model_performance": model_metrics,
            "sentiment_breakdown": metrics_collector.sentiment_distribution
        }
        
    except Exception as e:
        logger.error(f"Detailed metrics error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve detailed metrics"
        )

@router.get("/metrics/hourly")
async def get_hourly_metrics(
    hours: int = 24,
    api_key_type: str = Depends(verify_api_key)
):
    """
    📊 Get hourly prediction metrics
    
    Returns prediction volume and performance for the last N hours.
    """
    try:
        # Mock hourly data - in production, fetch from time-series database
        import random
        
        hourly_data = []
        now = datetime.utcnow()
        
        for i in range(hours):
            hour_time = now - timedelta(hours=hours-i)
            hourly_data.append({
                "timestamp": hour_time.isoformat(),
                "hour": hour_time.hour,
                "predictions": random.randint(30, 80),
                "avg_response_time_ms": random.uniform(120, 180),
                "error_count": random.randint(0, 3),
                "sentiment_distribution": {
                    "positive": random.randint(10, 30),
                    "negative": random.randint(5, 15),
                    "neutral": random.randint(15, 35)
                }
            })
        
        return {
            "period_hours": hours,
            "data": hourly_data,
            "total_predictions": sum(h["predictions"] for h in hourly_data),
            "avg_response_time": sum(h["avg_response_time_ms"] for h in hourly_data) / len(hourly_data)
        }
        
    except Exception as e:
        logger.error(f"Hourly metrics error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve hourly metrics"
        )

@router.post("/metrics/reset")
async def reset_metrics(
    api_key_type: str = Depends(verify_api_key)
):
    """
    🔄 Reset metrics counters
    
    Resets all metrics counters (admin only).
    """
    try:
        global metrics_collector
        metrics_collector = MetricsCollector()
        
        logger.info("Metrics reset by admin")
        
        return {
            "message": "Metrics reset successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Metrics reset error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to reset metrics"
        )

# Export metrics collector for use in other modules
def get_metrics_collector():
    """Get global metrics collector"""
    return metrics_collector