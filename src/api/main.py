# File: src/api/main.py
"""
FastAPI Production API for Crypto Sentiment Analysis
Enterprise-grade API with authentication, monitoring, and caching
"""

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import asyncio
import time
import uuid
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, timedelta
import redis
import json

from .models import (
    PredictionRequest, PredictionResponse, BatchPredictionRequest, 
    BatchPredictionResponse, HealthResponse, ModelInfoResponse,
    ErrorResponse
)
from .dependencies import get_predictor, get_redis_client, verify_api_key
from .routers import prediction, health, monitoring
from .middleware import LoggingMiddleware, RateLimitMiddleware
from src.utils.logger import setup_logger
from src.utils.config import Config

# Setup logging
logger = setup_logger(__name__)

# Global variables for model and services
predictor = None
redis_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global predictor, redis_client
    
    logger.info("🚀 Starting Crypto Sentiment API...")
    
    # Load model and services
    try:
        from src.serving.predictor import SentimentPredictor
        predictor = SentimentPredictor()
        await predictor.load_model()
        logger.info("✅ Model loaded successfully")
        
        # Initialize Redis
        redis_client = redis.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            db=Config.REDIS_DB,
            decode_responses=True
        )
        redis_client.ping()
        logger.info("✅ Redis connected successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise
    
    logger.info("🎉 API startup complete!")
    yield
    
    # Cleanup
    logger.info("🔄 Shutting down API...")
    if redis_client:
        redis_client.close()
    logger.info("✅ API shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="🔮 Crypto Sentiment Intelligence API",
    description="""
    **Advanced MLOps-powered Cryptocurrency Sentiment Analysis API**
    
    This production-grade API provides real-time sentiment analysis for cryptocurrency-related text
    using advanced machine learning models trained on 614+ high-quality Reddit posts.
    
    ## 🎯 Key Features
    - **Real-time Predictions**: Get sentiment scores in <200ms
    - **Batch Processing**: Analyze multiple texts efficiently
    - **High Accuracy**: 84.6% accuracy with zero data leakage
    - **Production Ready**: Enterprise-grade monitoring and caching
    - **Explainable AI**: SHAP-powered feature importance
    
    ## 🔐 Authentication
    All endpoints require an API key in the `X-API-Key` header.
    
    ## 📊 Models
    - **Primary**: LightGBM (84.6% accuracy, 43 features)
    - **Backup**: Ensemble model for comparison
    - **Training**: 614 samples with 5-fold cross-validation
    """,
    version="2.0.0",
    contact={
        "name": "Crypto Sentiment Team",
        "email": "support@crypto-sentiment.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure based on your needs
)

app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware, calls=100, period=60)  # 100 calls per minute

# Include routers
app.include_router(prediction.router, prefix="/api/v1", tags=["Prediction"])
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(monitoring.router, prefix="/api/v1", tags=["Monitoring"])

@app.get("/", response_model=Dict[str, Any])
async def root():
    """API Welcome endpoint"""
    return {
        "message": "🔮 Crypto Sentiment Intelligence API",
        "version": "2.0.0",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "docs": "/docs",
        "health": "/api/v1/health",
        "features": [
            "Real-time sentiment analysis",
            "Batch processing",
            "High accuracy (84.6%)",
            "Production monitoring",
            "Explainable AI"
        ]
    }

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path)
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", 8000)),
        reload=Config.DEBUG,
        log_level="info"
    )