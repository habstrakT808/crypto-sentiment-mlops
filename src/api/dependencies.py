# File: src/api/dependencies.py
"""
FastAPI Dependencies
Authentication, model loading, and shared services
"""

from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import os
import redis
from functools import lru_cache

from src.serving.predictor import SentimentPredictor
from src.utils.logger import setup_logger
from src.utils.config import Config

logger = setup_logger(__name__)

# Security
security = HTTPBearer()

# Valid API keys (in production, use proper key management)
VALID_API_KEYS = {
    "dev-key-123": "development",
    "prod-key-456": "production",
    "demo-key-789": "demo"
}

@lru_cache()
def get_predictor() -> SentimentPredictor:
    """Get singleton predictor instance"""
    try:
        predictor = SentimentPredictor()
        predictor.load_model_sync()
        return predictor
    except Exception as e:
        logger.error(f"Failed to load predictor: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service unavailable"
        )

@lru_cache()
def get_redis_client() -> redis.Redis:
    """Get Redis client"""
    try:
        client = redis.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            db=Config.REDIS_DB,
            decode_responses=True
        )
        client.ping()
        return client
    except Exception as e:
        logger.warning(f"Redis unavailable: {e}")
        return None

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify API key authentication"""
    api_key = credentials.credentials
    
    if api_key not in VALID_API_KEYS:
        logger.warning(f"Invalid API key attempted: {api_key[:10]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return VALID_API_KEYS[api_key]

async def get_current_predictor() -> SentimentPredictor:
    """Get current predictor with error handling"""
    try:
        return get_predictor()
    except Exception as e:
        logger.error(f"Predictor error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service temporarily unavailable"
        )

async def get_current_redis() -> Optional[redis.Redis]:
    """Get current Redis client"""
    return get_redis_client()

# Rate limiting dependency
async def rate_limit_check(api_key_type: str = Depends(verify_api_key)):
    """Check rate limits based on API key type"""
    # Different limits for different key types
    limits = {
        "development": 1000,  # 1000 requests per hour
        "production": 10000,  # 10000 requests per hour
        "demo": 100           # 100 requests per hour
    }
    
    # In a real implementation, you'd check against Redis
    # For now, we'll just pass through
    return True