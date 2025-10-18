"""
Configuration Management
Centralized configuration for the entire application
"""

import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
import yaml

# Load environment variables
load_dotenv()

# Project root directory
ROOT_DIR = Path(__file__).parent.parent.parent

class Config:
    """Application configuration"""
    
    # Environment
    ENV = os.getenv("APP_ENV", "development")
    DEBUG = ENV == "development"
    
    # Reddit API
    REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
    REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
    REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "crypto_sentiment_bot/1.0")
    
    # Database
    POSTGRES_USER = os.getenv("POSTGRES_USER", "crypto_user")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "crypto_password_2024")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "crypto_sentiment_db")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )
    
    # Redis
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))
    
    # MLflow
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "crypto_sentiment_analysis")
    
    # Paths
    DATA_DIR = ROOT_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    EXTERNAL_DATA_DIR = DATA_DIR / "external"
    MODELS_DIR = ROOT_DIR / "models"
    LOGS_DIR = ROOT_DIR / "logs"
    
    # Data Collection
    DATA_COLLECTION_INTERVAL = int(os.getenv("DATA_COLLECTION_INTERVAL", "3600"))
    MAX_POSTS_PER_SUBREDDIT = int(os.getenv("MAX_POSTS_PER_SUBREDDIT", "1000"))
    
    # Subreddits to monitor
    SUBREDDITS = [
        "cryptocurrency",
        "bitcoin",
        "ethereum",
        "cryptomarkets",
        "defi",
        "altcoin"
    ]
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.EXTERNAL_DATA_DIR,
            cls.MODELS_DIR,
            cls.LOGS_DIR
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate(cls) -> Dict[str, Any]:
        """Validate configuration and return status"""
        issues = []
        
        if not cls.REDDIT_CLIENT_ID or cls.REDDIT_CLIENT_ID == "your_client_id_here":
            issues.append("REDDIT_CLIENT_ID not configured")
        
        if not cls.REDDIT_CLIENT_SECRET or cls.REDDIT_CLIENT_SECRET == "your_client_secret_here":
            issues.append("REDDIT_CLIENT_SECRET not configured")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues
        }
    
    @classmethod
    def get_summary(cls) -> str:
        """Get configuration summary"""
        return f"""
Configuration Summary:
=====================
Environment: {cls.ENV}
Database: {cls.POSTGRES_HOST}:{cls.POSTGRES_PORT}/{cls.POSTGRES_DB}
Redis: {cls.REDIS_HOST}:{cls.REDIS_PORT}
MLflow: {cls.MLFLOW_TRACKING_URI}
Data Directory: {cls.DATA_DIR}
Subreddits: {', '.join(cls.SUBREDDITS)}
        """

# Create directories on import
Config.ensure_directories()