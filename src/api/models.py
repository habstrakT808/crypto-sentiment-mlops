# File: src/api/models.py
"""
Pydantic Models for API Request/Response
Type-safe data models with validation
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum

class SentimentLabel(str, Enum):
    """Sentiment labels"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class PredictionRequest(BaseModel):
    """Single prediction request"""
    text: str = Field(
        ...,
        description="Text to analyze for sentiment",
        example="Bitcoin is going to the moon! 🚀",
        min_length=1,
        max_length=10000
    )
    include_features: bool = Field(
        False,
        description="Include feature breakdown in response"
    )
    include_explanation: bool = Field(
        False,
        description="Include SHAP explanation"
    )

    @validator('text')
    def validate_text(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Text cannot be empty")
        return v.strip()

class PredictionResponse(BaseModel):
    """Single prediction response"""
    model_config = {"protected_namespaces": ()}
    
    prediction: SentimentLabel = Field(..., description="Predicted sentiment")
    confidence: float = Field(..., description="Prediction confidence (0-1)", ge=0, le=1)
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_version: str = Field(..., description="Model version used")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    features: Optional[Dict[str, float]] = Field(None, description="Feature values")
    explanation: Optional[Dict[str, Any]] = Field(None, description="SHAP explanation")

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    texts: List[str] = Field(
        ...,
        description="List of texts to analyze",
        min_items=1,
        max_items=100
    )
    include_features: bool = Field(False, description="Include features for each prediction")
    include_explanation: bool = Field(False, description="Include SHAP explanations")

    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError("Texts list cannot be empty")
        
        cleaned_texts = []
        for i, text in enumerate(v):
            if not text or text.strip() == "":
                raise ValueError(f"Text at index {i} cannot be empty")
            cleaned_texts.append(text.strip())
        
        return cleaned_texts

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_count: int = Field(..., description="Total number of predictions")
    processing_time_ms: float = Field(..., description="Total processing time")
    average_confidence: float = Field(..., description="Average confidence score")
    sentiment_distribution: Dict[str, int] = Field(..., description="Distribution of sentiments")

class HealthResponse(BaseModel):
    """Health check response"""
    model_config = {"protected_namespaces": ()}
    
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    model_status: str = Field(..., description="Model status")
    redis_status: str = Field(..., description="Redis status")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    cpu_usage_percent: float = Field(..., description="CPU usage percentage")

class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_config = {"protected_namespaces": ()}
    
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    accuracy: float = Field(..., description="Model accuracy")
    f1_score: float = Field(..., description="F1 score")
    training_samples: int = Field(..., description="Number of training samples")
    features_count: int = Field(..., description="Number of features")
    trained_at: datetime = Field(..., description="Training timestamp")
    top_features: List[Dict[str, Union[str, float]]] = Field(..., description="Top important features")

class ErrorResponse(BaseModel):
    """Error response"""
    error: bool = Field(True, description="Error flag")
    message: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
    timestamp: datetime = Field(..., description="Error timestamp")
    path: str = Field(..., description="Request path")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")

class MetricsResponse(BaseModel):
    """Metrics response"""
    total_predictions: int = Field(..., description="Total predictions made")
    predictions_last_24h: int = Field(..., description="Predictions in last 24 hours")
    average_response_time_ms: float = Field(..., description="Average response time")
    accuracy_last_100: float = Field(..., description="Accuracy on last 100 predictions")
    sentiment_distribution_24h: Dict[str, int] = Field(..., description="Sentiment distribution")
    error_rate_24h: float = Field(..., description="Error rate in last 24 hours")
    uptime_percentage: float = Field(..., description="Service uptime percentage")

class StreamingRequest(BaseModel):
    """Streaming prediction request"""
    subreddits: List[str] = Field(
        ["cryptocurrency", "bitcoin"],
        description="Subreddits to monitor",
        max_items=10
    )
    max_posts: int = Field(
        10,
        description="Maximum posts to analyze",
        ge=1,
        le=100
    )
    min_confidence: float = Field(
        0.7,
        description="Minimum confidence threshold",
        ge=0,
        le=1
    )