# File: src/api/routers/prediction.py (UPDATED)
"""
Prediction API Router - IMPROVED
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import asyncio

from src.serving.predictor import SentimentPredictor
from src.api.dependencies import get_predictor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["prediction"])

class PredictionRequest(BaseModel):
    text: str
    include_features: bool = False
    include_explanation: bool = False

class BatchPredictionRequest(BaseModel):
    texts: List[str]
    include_features: bool = False
    include_explanation: bool = False

@router.post("/predict")
async def predict_sentiment(
    request: PredictionRequest,
    predictor: SentimentPredictor = Depends(get_predictor)
):
    """
    🔮 Predict sentiment for single text (IMPROVED)
    
    **Improvements:**
    - Aggressive threshold-based classification
    - Keyword-based hints for short texts
    - Better handling of ambiguous cases
    """
    try:
        result = await predictor.predict_single(
            text=request.text,
            include_features=request.include_features,
            include_explanation=request.include_explanation
        )
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict-dev")
async def predict_sentiment_dev(
    request: PredictionRequest,
    predictor: SentimentPredictor = Depends(get_predictor)
):
    """
    🧪 Development prediction endpoint (always includes features & explanation)
    """
    try:
        result = await predictor.predict_single(
            text=request.text,
            include_features=True,
            include_explanation=True
        )
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/batch")
async def predict_batch(
    request: BatchPredictionRequest,
    predictor: SentimentPredictor = Depends(get_predictor)
):
    """
    🔮 Predict sentiment for multiple texts (IMPROVED)
    """
    try:
        results = await predictor.predict_batch(
            texts=request.texts,
            include_features=request.include_features,
            include_explanation=request.include_explanation
        )
        return {"predictions": results}
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))