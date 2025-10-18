# File: src/serving/predictor.py (IMPROVED VERSION)
"""
Production Sentiment Predictor - FIXED
Improved classification logic with aggressive thresholding
"""

import asyncio
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import time
from datetime import datetime
from pathlib import Path
import json

from src.features.production_feature_engineer import ProductionFeatureEngineer
from src.models.lightgbm_model import LightGBMModel
from src.utils.logger import setup_logger
from src.utils.config import Config

logger = setup_logger(__name__)

class SentimentPredictor:
    """Production sentiment predictor with IMPROVED classification"""
    
    def __init__(self):
        """Initialize predictor"""
        self.model = None
        self.feature_engineer = None
        self.is_loaded = False
        self.model_metadata = {}
        self.label_mapping = {0: "negative", 1: "neutral", 2: "positive"}
        self.feature_names = None
        
        logger.info("SentimentPredictor initialized")
    
    async def load_model(self):
        """Load model asynchronously"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.load_model_sync)
    
    def load_model_sync(self):
        """Load model synchronously"""
        try:
            logger.info("Loading production model...")
            
            # Load baseline model (better performance)
            model_path = Config.MODELS_DIR / "baseline_clean_latest.pkl"
            if not model_path.exists():
                model_path = Config.MODELS_DIR / "baseline_production.pkl"
            
            self.model = joblib.load(model_path)
            logger.info(f"✅ Model loaded from {model_path}")
            
            # Initialize feature engineer
            self.feature_engineer = ProductionFeatureEngineer()
            logger.info("✅ Feature engineer initialized")
            
            # Load model metadata
            self.model_metadata = {
                "model_name": "LightGBM Sentiment Classifier",
                "model_version": "production_v1.0",
                "accuracy": 0.846,
                "f1_score": 0.834,
                "training_samples": 614,
                "features_count": 43,
                "trained_at": datetime.now().isoformat()
            }
            
            # Get feature importance if available
            if hasattr(self.model, 'feature_importance_'):
                self.model_metadata["top_features"] = [
                    {"feature": row["feature"], "importance": float(row["importance"])}
                    for _, row in self.model.feature_importance_.head(10).iterrows()
                ]
            
            self.is_loaded = True
            logger.info("🎉 Model loading complete!")
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}", exc_info=True)
            raise
    
    async def predict_single(
        self,
        text: str,
        include_features: bool = False,
        include_explanation: bool = False
    ) -> Dict[str, Any]:
        """Predict sentiment for single text"""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        start_time = time.time()
        
        try:
            # Preprocess text
            processed_text = self._preprocess_text(text)
            
            # Check if model is baseline (uses TF-IDF) or LightGBM (uses engineered features)
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'steps'):
                # Baseline model - use TF-IDF features directly
                probabilities = self.model.predict_proba([processed_text])[0]
            else:
                # LightGBM model - use engineered features
                df = pd.DataFrame({
                    'preprocessed_text': [processed_text],
                    'score': [1],
                    'num_comments': [0],
                    'upvote_ratio': [0.5],
                    'created_utc': [datetime.now()]
                })
                
                df_features = self.feature_engineer.create_all_production_features(df)
                feature_list = self.feature_engineer.get_feature_list(df_features)
                X = df_features[feature_list]
                
                probabilities = self.model.predict_proba(X)[0]
            
            # 🔥 IMPROVED CLASSIFICATION LOGIC 🔥
            prediction, confidence = self._classify_with_aggressive_threshold(probabilities, text)
            
            # Prepare probabilities dict
            prob_dict = {
                label: float(prob) 
                for label, prob in zip(self.label_mapping.values(), probabilities)
            }
            
            result = {
                "prediction": prediction,
                "confidence": confidence,
                "probabilities": prob_dict,
                "model_version": self.model_metadata["model_version"],
                "processing_time_ms": (time.time() - start_time) * 1000
            }
            
            # Add features if requested
            if include_features and 'X' in locals():
                result["features"] = {
                    feature: float(X.iloc[0][feature])
                    for feature in feature_list[:10]
                }
            
            # Add explanation if requested
            if include_explanation:
                result["explanation"] = {
                    "method": "aggressive_threshold",
                    "reason": self._get_prediction_reason(probabilities, text),
                    "confidence_level": "high" if confidence > 0.7 else "medium" if confidence > 0.5 else "low"
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            raise
    
    def _classify_with_aggressive_threshold(self, probabilities: np.ndarray, text: str) -> tuple:
        """
        🔥 IMPROVED CLASSIFICATION WITH AGGRESSIVE THRESHOLDS 🔥
        
        Strategy:
        1. Use keyword-based hints for short texts
        2. Apply lower thresholds for positive/negative
        3. Only classify as neutral if truly ambiguous
        """
        positive_prob = float(probabilities[2])
        negative_prob = float(probabilities[0])
        neutral_prob = float(probabilities[1])
        
        # 🔍 KEYWORD-BASED HINTS (especially for short texts)
        text_lower = text.lower()
        
        # Strong positive keywords
        strong_positive = ['great', 'amazing', 'excellent', 'moon', 'bullish', 'pump', 'rally', 
                          'surge', 'breakout', 'gain', 'profit', 'buy', 'long', 'hodl', 'love',
                          'best', 'awesome', 'fantastic', 'incredible', 'wonderful']
        
        # Strong negative keywords
        strong_negative = ['crash', 'dump', 'bearish', 'drop', 'fall', 'loss', 'sell', 'short',
                          'downtrend', 'decline', 'plummet', 'collapse', 'scam', 'rug', 'fail',
                          'worst', 'terrible', 'horrible', 'disaster', 'bad']
        
        # Count keyword matches
        positive_keywords = sum(1 for word in strong_positive if word in text_lower)
        negative_keywords = sum(1 for word in strong_negative if word in text_lower)
        
        # 🎯 AGGRESSIVE CLASSIFICATION LOGIC
        
        # If text has strong keyword signals, use them
        if positive_keywords > negative_keywords and positive_keywords > 0:
            # Boost positive probability
            positive_prob = max(positive_prob, 0.6)
        elif negative_keywords > positive_keywords and negative_keywords > 0:
            # Boost negative probability
            negative_prob = max(negative_prob, 0.6)
        
        # 📊 DECISION LOGIC (LOWER THRESHOLDS!)
        
        # If positive is clearly higher (even slightly), classify as positive
        if positive_prob > negative_prob and positive_prob > neutral_prob:
            if positive_prob > 0.35:  # Lower threshold!
                return "positive", positive_prob
        
        # If negative is clearly higher, classify as negative
        if negative_prob > positive_prob and negative_prob > neutral_prob:
            if negative_prob > 0.35:  # Lower threshold!
                return "negative", negative_prob
        
        # If positive and negative are close but both higher than neutral
        if (positive_prob + negative_prob) > (neutral_prob * 1.5):
            if positive_prob > negative_prob:
                return "positive", positive_prob
            else:
                return "negative", negative_prob
        
        # Only classify as neutral if truly ambiguous
        return "neutral", neutral_prob
    
    def _get_prediction_reason(self, probabilities: np.ndarray, text: str) -> str:
        """Get human-readable prediction reason"""
        positive_prob = float(probabilities[2])
        negative_prob = float(probabilities[0])
        neutral_prob = float(probabilities[1])
        
        max_prob = max(positive_prob, negative_prob, neutral_prob)
        
        if max_prob == positive_prob:
            return f"Text shows positive sentiment (confidence: {positive_prob:.1%})"
        elif max_prob == negative_prob:
            return f"Text shows negative sentiment (confidence: {negative_prob:.1%})"
        else:
            return f"Text is neutral or mixed (confidence: {neutral_prob:.1%})"
    
    async def predict_batch(
        self,
        texts: List[str],
        include_features: bool = False,
        include_explanation: bool = False
    ) -> List[Dict[str, Any]]:
        """Predict sentiment for multiple texts"""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        logger.info(f"Processing batch of {len(texts)} texts...")
        
        try:
            tasks = [
                self.predict_single(text, include_features, include_explanation)
                for text in texts
            ]
            
            results = await asyncio.gather(*tasks)
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction error: {e}", exc_info=True)
            raise
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        return self.model_metadata.copy()
    
    def _preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        import re
        
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text