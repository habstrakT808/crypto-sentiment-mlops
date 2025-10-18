"""
Base Model Class
Abstract base class for all ML models
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
import joblib
from pathlib import Path
import json
from datetime import datetime

from src.utils.logger import setup_logger
from src.utils.config import Config
from src.evaluation.metrics import SentimentMetrics

logger = setup_logger(__name__)


class BaseModel(ABC):
    """Abstract base class for all models"""
    
    def __init__(self, model_name: str, model_version: str = "1.0.0"):
        """
        Initialize base model
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
        """
        self.model_name = model_name
        self.model_version = model_version
        self.model = None
        self.is_trained = False
        self.training_history = {}
        self.metadata = {
            'model_name': model_name,
            'model_version': model_version,
            'created_at': datetime.now().isoformat(),
            'framework': self._get_framework_name()
        }
        
        logger.info(f"Initialized {model_name} v{model_version}")
    
    @abstractmethod
    def _get_framework_name(self) -> str:
        """Get framework name (sklearn, pytorch, etc)"""
        pass
    
    @abstractmethod
    def train(
        self, 
        X_train: Any, 
        y_train: Any, 
        X_val: Any = None, 
        y_val: Any = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            **kwargs: Additional training parameters
            
        Returns:
            Training history dictionary
        """
        pass
    
    @abstractmethod
    def predict(self, X: Any) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Predictions array
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Input features
            
        Returns:
            Probability predictions array
        """
        pass
    
    def evaluate(
        self, 
        X_test: Any, 
        y_test: Any
    ) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {self.model_name}...")
        
        # Get predictions
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        # Calculate comprehensive metrics (includes per_class, confusion_matrix, report)
        metrics = SentimentMetrics.calculate_all_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            class_names=['negative', 'neutral', 'positive']
        )
        
        logger.info(f"Evaluation complete. Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_weighted']:.4f}")
        
        return {
            'metrics': metrics,
            'classification_report': metrics.get('classification_report')
        }
    
    def save(self, path: Path = None) -> Path:
        """
        Save model to disk
        
        Args:
            path: Optional save path
            
        Returns:
            Path where model was saved
        """
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = Config.MODELS_DIR / f"{self.model_name}_{timestamp}.pkl"
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and metadata
        save_dict = {
            'model': self.model,
            'metadata': self.metadata,
            'training_history': self.training_history,
            'is_trained': self.is_trained
        }
        
        joblib.dump(save_dict, path)
        logger.info(f"Model saved to {path}")
        
        # Save metadata as JSON
        metadata_path = path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        return path
    
    def load(self, path: Path):
        """
        Load model from disk
        
        Args:
            path: Path to load model from
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        save_dict = joblib.load(path)
        
        self.model = save_dict['model']
        self.metadata = save_dict['metadata']
        self.training_history = save_dict['training_history']
        self.is_trained = save_dict['is_trained']
        
        logger.info(f"Model loaded from {path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'is_trained': self.is_trained,
            'metadata': self.metadata,
            'training_history': self.training_history
        }