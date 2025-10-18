"""
Baseline Model
Logistic Regression with TF-IDF features
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

from .base_model import BaseModel
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class BaselineModel(BaseModel):
    """Baseline sentiment classifier using Logistic Regression"""
    
    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42
    ):
        """
        Initialize baseline model
        
        Args:
            max_features: Maximum number of TF-IDF features
            ngram_range: N-gram range for TF-IDF
            C: Regularization parameter
            max_iter: Maximum iterations
            random_state: Random seed
        """
        super().__init__(model_name="baseline_logistic_regression", model_version="1.0.0")
        
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        
        # Create pipeline
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )),
            ('classifier', LogisticRegression(
                C=C,
                max_iter=max_iter,
                random_state=random_state,
                class_weight='balanced',
                multi_class='multinomial',
                solver='lbfgs',
                n_jobs=-1
            ))
        ])
        
        # Update metadata
        self.metadata.update({
            'max_features': max_features,
            'ngram_range': ngram_range,
            'C': C,
            'max_iter': max_iter
        })
        
        logger.info(f"BaselineModel initialized with max_features={max_features}")
    
    def _get_framework_name(self) -> str:
        """Get framework name"""
        return "scikit-learn"
    
    def train(
        self,
        X_train: pd.Series,
        y_train: np.ndarray,
        X_val: pd.Series = None,
        y_val: np.ndarray = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the baseline model
        
        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training history
        """
        logger.info(f"Training baseline model on {len(X_train)} samples...")
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Training metrics
        train_score = self.model.score(X_train, y_train)
        
        self.training_history = {
            'train_accuracy': train_score,
            'train_samples': len(X_train)
        }
        
        # Validation metrics if provided
        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)
            self.training_history['val_accuracy'] = val_score
            self.training_history['val_samples'] = len(X_val)
            logger.info(f"Validation accuracy: {val_score:.4f}")
        
        logger.info(f"Training complete. Train accuracy: {train_score:.4f}")
        
        # Update metadata
        self.metadata['trained_at'] = pd.Timestamp.now().isoformat()
        self.metadata['training_samples'] = len(X_train)
        
        return self.training_history
    
    def predict(self, X: pd.Series) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input texts
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.Series) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Input texts
            
        Returns:
            Class probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get top features for each class
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary with top features per class
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Get feature names and coefficients
        feature_names = self.model.named_steps['tfidf'].get_feature_names_out()
        coefficients = self.model.named_steps['classifier'].coef_
        
        # Get top features for each class
        feature_importance = {}
        class_names = ['negative', 'neutral', 'positive']
        
        for idx, class_name in enumerate(class_names):
            coef = coefficients[idx]
            top_indices = np.argsort(np.abs(coef))[-top_n:][::-1]
            top_features = [(feature_names[i], coef[i]) for i in top_indices]
            feature_importance[class_name] = top_features
        
        return feature_importance