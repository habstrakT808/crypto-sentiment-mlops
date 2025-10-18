"""
Ensemble Model
Combine multiple models for better predictions
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import LogisticRegression

from .base_model import BaseModel
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class EnsembleModel(BaseModel):
    """Ensemble of multiple sentiment models"""
    
    def __init__(
        self,
        models: List[BaseModel],
        ensemble_method: str = 'soft_voting',
        weights: List[float] = None
    ):
        """
        Initialize ensemble model
        
        Args:
            models: List of trained models
            ensemble_method: 'soft_voting', 'hard_voting', or 'stacking'
            weights: Optional weights for voting (must sum to 1)
        """
        super().__init__(model_name="ensemble_sentiment", model_version="1.0.0")
        
        self.models = models
        self.ensemble_method = ensemble_method
        self.weights = weights
        self.meta_model = None
        
        # Validate weights
        if weights is not None:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.0")
        else:
            # Equal weights
            self.weights = [1.0 / len(models)] * len(models)
        
        # Update metadata
        self.metadata.update({
            'num_models': len(models),
            'ensemble_method': ensemble_method,
            'model_names': [m.model_name for m in models],
            'weights': self.weights
        })
        
        logger.info(f"EnsembleModel initialized with {len(models)} models using {ensemble_method}")
    
    def _get_framework_name(self) -> str:
        """Get framework name"""
        return "ensemble"
    
    def train(
        self,
        X_train: Any,
        y_train: np.ndarray,
        X_val: Any = None,
        y_val: Any = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train ensemble (only for stacking method)
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Training history
        """
        if self.ensemble_method == 'stacking':
            logger.info("Training meta-model for stacking ensemble...")
            
            # Get predictions from all base models
            base_predictions = []
            for model in self.models:
                if not model.is_trained:
                    raise ValueError(f"Model {model.model_name} must be trained first")
                pred_proba = model.predict_proba(X_train)
                base_predictions.append(pred_proba)
            
            # Stack predictions
            X_meta = np.hstack(base_predictions)
            
            # Train meta-model
            self.meta_model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                multi_class='multinomial'
            )
            self.meta_model.fit(X_meta, y_train)
            
            train_score = self.meta_model.score(X_meta, y_train)
            
            self.training_history = {
                'train_accuracy': train_score,
                'train_samples': len(X_train)
            }
            
            # Validation
            if X_val is not None and y_val is not None:
                base_val_predictions = []
                for model in self.models:
                    pred_proba = model.predict_proba(X_val)
                    base_val_predictions.append(pred_proba)
                
                X_val_meta = np.hstack(base_val_predictions)
                val_score = self.meta_model.score(X_val_meta, y_val)
                
                self.training_history['val_accuracy'] = val_score
                logger.info(f"Meta-model validation accuracy: {val_score:.4f}")
            
            logger.info(f"Meta-model training complete. Train accuracy: {train_score:.4f}")
        
        self.is_trained = True
        return self.training_history
    
    def predict(self, X: Any) -> np.ndarray:
        """
        Make ensemble predictions
        
        Args:
            X: Input features
            
        Returns:
            Predicted labels
        """
        if self.ensemble_method == 'soft_voting':
            return self._soft_voting_predict(X)
        elif self.ensemble_method == 'hard_voting':
            return self._hard_voting_predict(X)
        elif self.ensemble_method == 'stacking':
            return self._stacking_predict(X)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Input features
            
        Returns:
            Class probabilities
        """
        if self.ensemble_method == 'soft_voting':
            return self._soft_voting_predict_proba(X)
        elif self.ensemble_method == 'hard_voting':
            # For hard voting, convert predictions to probabilities
            predictions = self._hard_voting_predict(X)
            n_classes = len(np.unique(predictions))
            proba = np.zeros((len(predictions), n_classes))
            for i, pred in enumerate(predictions):
                proba[i, pred] = 1.0
            return proba
        elif self.ensemble_method == 'stacking':
            return self._stacking_predict_proba(X)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def _soft_voting_predict_proba(self, X: Any) -> np.ndarray:
        """Soft voting: average probabilities"""
        all_probas = []
        
        for model, weight in zip(self.models, self.weights):
            proba = model.predict_proba(X)
            all_probas.append(proba * weight)
        
        # Average weighted probabilities
        ensemble_proba = np.sum(all_probas, axis=0)
        return ensemble_proba
    
    def _soft_voting_predict(self, X: Any) -> np.ndarray:
        """Soft voting: predict based on average probabilities"""
        ensemble_proba = self._soft_voting_predict_proba(X)
        return np.argmax(ensemble_proba, axis=1)
    
    def _hard_voting_predict(self, X: Any) -> np.ndarray:
        """Hard voting: majority vote"""
        all_predictions = []
        
        for model in self.models:
            pred = model.predict(X)
            all_predictions.append(pred)
        
        # Stack predictions
        all_predictions = np.array(all_predictions).T
        
        # Majority vote
        from scipy import stats
        ensemble_pred = stats.mode(all_predictions, axis=1)[0].flatten()
        return ensemble_pred
    
    def _stacking_predict_proba(self, X: Any) -> np.ndarray:
        """Stacking: use meta-model"""
        if self.meta_model is None:
            raise ValueError("Meta-model not trained. Call train() first.")
        
        # Get base model predictions
        base_predictions = []
        for model in self.models:
            pred_proba = model.predict_proba(X)
            base_predictions.append(pred_proba)
        
        # Stack predictions
        X_meta = np.hstack(base_predictions)
        
        # Predict with meta-model
        return self.meta_model.predict_proba(X_meta)
    
    def _stacking_predict(self, X: Any) -> np.ndarray:
        """Stacking: predict with meta-model"""
        if self.meta_model is None:
            raise ValueError("Meta-model not trained. Call train() first.")
        
        # Get base model predictions
        base_predictions = []
        for model in self.models:
            pred_proba = model.predict_proba(X)
            base_predictions.append(pred_proba)
        
        # Stack predictions
        X_meta = np.hstack(base_predictions)
        
        # Predict with meta-model
        return self.meta_model.predict(X_meta)
    
    def get_model_contributions(self, X: Any) -> Dict[str, np.ndarray]:
        """
        Get individual model predictions
        
        Args:
            X: Input features
            
        Returns:
            Dictionary with model predictions
        """
        contributions = {}
        
        for model in self.models:
            contributions[model.model_name] = {
                'predictions': model.predict(X),
                'probabilities': model.predict_proba(X)
            }
        
        return contributions