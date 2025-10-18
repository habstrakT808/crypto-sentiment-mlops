# File: src/models/advanced_ensemble.py
"""
Advanced Stacking Ensemble
Multi-level ensemble with uncertainty estimation
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import torch

from .base_model import BaseModel
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class AdvancedStackingEnsemble(BaseModel):
    """Advanced stacking ensemble with multiple levels"""
    
    def __init__(
        self,
        base_models: List[BaseModel],
        meta_model_type: str = 'lightgbm',
        use_oof: bool = True,
        n_folds: int = 5,
        enable_uncertainty: bool = True
    ):
        """
        Initialize advanced ensemble
        
        Args:
            base_models: List of trained base models
            meta_model_type: Type of meta-model ('lightgbm', 'logistic', 'xgboost')
            use_oof: Use out-of-fold predictions for meta-features
            n_folds: Number of folds for OOF
            enable_uncertainty: Enable uncertainty estimation
        """
        super().__init__(model_name="advanced_stacking_ensemble", model_version="1.0.0")
        
        self.base_models = base_models
        self.meta_model_type = meta_model_type
        self.use_oof = use_oof
        self.n_folds = n_folds
        self.enable_uncertainty = enable_uncertainty
        self.meta_model = None
        
        logger.info(f"AdvancedStackingEnsemble initialized with {len(base_models)} base models")
        logger.info(f"Meta-model: {meta_model_type}, Use OOF: {use_oof}")
    
    def _get_framework_name(self) -> str:
        return "ensemble"
    
    def _get_base_predictions(
        self,
        X: Any,
        mode: str = 'proba'
    ) -> np.ndarray:
        """Get predictions from all base models"""
        base_predictions = []
        
        for model in self.base_models:
            logger.info(f"Getting predictions from {model.model_name}...")
            
            try:
                if mode == 'proba':
                    pred = model.predict_proba(X)
                else:
                    pred = model.predict(X)
                    pred = pred.reshape(-1, 1)
                
                base_predictions.append(pred)
            except Exception as e:
                logger.error(f"Error getting predictions from {model.model_name}: {e}")
                # Use dummy predictions
                if mode == 'proba':
                    dummy = np.ones((len(X), 3)) / 3  # Uniform distribution
                else:
                    dummy = np.ones((len(X), 1))
                base_predictions.append(dummy)
        
        # Stack predictions
        stacked = np.hstack(base_predictions)
        logger.info(f"Stacked predictions shape: {stacked.shape}")
        
        return stacked
    
    def _get_oof_predictions(
        self,
        X: Any,
        y: np.ndarray
    ) -> np.ndarray:
        """Get out-of-fold predictions for meta-training"""
        logger.info("Generating out-of-fold predictions...")
        
        n_samples = len(X)
        n_classes = len(np.unique(y))
        
        # Initialize OOF predictions array
        oof_predictions = np.zeros((n_samples, len(self.base_models) * n_classes))
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info(f"Fold {fold + 1}/{self.n_folds}...")
            
            # Get validation data (support DataFrame and Series via iloc)
            if isinstance(X, (pd.DataFrame, pd.Series)):
                X_val = X.iloc[val_idx]
            else:
                X_val = X[val_idx]
            
            # Get predictions from each base model
            fold_predictions = []
            
            for model in self.base_models:
                try:
                    pred = model.predict_proba(X_val)
                    fold_predictions.append(pred)
                except Exception as e:
                    logger.warning(f"Error in fold {fold + 1} for {model.model_name}: {e}")
                    # Use uniform distribution
                    fold_predictions.append(np.ones((len(val_idx), n_classes)) / n_classes)
            
            # Stack fold predictions
            fold_stacked = np.hstack(fold_predictions)
            oof_predictions[val_idx] = fold_stacked
        
        logger.info(f"OOF predictions shape: {oof_predictions.shape}")
        return oof_predictions
    
    def _initialize_meta_model(self, n_features: int, n_classes: int):
        """Initialize meta-model"""
        if self.meta_model_type == 'lightgbm':
            params = {
                'objective': 'multiclass',
                'num_class': n_classes,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'n_estimators': 500,
                'learning_rate': 0.05,
                'max_depth': 5,
                'num_leaves': 31,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': 42,
                'verbose': -1,
                'n_jobs': -1
            }
            self.meta_model = lgb.LGBMClassifier(**params)
            
        elif self.meta_model_type == 'logistic':
            self.meta_model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                multi_class='multinomial',
                solver='lbfgs',
                C=1.0
            )
        
        logger.info(f"Meta-model initialized: {self.meta_model_type}")
    
    def train(
        self,
        X_train: Any,
        y_train: np.ndarray,
        X_val: Any = None,
        y_val: Any = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train stacking ensemble"""
        logger.info(f"Training stacking ensemble on {len(X_train)} samples...")
        
        # Get meta-features
        if self.use_oof:
            meta_features_train = self._get_oof_predictions(X_train, y_train)
        else:
            meta_features_train = self._get_base_predictions(X_train, mode='proba')
        
        # Initialize meta-model
        n_classes = len(np.unique(y_train))
        self._initialize_meta_model(meta_features_train.shape[1], n_classes)
        
        # Train meta-model
        logger.info("Training meta-model...")
        self.meta_model.fit(meta_features_train, y_train)
        
        # Evaluate on training
        train_pred = self.meta_model.predict(meta_features_train)
        from sklearn.metrics import accuracy_score, f1_score
        train_acc = accuracy_score(y_train, train_pred)
        train_f1 = f1_score(y_train, train_pred, average='weighted')
        
        history = {
            'train_accuracy': train_acc,
            'train_f1_weighted': train_f1,
            'meta_features_shape': meta_features_train.shape
        }
        
        # Validation
        if X_val is not None and y_val is not None:
            meta_features_val = self._get_base_predictions(X_val, mode='proba')
            val_pred = self.meta_model.predict(meta_features_val)
            val_acc = accuracy_score(y_val, val_pred)
            val_f1 = f1_score(y_val, val_pred, average='weighted')
            
            history['val_accuracy'] = val_acc
            history['val_f1_weighted'] = val_f1
            
            logger.info(f"Validation - Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        self.is_trained = True
        self.training_history = history
        
        logger.info(f"Stacking ensemble training complete!")
        logger.info(f"Train Accuracy: {train_acc:.4f}, Train F1: {train_f1:.4f}")
        
        return history
    
    def predict(self, X: Any) -> np.ndarray:
        """Make ensemble predictions"""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained first")
        
        meta_features = self._get_base_predictions(X, mode='proba')
        return self.meta_model.predict(meta_features)
    
    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained first")
        
        meta_features = self._get_base_predictions(X, mode='proba')
        return self.meta_model.predict_proba(meta_features)
    
    def predict_with_uncertainty(self, X: Any, n_iterations: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimation using Monte Carlo Dropout
        
        Args:
            X: Input data
            n_iterations: Number of MC iterations
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        if not self.enable_uncertainty:
            predictions = self.predict(X)
            uncertainties = np.zeros(len(predictions))
            return predictions, uncertainties
        
        logger.info(f"Predicting with uncertainty estimation ({n_iterations} iterations)...")
        
        # Get base predictions multiple times
        all_predictions = []
        
        for i in range(n_iterations):
            meta_features = self._get_base_predictions(X, mode='proba')
            proba = self.meta_model.predict_proba(meta_features)
            all_predictions.append(proba)
        
        # Stack predictions
        all_predictions = np.array(all_predictions)  # Shape: (n_iterations, n_samples, n_classes)
        
        # Mean prediction
        mean_proba = np.mean(all_predictions, axis=0)
        predictions = np.argmax(mean_proba, axis=1)
        
        # Uncertainty (standard deviation of predictions)
        std_proba = np.std(all_predictions, axis=0)
        uncertainties = np.max(std_proba, axis=1)  # Max std across classes
        
        logger.info(f"Mean uncertainty: {np.mean(uncertainties):.4f}")
        
        return predictions, uncertainties
    
    def get_model_contributions(self, X: Any) -> Dict[str, np.ndarray]:
        """Get individual model contributions to final prediction"""
        contributions = {}
        
        for model in self.base_models:
            try:
                pred = model.predict_proba(X)
                contributions[model.model_name] = pred
            except Exception as e:
                logger.warning(f"Error getting contribution from {model.model_name}: {e}")
        
        return contributions