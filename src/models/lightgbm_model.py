# File: src/models/lightgbm_model.py
"""
LightGBM Model
Gradient boosting with advanced feature engineering
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import joblib

from .base_model import BaseModel
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class LightGBMModel(BaseModel):
    """LightGBM classifier with engineered features"""
    
    def __init__(
        self,
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        max_depth: int = 7,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 0.1,
        random_state: int = 42,
        use_cv: bool = True,
        n_folds: int = 5
    ):
        """Initialize LightGBM model"""
        super().__init__(model_name="lightgbm_sentiment", model_version="1.0.0")
        
        self.params = {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'min_child_samples': min_child_samples,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': random_state,
            'verbose': -1,
            'n_jobs': -1
        }
        
        self.use_cv = use_cv
        self.n_folds = n_folds
        self.feature_importance_ = None
        self.trained_feature_columns = None
        
        logger.info(f"LightGBMModel initialized with {n_estimators} estimators")
    
    def _get_framework_name(self) -> str:
        return "lightgbm"
    
    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for LightGBM"""
        # Select numeric features only
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove ID columns and label columns
        exclude_cols = ['post_id', 'auto_label', 'auto_label_id', 'is_augmented']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        X_prepared = X[numeric_cols].copy()
        
        # Fill missing values
        X_prepared = X_prepared.fillna(X_prepared.mean())
        
        return X_prepared
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame = None,
        y_val: np.ndarray = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train LightGBM with optional cross-validation"""
        logger.info(f"Training LightGBM on {len(X_train)} samples...")
        
        # Prepare features
        X_train_prepared = self._prepare_features(X_train)
        
        logger.info(f"Using {len(X_train_prepared.columns)} features")
        logger.info(f"Features: {X_train_prepared.columns.tolist()[:10]}...")
        
        # Calculate class weights
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        
        logger.info(f"Class weights: {class_weight_dict}")
        
        # Update params with num_class
        self.params['num_class'] = len(classes)
        
        if self.use_cv and len(X_train) > 100:
            # Cross-validation training
            logger.info(f"Training with {self.n_folds}-fold cross-validation...")
            
            cv_scores = []
            cv_models = []
            
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_prepared, y_train)):
                logger.info(f"Training fold {fold + 1}/{self.n_folds}...")
                
                X_fold_train = X_train_prepared.iloc[train_idx]
                y_fold_train = y_train[train_idx]
                X_fold_val = X_train_prepared.iloc[val_idx]
                y_fold_val = y_train[val_idx]
                
                # Create datasets
                train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
                val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)
                
                # Train model
                model = lgb.train(
                    self.params,
                    train_data,
                    valid_sets=[train_data, val_data],
                    valid_names=['train', 'valid'],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=50),
                        lgb.log_evaluation(period=100)
                    ]
                )
                
                # Evaluate
                y_pred = model.predict(X_fold_val)
                y_pred_class = np.argmax(y_pred, axis=1)
                
                from sklearn.metrics import accuracy_score
                fold_acc = accuracy_score(y_fold_val, y_pred_class)
                cv_scores.append(fold_acc)
                cv_models.append(model)
                
                logger.info(f"Fold {fold + 1} accuracy: {fold_acc:.4f}")
            
            # Use the best fold model
            best_fold = np.argmax(cv_scores)
            self.model = cv_models[best_fold]
            
            logger.info(f"CV Scores: {cv_scores}")
            logger.info(f"Mean CV Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
            logger.info(f"Using model from fold {best_fold + 1} (accuracy: {cv_scores[best_fold]:.4f})")
            
            self.training_history = {
                'cv_scores': cv_scores,
                'mean_cv_score': np.mean(cv_scores),
                'std_cv_score': np.std(cv_scores),
                'best_fold': best_fold
            }
            
        else:
            # Single training
            logger.info("Training single model...")
            
            train_data = lgb.Dataset(X_train_prepared, label=y_train)
            
            if X_val is not None and y_val is not None:
                X_val_prepared = self._prepare_features(X_val)
                val_data = lgb.Dataset(X_val_prepared, label=y_val, reference=train_data)
                valid_sets = [train_data, val_data]
                valid_names = ['train', 'valid']
            else:
                valid_sets = [train_data]
                valid_names = ['train']
            
            self.model = lgb.train(
                self.params,
                train_data,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=100)
                ]
            )
            
            self.training_history = {
                'best_iteration': self.model.best_iteration,
                'best_score': self.model.best_score
            }
        
        # Store feature importance
        self.feature_importance_ = pd.DataFrame({
            'feature': X_train_prepared.columns,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        # Persist exact feature order used during training
        self.trained_feature_columns = X_train_prepared.columns.tolist()
        
        logger.info("\nTop 10 Important Features:")
        logger.info(self.feature_importance_.head(10).to_string())
        
        self.is_trained = True
        self.metadata['trained_at'] = pd.Timestamp.now().isoformat()
        self.metadata['training_samples'] = len(X_train)
        
        logger.info("LightGBM training complete!")
        return self.training_history
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        X_prepared = self._prepare_features(X)
        y_pred = self.model.predict(X_prepared)
        return np.argmax(y_pred, axis=1)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        X_prepared = self._prepare_features(X)
        return self.model.predict(X_prepared)
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance"""
        if self.feature_importance_ is None:
            raise ValueError("Model must be trained first")
        
        return self.feature_importance_.head(top_n)