"""
Model Trainer
Orchestrate model training with MLflow tracking
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split
from pathlib import Path

from .base_model import BaseModel
from src.utils.logger import setup_logger
from src.utils.config import Config

logger = setup_logger(__name__)


class ModelTrainer:
    """Orchestrate model training pipeline"""
    
    def __init__(
        self,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42,
        stratify: bool = True
    ):
        """
        Initialize model trainer
        
        Args:
            test_size: Test set size
            val_size: Validation set size
            random_state: Random seed
            stratify: Whether to stratify split
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.stratify = stratify
        
        logger.info(
            f"ModelTrainer initialized: "
            f"test_size={test_size}, val_size={val_size}, stratify={stratify}"
        )
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        text_column: str = 'preprocessed_text',
        label_column: str = 'auto_label_id'
    ) -> Tuple[pd.Series, pd.Series, pd.Series, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training
        
        Args:
            df: Input dataframe
            text_column: Column with text data
            label_column: Column with labels
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info(f"Preparing data from {len(df)} samples...")
        
        X = df[text_column]
        y = df[label_column].values
        
        # If stratified split is requested but any class has < 2 samples,
        # disable stratification to avoid ValueError from scikit-learn
        stratify_enabled = self.stratify
        if self.stratify:
            try:
                class_counts = np.bincount(y)
                min_count = class_counts.min() if len(class_counts) > 0 else 0
                if min_count < 2:
                    logger.warning(
                        "Stratified split disabled: at least one class has fewer than 2 samples "
                        f"(class counts={class_counts.tolist()})."
                    )
                    stratify_enabled = False
            except Exception as e:
                logger.warning(f"Could not compute class counts for stratification: {e}. Disabling stratify.")
                stratify_enabled = False
        
        # First split: train+val vs test
        stratify_y = y if stratify_enabled else None
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_y
        )
        
        # Second split: train vs val
        val_size_adjusted = self.val_size / (1 - self.test_size)
        stratify_y_temp = y_temp if stratify_enabled else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=stratify_y_temp
        )
        
        logger.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        logger.info(f"Train label distribution: {np.bincount(y_train)}")
        logger.info(f"Val label distribution: {np.bincount(y_val)}")
        logger.info(f"Test label distribution: {np.bincount(y_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_model(
        self,
        model: BaseModel,
        X_train: pd.Series,
        y_train: np.ndarray,
        X_val: pd.Series,
        y_val: np.ndarray,
        save_model: bool = True,
        **kwargs
    ) -> Tuple[BaseModel, Dict[str, Any]]:
        """
        Train a single model
        
        Args:
            model: Model to train
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            save_model: Whether to save trained model
            
        Returns:
            Tuple of (trained_model, training_history)
        """
        logger.info(f"Training {model.model_name}...")
        
        # Train model
        history = model.train(X_train, y_train, X_val, y_val, **kwargs)
        
        # Save model
        if save_model:
            save_path = Config.MODELS_DIR / f"{model.model_name}_latest.pkl"
            model.save(save_path)
        
        logger.info(f"{model.model_name} training complete!")
        
        return model, history
    
    def evaluate_model(
        self,
        model: BaseModel,
        X_test: pd.Series,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate model on test set
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating {model.model_name}...")
        
        results = model.evaluate(X_test, y_test)
        
        logger.info(f"{model.model_name} evaluation complete!")
        logger.info(f"Test Accuracy: {results['metrics']['accuracy']:.4f}")
        logger.info(f"Test F1 (weighted): {results['metrics']['f1_weighted']:.4f}")
        
        return results
    
    def train_and_evaluate(
        self,
        model: BaseModel,
        df: pd.DataFrame,
        text_column: str = 'preprocessed_text',
        label_column: str = 'auto_label_id',
        save_model: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Complete training and evaluation pipeline
        
        Args:
            model: Model to train
            df: Input dataframe
            text_column: Text column name
            label_column: Label column name
            save_model: Whether to save model
            
        Returns:
            Complete results dictionary
        """
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(
            df, text_column, label_column
        )
        
        # Train model
        trained_model, history = self.train_model(
            model, X_train, y_train, X_val, y_val, save_model, **kwargs
        )
        
        # Evaluate model
        evaluation = self.evaluate_model(trained_model, X_test, y_test)
        
        # Compile results
        results = {
            'model': trained_model,
            'training_history': history,
            'evaluation': evaluation,
            'data_splits': {
                'train_size': len(X_train),
                'val_size': len(X_val),
                'test_size': len(X_test)
            }
        }
        
        return results