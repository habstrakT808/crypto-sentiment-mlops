"""
Model Evaluator
Evaluate model performance with visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
from pathlib import Path

from .metrics import SentimentMetrics
from src.utils.logger import setup_logger
from src.utils.config import Config

logger = setup_logger(__name__)


class ModelEvaluator:
    """Evaluate and visualize model performance"""
    
    def __init__(self):
        """Initialize evaluator"""
        self.metrics_calculator = SentimentMetrics()
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None,
        class_names: list = None
    ) -> Dict[str, Any]:
        """
        Evaluate predictions
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            class_names: Class names
            
        Returns:
            Evaluation results
        """
        metrics = self.metrics_calculator.calculate_all_metrics(
            y_true, y_pred, y_pred_proba, class_names
        )
        
        return metrics
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: list = None,
        save_path: Path = None,
        figsize: tuple = (8, 6)
    ):
        """Plot confusion matrix"""
        if class_names is None:
            class_names = ['Negative', 'Neutral', 'Positive']
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_training_history(
        self,
        history: Dict[str, list],
        save_path: Path = None,
        figsize: tuple = (12, 4)
    ):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Loss plot
        if 'train_loss' in history:
            axes[0].plot(history['train_loss'], label='Train Loss')
            if 'val_loss' in history:
                axes[0].plot(history['val_loss'], label='Val Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training and Validation Loss')
            axes[0].legend()
            axes[0].grid(True)
        
        # Accuracy plot
        if 'train_acc' in history:
            axes[1].plot(history['train_acc'], label='Train Accuracy')
            if 'val_acc' in history:
                axes[1].plot(history['val_acc'], label='Val Accuracy')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Training and Validation Accuracy')
            axes[1].legend()
            axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history saved to {save_path}")
        
        plt.show()