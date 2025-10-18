"""
Custom Metrics
Sentiment analysis specific metrics
"""

import numpy as np
from typing import Dict, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    cohen_kappa_score, matthews_corrcoef
)

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SentimentMetrics:
    """Calculate sentiment analysis metrics"""
    
    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None,
        class_names: List[str] = None
    ) -> Dict[str, any]:
        """
        Calculate comprehensive metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            class_names: Class names
            
        Returns:
            Dictionary with all metrics
        """
        if class_names is None:
            class_names = ['negative', 'neutral', 'positive']
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'cohen_kappa': cohen_kappa_score(y_true, y_pred),
            'matthews_corrcoef': matthews_corrcoef(y_true, y_pred)
        }
        
        # Per-class metrics
        # Ensure arrays cover all classes even if some labels are missing in y_true/y_pred
        all_labels = list(range(len(class_names)))
        precision_per_class = precision_score(
            y_true, y_pred, labels=all_labels, average=None, zero_division=0
        )
        recall_per_class = recall_score(
            y_true, y_pred, labels=all_labels, average=None, zero_division=0
        )
        f1_per_class = f1_score(
            y_true, y_pred, labels=all_labels, average=None, zero_division=0
        )
        
        metrics['per_class'] = {}
        for i, class_name in enumerate(class_names):
            metrics['per_class'][class_name] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1': float(f1_per_class[i])
            }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=all_labels)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        report = classification_report(
            y_true, y_pred,
            labels=all_labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        metrics['classification_report'] = report
        
        # ROC AUC if probabilities provided
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(
                        y_true, y_pred_proba,
                        multi_class='ovr', average='weighted'
                    )
            except Exception as e:
                logger.warning(f"Could not compute ROC AUC: {e}")
                metrics['roc_auc'] = 0.0
        
        return metrics
    
    @staticmethod
    def print_metrics_summary(metrics: Dict[str, any]):
        """Print metrics summary"""
        print("\n" + "="*50)
        print("METRICS SUMMARY")
        print("="*50)
        print(f"Accuracy:           {metrics['accuracy']:.4f}")
        print(f"F1 Score (macro):   {metrics['f1_macro']:.4f}")
        print(f"F1 Score (weighted):{metrics['f1_weighted']:.4f}")
        print(f"Precision (macro):  {metrics['precision_macro']:.4f}")
        print(f"Recall (macro):     {metrics['recall_macro']:.4f}")
        print(f"Cohen's Kappa:      {metrics['cohen_kappa']:.4f}")
        
        if 'roc_auc' in metrics:
            print(f"ROC AUC:            {metrics['roc_auc']:.4f}")
        
        print("\nPer-Class Metrics:")
        print("-"*50)
        for class_name, class_metrics in metrics['per_class'].items():
            print(f"{class_name:12s}: P={class_metrics['precision']:.4f}, "
                  f"R={class_metrics['recall']:.4f}, F1={class_metrics['f1']:.4f}")
        print("="*50 + "\n")