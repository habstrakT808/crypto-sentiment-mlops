"""
Model Comparator
Compare multiple models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from pathlib import Path

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelComparator:
    """Compare multiple models"""
    
    def __init__(self):
        """Initialize comparator"""
        pass
    
    def compare_metrics(
        self,
        results: Dict[str, Dict],
        metrics: List[str] = None
    ) -> pd.DataFrame:
        """
        Compare metrics across models
        
        Args:
            results: Dictionary of model results
            metrics: List of metrics to compare
            
        Returns:
            Comparison dataframe
        """
        if metrics is None:
            metrics = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']
        
        comparison_data = []
        
        for model_name, result in results.items():
            row = {'model': model_name}
            for metric in metrics:
                row[metric] = result['evaluation']['metrics'].get(metric, 0.0)
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('f1_weighted', ascending=False)
        
        logger.info("Model comparison:")
        logger.info(f"\n{df.to_string()}")
        
        return df
    
    def plot_comparison(
        self,
        comparison_df: pd.DataFrame,
        save_path: Path = None,
        figsize: tuple = (12, 6)
    ):
        """Plot model comparison"""
        metrics = [col for col in comparison_df.columns if col != 'model']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(comparison_df))
        width = 0.8 / len(metrics)
        
        for i, metric in enumerate(metrics):
            offset = width * i - (width * len(metrics) / 2)
            ax.bar(
                x + offset,
                comparison_df[metric],
                width,
                label=metric.replace('_', ' ').title()
            )
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
        
        plt.show()