# File: src/explainability/shap_explainer.py
"""
Model Explainability
SHAP values for model interpretation
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logger import setup_logger
from src.utils.config import Config

logger = setup_logger(__name__)


class SHAPExplainer:
    """SHAP-based model explainability"""
    
    def __init__(self, model: Any, model_type: str = 'tree'):
        """
        Initialize SHAP explainer
        
        Args:
            model: Trained model
            model_type: Type of model ('tree', 'linear', 'deep')
        """
        self.model = model
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None
        self.feature_columns = None
        self.model_feature_order = None
        self._X_last = None  # X matrix actually used for SHAP (numeric, ordered)
        
        logger.info(f"SHAPExplainer initialized for {model_type} model")
    
    def fit(self, X: pd.DataFrame, max_samples: int = 100):
        """
        Fit SHAP explainer
        
        Args:
            X: Feature matrix
            max_samples: Maximum samples for background data
        """
        logger.info("Fitting SHAP explainer...")
        
        # Sample background data
        if len(X) > max_samples:
            background = X.sample(n=max_samples, random_state=42)
        else:
            background = X
        
        # Keep only numeric/bool features (required by tree SHAP for LightGBM)
        self.feature_columns = background.select_dtypes(include=[np.number, bool]).columns
        if len(self.feature_columns) == 0:
            raise ValueError("No numeric/bool features available for SHAP. Provide numeric feature matrix.")
        background_num = background[self.feature_columns]
        
        # If LightGBM model saved feature order, align
        try:
            self.model_feature_order = getattr(self.model, 'feature_name')() if hasattr(self.model, 'feature_name') else None
        except Exception:
            self.model_feature_order = None
        
        # Initialize explainer based on model type
        if self.model_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_type == 'linear':
            self.explainer = shap.LinearExplainer(self.model, background_num)
        elif self.model_type == 'deep':
            self.explainer = shap.DeepExplainer(self.model, background_num)
        else:
            # Use KernelExplainer as fallback (slower but model-agnostic)
            self.explainer = shap.KernelExplainer(self.model.predict_proba, background_num)
        
        logger.info("SHAP explainer fitted")
    
    def explain(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate SHAP values
        
        Args:
            X: Feature matrix
            
        Returns:
            SHAP values
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        
        logger.info(f"Calculating SHAP values for {len(X)} samples...")
        
        # Align columns and enforce numeric types
        if self.feature_columns is not None:
            X_num = X[self.feature_columns].copy()
        else:
            X_num = X.select_dtypes(include=[np.number, bool]).copy()
        X_num = X_num.astype(float)
        
        # Reorder columns to match model training order if available
        if self.model_feature_order:
            common = [c for c in self.model_feature_order if c in X_num.columns]
            if len(common) > 0:
                X_num = X_num.reindex(columns=common)
                self.feature_columns = X_num.columns
        
        # Keep the exact matrix used for SHAP (for consistent plotting)
        self._X_last = X_num.copy()
        
        self.shap_values = self.explainer.shap_values(X_num)

        # Align SHAP values shape to feature count (LightGBM often returns an extra bias column)
        try:
            n_features = X_num.shape[1]
            if isinstance(self.shap_values, list):
                cleaned = []
                for sv in self.shap_values:
                    if sv.ndim == 2 and sv.shape[1] == n_features + 1:
                        cleaned.append(sv[:, :-1])
                    elif sv.ndim == 3 and sv.shape[2] == n_features + 1:
                        cleaned.append(sv[:, :, :-1])
                    else:
                        cleaned.append(sv)
                self.shap_values = cleaned
            else:
                sv = self.shap_values
                if sv.ndim == 2 and sv.shape[1] == n_features + 1:
                    self.shap_values = sv[:, :-1]
                elif sv.ndim == 3 and sv.shape[2] == n_features + 1:
                    self.shap_values = sv[:, :, :-1]
        except Exception:
            # If any unexpected format, keep original and let plotting raise a clearer error
            pass
        
        logger.info("SHAP values calculated")
        return self.shap_values
    
    def plot_summary(self, X: pd.DataFrame, save_path: str = None):
        """Plot SHAP summary"""
        if self.shap_values is None:
            self.explain(X)
        
        plt.figure(figsize=(12, 8))
        
        plot_X = self._X_last if self._X_last is not None else X[self.feature_columns]
        if isinstance(self.shap_values, list):
            # Multi-class classification
            shap.summary_plot(self.shap_values[1], plot_X, show=False)  # Show positive class
        else:
            shap.summary_plot(self.shap_values, plot_X, show=False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"SHAP summary plot saved to {save_path}")
        
        plt.close()
    
    def plot_feature_importance(self, X: pd.DataFrame, save_path: str = None, top_n: int = 20):
        """Plot feature importance based on SHAP values"""
        if self.shap_values is None:
            self.explain(X)
        
        # Calculate mean absolute SHAP values
        if isinstance(self.shap_values, list):
            # Multi-class: average across classes
            mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in self.shap_values], axis=0)
        else:
            mean_shap = np.abs(self.shap_values).mean(axis=0)
        mean_shap = np.asarray(mean_shap).ravel()
        
        # Use the exact matrix used during explain for column names
        plot_X = self._X_last if self._X_last is not None else X[self.feature_columns]
        feature_names = list(plot_X.columns)
        
        # Align lengths if any off-by-one remains
        n = min(len(feature_names), len(mean_shap))
        feature_names = feature_names[:n]
        mean_shap = mean_shap[:n]
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_shap
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, y='feature', x='importance', palette='viridis')
        plt.title(f'Top {top_n} Features by SHAP Importance')
        plt.xlabel('Mean |SHAP Value|')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.close()
        
        return importance_df
    
    def plot_waterfall(self, X: pd.DataFrame, sample_idx: int = 0, save_path: str = None):
        """Plot SHAP waterfall for a single prediction"""
        if self.shap_values is None:
            self.explain(X)
        
        plt.figure(figsize=(10, 6))
        
        if isinstance(self.shap_values, list):
            # Multi-class: show positive class
            shap.waterfall_plot(
                shap.Explanation(
                    values=self.shap_values[1][sample_idx],
                    base_values=self.explainer.expected_value[1],
                    data=X[self.feature_columns].iloc[sample_idx],
                    feature_names=X[self.feature_columns].columns.tolist()
                ),
                show=False
            )
        else:
            shap.waterfall_plot(
                shap.Explanation(
                    values=self.shap_values[sample_idx],
                    base_values=self.explainer.expected_value,
                    data=X[self.feature_columns].iloc[sample_idx],
                    feature_names=X[self.feature_columns].columns.tolist()
                ),
                show=False
            )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"SHAP waterfall plot saved to {save_path}")
        
        plt.close()
    
    def get_top_features_per_sample(
        self,
        X: pd.DataFrame,
        sample_idx: int = 0,
        top_n: int = 5
    ) -> pd.DataFrame:
        """Get top contributing features for a specific sample"""
        if self.shap_values is None:
            self.explain(X)
        
        if isinstance(self.shap_values, list):
            # Multi-class: use positive class
            sample_shap = self.shap_values[1][sample_idx]
        else:
            sample_shap = self.shap_values[sample_idx]
        
        # Get top features by absolute SHAP value
        top_indices = np.argsort(np.abs(sample_shap))[-top_n:][::-1]
        
        top_features = pd.DataFrame({
            'feature': X[self.feature_columns].columns[top_indices],
            'shap_value': sample_shap[top_indices],
            'feature_value': X[self.feature_columns].iloc[sample_idx, top_indices].values
        })
        
        return top_features