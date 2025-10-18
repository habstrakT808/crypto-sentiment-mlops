"""
Confidence Filter
Filter labels based on confidence and agreement scores
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ConfidenceFilter:
    """Filter auto-labeled data based on confidence metrics"""
    
    def __init__(
        self,
        min_confidence: float = 0.6,
        min_agreement: float = 0.67,
        use_both_criteria: bool = True
    ):
        """
        Initialize confidence filter
        
        Args:
            min_confidence: Minimum confidence threshold
            min_agreement: Minimum agreement threshold (0.67 = 2/3 models agree)
            use_both_criteria: Whether both criteria must be met
        """
        self.min_confidence = min_confidence
        self.min_agreement = min_agreement
        self.use_both_criteria = use_both_criteria
        
        logger.info(
            f"ConfidenceFilter initialized: "
            f"min_confidence={min_confidence}, "
            f"min_agreement={min_agreement}, "
            f"use_both={use_both_criteria}"
        )
    
    def filter(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filter dataframe based on confidence criteria
        
        Args:
            df: Labeled dataframe
            
        Returns:
            Tuple of (high_confidence_df, low_confidence_df)
        """
        logger.info(f"Filtering {len(df)} samples...")
        
        # Create filter masks
        confidence_mask = df['label_confidence'] >= self.min_confidence
        agreement_mask = df['label_agreement'] >= self.min_agreement
        
        # Apply filtering logic
        if self.use_both_criteria:
            high_confidence_mask = confidence_mask & agreement_mask
        else:
            high_confidence_mask = confidence_mask | agreement_mask
        
        # Split dataframe
        high_confidence_df = df[high_confidence_mask].copy()
        low_confidence_df = df[~high_confidence_mask].copy()
        
        # Log results
        logger.info(f"High confidence samples: {len(high_confidence_df)} ({len(high_confidence_df)/len(df)*100:.1f}%)")
        logger.info(f"Low confidence samples: {len(low_confidence_df)} ({len(low_confidence_df)/len(df)*100:.1f}%)")
        
        # Log label distribution in high confidence set
        if len(high_confidence_df) > 0:
            label_dist = high_confidence_df['auto_label'].value_counts()
            logger.info(f"High confidence label distribution:\n{label_dist}")
        
        return high_confidence_df, low_confidence_df
    
    def get_filter_stats(
        self,
        high_conf_df: pd.DataFrame,
        low_conf_df: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Get filtering statistics
        
        Args:
            high_conf_df: High confidence dataframe
            low_conf_df: Low confidence dataframe
            
        Returns:
            Dictionary with statistics
        """
        total = len(high_conf_df) + len(low_conf_df)
        
        stats = {
            'total_samples': total,
            'high_confidence_count': len(high_conf_df),
            'low_confidence_count': len(low_conf_df),
            'high_confidence_percentage': len(high_conf_df) / total * 100 if total > 0 else 0,
            'high_confidence_label_dist': high_conf_df['auto_label'].value_counts().to_dict() if len(high_conf_df) > 0 else {},
            'low_confidence_label_dist': low_conf_df['auto_label'].value_counts().to_dict() if len(low_conf_df) > 0 else {},
            'high_confidence_avg_confidence': float(high_conf_df['label_confidence'].mean()) if len(high_conf_df) > 0 else 0.0,
            'high_confidence_avg_agreement': float(high_conf_df['label_agreement'].mean()) if len(high_conf_df) > 0 else 0.0
        }
        
        return stats
    
    def suggest_optimal_thresholds(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Suggest optimal filtering thresholds based on data distribution
        
        Args:
            df: Labeled dataframe
            
        Returns:
            Dictionary with suggested thresholds
        """
        # Calculate percentiles
        confidence_percentiles = df['label_confidence'].quantile([0.25, 0.5, 0.75, 0.9]).to_dict()
        agreement_percentiles = df['label_agreement'].quantile([0.25, 0.5, 0.75, 0.9]).to_dict()
        
        # Suggest thresholds (aim for top 70-80% of data)
        suggested = {
            'conservative': {
                'min_confidence': confidence_percentiles[0.75],
                'min_agreement': 1.0,  # All models must agree
                'expected_retention': '~25%'
            },
            'moderate': {
                'min_confidence': confidence_percentiles[0.5],
                'min_agreement': 0.67,  # 2/3 models agree
                'expected_retention': '~50%'
            },
            'aggressive': {
                'min_confidence': confidence_percentiles[0.25],
                'min_agreement': 0.67,
                'expected_retention': '~75%'
            }
        }
        
        logger.info("Suggested thresholds:")
        for strategy, thresholds in suggested.items():
            logger.info(f"  {strategy}: {thresholds}")
        
        return suggested