"""
Label Validator
Validate quality of auto-generated labels
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.metrics import cohen_kappa_score

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class LabelValidator:
    """Validate auto-labeled data quality"""
    
    def __init__(self):
        """Initialize label validator"""
        pass
    
    def validate_distribution(
        self,
        df: pd.DataFrame,
        expected_distribution: Dict[str, float] = None
    ) -> Dict[str, any]:
        """
        Validate label distribution
        
        Args:
            df: Labeled dataframe
            expected_distribution: Expected distribution (optional)
            
        Returns:
            Validation results
        """
        actual_dist = df['auto_label'].value_counts(normalize=True).to_dict()
        
        # Default expected distribution (roughly balanced)
        if expected_distribution is None:
            expected_distribution = {
                'negative': 0.33,
                'neutral': 0.34,
                'positive': 0.33
            }
        
        # Calculate distribution divergence
        divergence = {}
        for label in ['negative', 'neutral', 'positive']:
            actual = actual_dist.get(label, 0.0)
            expected = expected_distribution.get(label, 0.33)
            divergence[label] = abs(actual - expected)
        
        avg_divergence = np.mean(list(divergence.values()))
        
        # Check if distribution is reasonable
        is_valid = avg_divergence < 0.2  # Less than 20% divergence
        
        result = {
            'is_valid': is_valid,
            'actual_distribution': actual_dist,
            'expected_distribution': expected_distribution,
            'divergence': divergence,
            'avg_divergence': avg_divergence,
            'message': 'Distribution is acceptable' if is_valid else 'Distribution is skewed'
        }
        
        logger.info(f"Distribution validation: {result['message']}")
        logger.info(f"Actual distribution: {actual_dist}")
        
        return result
    
    def validate_inter_annotator_agreement(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate inter-annotator agreement between models
        
        Args:
            df: Labeled dataframe
            
        Returns:
            Agreement metrics
        """
        # Cohen's Kappa between each pair of models
        agreements = {}
        
        model_pairs = [
            ('textblob_prediction', 'vader_prediction'),
            ('textblob_prediction', 'finbert_prediction'),
            ('vader_prediction', 'finbert_prediction')
        ]
        
        for model1, model2 in model_pairs:
            if model1 in df.columns and model2 in df.columns:
                kappa = cohen_kappa_score(df[model1], df[model2])
                agreements[f"{model1}_vs_{model2}"] = kappa
        
        avg_kappa = np.mean(list(agreements.values())) if agreements else 0.0
        
        # Interpret kappa
        if avg_kappa > 0.8:
            interpretation = "Almost perfect agreement"
        elif avg_kappa > 0.6:
            interpretation = "Substantial agreement"
        elif avg_kappa > 0.4:
            interpretation = "Moderate agreement"
        elif avg_kappa > 0.2:
            interpretation = "Fair agreement"
        else:
            interpretation = "Slight agreement"
        
        result = {
            'pairwise_kappa': agreements,
            'average_kappa': avg_kappa,
            'interpretation': interpretation
        }
        
        logger.info(f"Inter-annotator agreement: {avg_kappa:.3f} ({interpretation})")
        
        return result
    
    def validate_confidence_distribution(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate confidence score distribution
        
        Args:
            df: Labeled dataframe
            
        Returns:
            Validation results
        """
        confidence_stats = {
            'mean': float(df['label_confidence'].mean()),
            'median': float(df['label_confidence'].median()),
            'std': float(df['label_confidence'].std()),
            'min': float(df['label_confidence'].min()),
            'max': float(df['label_confidence'].max()),
            'q25': float(df['label_confidence'].quantile(0.25)),
            'q75': float(df['label_confidence'].quantile(0.75))
        }
        
        # Check if confidence distribution is healthy
        is_healthy = (
            confidence_stats['mean'] > 0.5 and
            confidence_stats['std'] < 0.3 and
            confidence_stats['median'] > 0.5
        )
        
        result = {
            'is_healthy': is_healthy,
            'statistics': confidence_stats,
            'message': 'Confidence distribution is healthy' if is_healthy else 'Confidence distribution needs attention'
        }
        
        logger.info(f"Confidence validation: {result['message']}")
        logger.info(f"Mean confidence: {confidence_stats['mean']:.3f}")
        
        return result
    
    def comprehensive_validation(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Run comprehensive validation
        
        Args:
            df: Labeled dataframe
            
        Returns:
            Complete validation report
        """
        logger.info("Running comprehensive label validation...")
        
        report = {
            'distribution_validation': self.validate_distribution(df),
            'inter_annotator_agreement': self.validate_inter_annotator_agreement(df),
            'confidence_validation': self.validate_confidence_distribution(df),
            'sample_size': len(df),
            'unique_labels': df['auto_label'].nunique()
        }
        
        # Overall validation status
        all_valid = (
            report['distribution_validation']['is_valid'] and
            report['inter_annotator_agreement']['average_kappa'] > 0.4 and
            report['confidence_validation']['is_healthy']
        )
        
        report['overall_status'] = 'PASSED' if all_valid else 'NEEDS_REVIEW'
        
        logger.info(f"Comprehensive validation: {report['overall_status']}")
        
        return report