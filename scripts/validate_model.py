# File: scripts/validate_model.py
"""
🔍 Model Validation Script
Validates model performance before deployment
"""

import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.production_feature_engineer import ProductionFeatureEngineer
from src.utils.logger import setup_logger
from sklearn.metrics import accuracy_score, f1_score, classification_report

logger = setup_logger(__name__)

def validate_model(
    model_path: str,
    test_data_path: str = None,
    min_accuracy: float = 0.80,
    min_f1: float = 0.75
) -> bool:
    """
    Validate model performance
    
    Args:
        model_path: Path to model file
        test_data_path: Path to test data (optional)
        min_accuracy: Minimum required accuracy
        min_f1: Minimum required F1 score
        
    Returns:
        True if validation passes, False otherwise
    """
    logger.info("="*80)
    logger.info("🔍 MODEL VALIDATION")
    logger.info("="*80)
    
    try:
        # Load model
        logger.info(f"Loading model from {model_path}...")
        model = joblib.load(model_path)
        logger.info("✅ Model loaded successfully")
        
        # Load test data
        if test_data_path:
            logger.info(f"Loading test data from {test_data_path}...")
            test_data = pd.read_csv(test_data_path)
            logger.info(f"✅ Loaded {len(test_data)} test samples")
        else:
            # Use default test data
            from src.utils.config import Config
            test_data_path = Config.PROCESSED_DATA_DIR / "labeled_data_large.csv"
            
            if not test_data_path.exists():
                test_data_path = Config.PROCESSED_DATA_DIR / "labeled_data.csv"
            
            logger.info(f"Loading default test data from {test_data_path}...")
            test_data = pd.read_csv(test_data_path)
            
            # Use last 20% as test set
            test_size = int(len(test_data) * 0.2)
            test_data = test_data.tail(test_size)
            logger.info(f"✅ Using {len(test_data)} samples for validation")
        
        # Feature engineering
        logger.info("Creating features...")
        feature_engineer = ProductionFeatureEngineer()
        test_data = feature_engineer.create_all_production_features(test_data)
        feature_list = feature_engineer.get_feature_list(test_data)
        
        X_test = test_data[feature_list]
        y_test = test_data['auto_label_id'].values
        
        logger.info(f"✅ Features created: {len(feature_list)} features")
        
        # Make predictions
        logger.info("Making predictions...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        logger.info("\n" + "="*80)
        logger.info("📊 VALIDATION RESULTS")
        logger.info("="*80)
        logger.info(f"Accuracy:     {accuracy:.4f} (threshold: {min_accuracy:.4f})")
        logger.info(f"F1 Weighted:  {f1_weighted:.4f} (threshold: {min_f1:.4f})")
        logger.info(f"F1 Macro:     {f1_macro:.4f}")
        
        # Classification report
        logger.info("\n📋 Classification Report:")
        logger.info("\n" + classification_report(
            y_test, y_pred,
            target_names=['negative', 'neutral', 'positive']
        ))
        
        # Confidence analysis
        max_proba = np.max(y_pred_proba, axis=1)
        avg_confidence = np.mean(max_proba)
        logger.info(f"\n💎 Average Confidence: {avg_confidence:.4f}")
        logger.info(f"   High confidence (>0.8): {(max_proba > 0.8).sum()} samples ({(max_proba > 0.8).sum()/len(max_proba)*100:.1f}%)")
        logger.info(f"   Medium confidence (0.6-0.8): {((max_proba > 0.6) & (max_proba <= 0.8)).sum()} samples")
        logger.info(f"   Low confidence (<0.6): {(max_proba <= 0.6).sum()} samples")
        
        # Validation decision
        logger.info("\n" + "="*80)
        logger.info("🎯 VALIDATION DECISION")
        logger.info("="*80)
        
        passed = True
        
        if accuracy < min_accuracy:
            logger.error(f"❌ Accuracy {accuracy:.4f} below threshold {min_accuracy:.4f}")
            passed = False
        else:
            logger.info(f"✅ Accuracy {accuracy:.4f} meets threshold {min_accuracy:.4f}")
        
        if f1_weighted < min_f1:
            logger.error(f"❌ F1 score {f1_weighted:.4f} below threshold {min_f1:.4f}")
            passed = False
        else:
            logger.info(f"✅ F1 score {f1_weighted:.4f} meets threshold {min_f1:.4f}")
        
        if passed:
            logger.info("\n🎉 MODEL VALIDATION PASSED! Model is ready for deployment.")
        else:
            logger.error("\n❌ MODEL VALIDATION FAILED! Model does not meet requirements.")
        
        logger.info("="*80)
        
        return passed
        
    except Exception as e:
        logger.error(f"❌ Validation error: {e}", exc_info=True)
        return False

def main():
    parser = argparse.ArgumentParser(description="Validate model performance")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/lightgbm_production.pkl",
        help="Path to model file"
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default=None,
        help="Path to test data (optional)"
    )
    parser.add_argument(
        "--min_accuracy",
        type=float,
        default=0.80,
        help="Minimum required accuracy (default: 0.80)"
    )
    parser.add_argument(
        "--min_f1",
        type=float,
        default=0.75,
        help="Minimum required F1 score (default: 0.75)"
    )
    
    args = parser.parse_args()
    
    # Validate model
    passed = validate_model(
        model_path=args.model_path,
        test_data_path=args.test_data_path,
        min_accuracy=args.min_accuracy,
        min_f1=args.min_f1
    )
    
    # Exit with appropriate code
    sys.exit(0 if passed else 1)

if __name__ == "__main__":
    main()