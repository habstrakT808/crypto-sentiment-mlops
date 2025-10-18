# File: scripts/retrain_with_balanced_data.py
"""
🎯 Retrain Model with Balanced Data
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.features.production_feature_engineer import ProductionFeatureEngineer
from src.models.baseline_model import BaselineModel
from src.models.lightgbm_model import LightGBMModel
from src.utils.logger import setup_logger
from src.utils.config import Config

logger = setup_logger(__name__)

def main():
    """Retrain with balanced data"""
    
    logger.info("="*80)
    logger.info("RETRAINING WITH BALANCED DATA")
    logger.info("="*80)
    
    # Load balanced data
    processed_files = list(Config.PROCESSED_DATA_DIR.glob("labeled_balanced_*.csv"))
    if not processed_files:
        logger.error("❌ No balanced data found! Run balance_dataset_smart.py first")
        return
    
    latest_file = max(processed_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading data from {latest_file}...")
    
    df = pd.read_csv(latest_file)
    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"Label distribution:\n{df['auto_label_id'].value_counts()}")
    
    # Feature engineering
    logger.info("\n🔧 Creating features...")
    feature_engineer = ProductionFeatureEngineer()
    df = feature_engineer.create_all_production_features(df)
    
    # Prepare data
    feature_list = feature_engineer.get_feature_list(df)
    X = df[feature_list]
    y = df['auto_label_id'].values
    X_text = df['full_text'] if 'full_text' in df.columns else df['preprocessed_text']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    X_train_text, X_test_text = train_test_split(
        X_text, test_size=0.2, stratify=y, random_state=42
    )
    
    logger.info(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    # Train Baseline
    logger.info("\n🤖 Training Baseline Model...")
    baseline = BaselineModel(max_features=5000)
    baseline.train(X_train_text, y_train)
    
    y_pred = baseline.predict(X_test_text)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    logger.info(f"✅ Baseline - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive'])}")
    
    # Save
    model_path = Config.MODELS_DIR / "baseline_balanced_latest.pkl"
    joblib.dump(baseline, model_path)
    logger.info(f"✅ Model saved to {model_path}")
    
    # Train LightGBM
    logger.info("\n🤖 Training LightGBM Model...")
    lgbm = LightGBMModel(n_estimators=500, use_cv=False)
    lgbm.train(X_train, y_train)
    
    y_pred = lgbm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    logger.info(f"✅ LightGBM - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive'])}")
    
    # Save
    model_path = Config.MODELS_DIR / "lightgbm_balanced_latest.pkl"
    joblib.dump(lgbm, model_path)
    logger.info(f"✅ Model saved to {model_path}")
    
    logger.info("\n" + "="*80)
    logger.info("✅ RETRAINING COMPLETE!")
    logger.info("="*80)

if __name__ == "__main__":
    main()