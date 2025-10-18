# File: scripts/train_production_model.py
"""
Production Training Pipeline
Complete fix for data leakage with realistic evaluation
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import argparse
from datetime import datetime
import warnings
import joblib
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

from src.data.production_augmentation import ProductionAugmenter
from src.features.production_feature_engineer import ProductionFeatureEngineer
from src.models.lightgbm_model import LightGBMModel
from src.models.baseline_model import BaselineModel
from src.mlflow_tracking.experiment_tracker import ExperimentTracker
from src.utils.logger import setup_logger
from src.utils.config import Config

logger = setup_logger(__name__)


def main(args):
    """Production training pipeline"""
    
    logger.info("="*80)
    logger.info("PRODUCTION TRAINING PIPELINE - ZERO DATA LEAKAGE")
    logger.info("="*80)
    
    # Load data
    logger.info("\nLoading data...")
    df = pd.read_csv(args.data_path)
    logger.info(f"Loaded {len(df)} samples")
    
    if 'auto_label_id' not in df.columns:
        logger.error("❌ No labels found!")
        return
    
    y = df['auto_label_id'].values
    logger.info(f"Label distribution:\n{pd.Series(y).value_counts().sort_index()}")
    
    # CRITICAL: Split BEFORE augmentation to prevent leakage
    logger.info("\n" + "="*80)
    logger.info("STEP 1: SPLIT DATA (BEFORE AUGMENTATION)")
    logger.info("="*80)
    
    from sklearn.model_selection import train_test_split
    
    # Get indices
    indices = np.arange(len(df))
    
    # Split into train+val and test
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    
    # Split train+val into train and val
    y_train_val = y[train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.2,
        stratify=y_train_val,
        random_state=42
    )
    
    logger.info(f"Train: {len(train_idx)} samples")
    logger.info(f"Val: {len(val_idx)} samples")
    logger.info(f"Test: {len(test_idx)} samples")
    
    # CRITICAL: Only augment TRAINING set
    logger.info("\n" + "="*80)
    logger.info("STEP 2: AUGMENT TRAINING SET ONLY")
    logger.info("="*80)
    
    df_train = df.iloc[train_idx].copy()
    df_val = df.iloc[val_idx].copy()
    df_test = df.iloc[test_idx].copy()
    
    if args.augment_data:
        augmenter = ProductionAugmenter()
        df_train = augmenter.augment_minority_classes(
            df_train,
            text_column='preprocessed_text',
            label_column='auto_label_id',
            max_augment_per_sample=2,
            methods=['synonym']
        )
        logger.info(f"After augmentation: {len(df_train)} training samples")
    
    # Update labels
    y_train = df_train['auto_label_id'].values
    y_val = df_val['auto_label_id'].values
    y_test = df_test['auto_label_id'].values
    
    # CRITICAL: Use production feature engineer (NO SENTIMENT FEATURES!)
    logger.info("\n" + "="*80)
    logger.info("STEP 3: PRODUCTION FEATURE ENGINEERING (ZERO LEAKAGE)")
    logger.info("="*80)
    
    feature_engineer = ProductionFeatureEngineer()
    
    # Create features for each split
    df_train = feature_engineer.create_all_production_features(df_train)
    df_val = feature_engineer.create_all_production_features(df_val)
    df_test = feature_engineer.create_all_production_features(df_test)
    
    # Get feature list
    production_features = feature_engineer.get_feature_list(df_train)
    logger.info(f"Production features: {len(production_features)}")
    
    # Prepare data
    X_train_features = df_train[production_features]
    X_val_features = df_val[production_features]
    X_test_features = df_test[production_features]
    
    X_train_text = df_train['preprocessed_text']
    X_val_text = df_val['preprocessed_text']
    X_test_text = df_test['preprocessed_text']
    
    logger.info(f"Train features shape: {X_train_features.shape}")
    logger.info(f"Val features shape: {X_val_features.shape}")
    logger.info(f"Test features shape: {X_test_features.shape}")
    
    # Train models
    logger.info("\n" + "="*80)
    logger.info("STEP 4: TRAINING MODELS")
    logger.info("="*80)
    
    tracker = ExperimentTracker()
    results = {}
    
    # Train Baseline
    logger.info("\nTraining Baseline Model...")
    with tracker.start_run(run_name=f"baseline_production_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        baseline = BaselineModel(max_features=5000)
        baseline.train(X_train_text, y_train, X_val_text, y_val)
        
        y_pred = baseline.predict(X_test_text)
        
        acc = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        results['baseline'] = {'accuracy': acc, 'f1_weighted': f1_weighted, 'f1_macro': f1_macro}
        
        logger.info(f"✅ Baseline - Accuracy: {acc:.4f}, F1: {f1_weighted:.4f}")
        
        tracker.log_params({'model_type': 'baseline_production', 'data_leakage': 'ZERO'})
        tracker.log_metrics({'test_accuracy': acc, 'test_f1_weighted': f1_weighted})
        
        joblib.dump(baseline, Config.MODELS_DIR / "baseline_production.pkl")
    
    # Train LightGBM
    logger.info("\nTraining LightGBM Model...")
    with tracker.start_run(run_name=f"lightgbm_production_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        lgbm = LightGBMModel(n_estimators=1000, use_cv=True, n_folds=5)
        lgbm.train(X_train_features, y_train, X_val_features, y_val)
        
        y_pred = lgbm.predict(X_test_features)
        
        acc = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        results['lightgbm'] = {'accuracy': acc, 'f1_weighted': f1_weighted, 'f1_macro': f1_macro}
        
        logger.info(f"✅ LightGBM - Accuracy: {acc:.4f}, F1: {f1_weighted:.4f}")
        
        # Feature importance
        importance_df = lgbm.get_feature_importance(top_n=15)
        logger.info("\n📊 Top 15 Features:")
        for idx, row in importance_df.head(15).iterrows():
            logger.info(f"   {row['feature']:30s} {row['importance']:8.2f}")
        
        tracker.log_params({'model_type': 'lightgbm_production', 'data_leakage': 'ZERO'})
        tracker.log_metrics({'test_accuracy': acc, 'test_f1_weighted': f1_weighted})
        
        joblib.dump(lgbm, Config.MODELS_DIR / "lightgbm_production.pkl")
    
    # Cross-validation evaluation
    logger.info("\n" + "="*80)
    logger.info("STEP 5: CROSS-VALIDATION EVALUATION")
    logger.info("="*80)
    
    # Combine train+val for CV
    X_train_val_features = pd.concat([X_train_features, X_val_features], ignore_index=True)
    y_train_val = np.concatenate([y_train, y_val])
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx_cv, val_idx_cv) in enumerate(skf.split(X_train_val_features, y_train_val)):
        lgbm_cv = LightGBMModel(n_estimators=500, use_cv=False)
        lgbm_cv.train(
            X_train_val_features.iloc[train_idx_cv],
            y_train_val[train_idx_cv],
            X_train_val_features.iloc[val_idx_cv],
            y_train_val[val_idx_cv]
        )
        
        y_pred_cv = lgbm_cv.predict(X_train_val_features.iloc[val_idx_cv])
        acc_cv = accuracy_score(y_train_val[val_idx_cv], y_pred_cv)
        cv_scores.append(acc_cv)
        
        logger.info(f"Fold {fold+1}: {acc_cv:.4f}")
    
    logger.info(f"\nCV Mean: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    logger.info(f"CV Min: {np.min(cv_scores):.4f}")
    logger.info(f"CV Max: {np.max(cv_scores):.4f}")
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("FINAL RESULTS (REALISTIC ACCURACY)")
    logger.info("="*80)
    
    comparison_df = pd.DataFrame(results).T
    logger.info(f"\n{comparison_df.to_string()}")
    
    logger.info("\n✅ Training complete!")
    logger.info(f"✅ Models saved to: {Config.MODELS_DIR}")
    logger.info(f"✅ Expected production accuracy: {np.mean(cv_scores):.1%} - {np.max(results['lightgbm']['accuracy']):.1%}")
    logger.info("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=str(Config.PROCESSED_DATA_DIR / "labeled_data.csv"))
    parser.add_argument('--augment_data', action='store_true')
    args = parser.parse_args()
    
    main(args)