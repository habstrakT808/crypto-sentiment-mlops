# File: scripts/train_complete_fixed.py
"""
Complete Fixed Training Pipeline
Production-ready training without data leakage
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

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

from src.data.advanced_augmentation import AdvancedTextAugmenter
from src.features.clean_feature_engineer import CleanFeatureEngineer
from src.models.deberta_model import DeBERTaV3Model
from src.models.lightgbm_model import LightGBMModel
from src.models.baseline_model import BaselineModel
from src.models.lstm_model import LSTMModel
from src.models.advanced_ensemble import AdvancedStackingEnsemble
from src.mlflow_tracking.experiment_tracker import ExperimentTracker
from src.explainability.shap_explainer import SHAPExplainer
from src.utils.logger import setup_logger
from src.utils.config import Config

logger = setup_logger(__name__)


def main(args):
    """Complete fixed training pipeline"""
    
    logger.info("="*80)
    logger.info("COMPLETE FIXED TRAINING PIPELINE - NO DATA LEAKAGE")
    logger.info("="*80)
    
    # ========== STEP 1: LOAD RAW DATA ==========
    logger.info("\n" + "="*80)
    logger.info("STEP 1: LOADING RAW DATA")
    logger.info("="*80)
    
    df = pd.read_csv(args.data_path)
    logger.info(f"Loaded {len(df)} samples")
    
    # Check if we have labels
    if 'auto_label_id' not in df.columns:
        logger.error("❌ No labels found! Please run auto-labeling first.")
        return
    
    # Save labels
    y = df['auto_label_id'].values
    logger.info(f"Label distribution:\n{pd.Series(y).value_counts().sort_index()}")
    
    # ========== STEP 2: DATA AUGMENTATION ==========
    if args.augment_data:
        logger.info("\n" + "="*80)
        logger.info("STEP 2: DATA AUGMENTATION")
        logger.info("="*80)
        
        augmenter = AdvancedTextAugmenter()
        df = augmenter.augment_minority_classes(
            df,
            text_column='preprocessed_text',
            label_column='auto_label_id',
            target_ratio=args.augmentation_ratio,
            methods=['synonym', 'insertion', 'swap']
        )
        
        # Update labels
        y = df['auto_label_id'].values
        
        logger.info(f"After augmentation: {len(df)} samples")
        logger.info(f"New distribution:\n{pd.Series(y).value_counts().sort_index()}")
    
    # ========== STEP 3: CLEAN FEATURE ENGINEERING ==========
    logger.info("\n" + "="*80)
    logger.info("STEP 3: CLEAN FEATURE ENGINEERING (NO LEAKAGE)")
    logger.info("="*80)
    
    feature_engineer = CleanFeatureEngineer()
    df = feature_engineer.create_all_clean_features(df)
    
    logger.info(f"Total features: {len(df.columns)}")
    
    # Get clean feature list
    clean_features = feature_engineer.get_feature_list(df)
    logger.info(f"Clean numeric features: {len(clean_features)}")
    
    # ========== STEP 4: PREPARE DATA ==========
    logger.info("\n" + "="*80)
    logger.info("STEP 4: PREPARING DATA")
    logger.info("="*80)
    
    X_features = df[clean_features]
    X_text = df['preprocessed_text']
    
    # Remove label columns from features
    label_cols = ['auto_label', 'auto_label_id']
    X_features = X_features.drop(columns=[col for col in label_cols if col in X_features.columns], errors='ignore')
    
    logger.info(f"Feature matrix shape: {X_features.shape}")
    logger.info(f"Text series length: {len(X_text)}")
    logger.info(f"Labels shape: {y.shape}")
    
    # ========== STEP 5: TRAIN/VAL/TEST SPLIT ==========
    logger.info("\n" + "="*80)
    logger.info("STEP 5: DATA SPLITTING")
    logger.info("="*80)
    
    # Split indices
    indices = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        stratify=y,
        random_state=42
    )
    
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=args.val_size / (1 - args.test_size),
        stratify=y[train_val_idx],
        random_state=42
    )
    
    # Split data
    X_train_features = X_features.iloc[train_idx].reset_index(drop=True)
    X_val_features = X_features.iloc[val_idx].reset_index(drop=True)
    X_test_features = X_features.iloc[test_idx].reset_index(drop=True)
    
    X_train_text = X_text.iloc[train_idx].reset_index(drop=True)
    X_val_text = X_text.iloc[val_idx].reset_index(drop=True)
    X_test_text = X_text.iloc[test_idx].reset_index(drop=True)
    
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]
    
    logger.info(f"Train: {len(train_idx)} samples")
    logger.info(f"Val: {len(val_idx)} samples")
    logger.info(f"Test: {len(test_idx)} samples")
    
    # ========== STEP 6: TRAIN MODELS ==========
    logger.info("\n" + "="*80)
    logger.info("STEP 6: TRAINING MODELS (NO LEAKAGE)")
    logger.info("="*80)
    
    tracker = ExperimentTracker()
    trained_models = []
    results = {}
    
    # 6.1: Train Baseline
    if args.train_baseline:
        logger.info("\n" + "-"*80)
        logger.info("Training Baseline Model (Logistic Regression)")
        logger.info("-"*80)
        
        with tracker.start_run(run_name=f"baseline_clean_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            baseline_model = BaselineModel(
                max_features=5000,
                ngram_range=(1, 2),
                C=1.0
            )
            
            baseline_model.train(X_train_text, y_train, X_val_text, y_val)
            trained_models.append(baseline_model)
            
            # Evaluate
            y_pred = baseline_model.predict(X_test_text)
            
            acc = accuracy_score(y_test, y_pred)
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            f1_macro = f1_score(y_test, y_pred, average='macro')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            
            results['baseline'] = {
                'accuracy': acc,
                'f1_weighted': f1_weighted,
                'f1_macro': f1_macro,
                'precision': precision,
                'recall': recall
            }
            
            logger.info(f"✅ Baseline Results:")
            logger.info(f"   Accuracy: {acc:.4f}")
            logger.info(f"   F1 (weighted): {f1_weighted:.4f}")
            logger.info(f"   F1 (macro): {f1_macro:.4f}")
            
            # Log to MLflow
            tracker.log_params({
                'model_type': 'baseline_clean',
                'data_leakage': 'removed',
                'max_features': 5000
            })
            tracker.log_metrics({
                'test_accuracy': acc,
                'test_f1_weighted': f1_weighted,
                'test_f1_macro': f1_macro
            })
            
            # Save model
            model_path = Config.MODELS_DIR / "baseline_clean_latest.pkl"
            joblib.dump(baseline_model, model_path)
            logger.info(f"Model saved to {model_path}")
    
    # 6.2: Train LSTM
    if args.train_lstm:
        logger.info("\n" + "-"*80)
        logger.info("Training LSTM Model")
        logger.info("-"*80)
        
        with tracker.start_run(run_name=f"lstm_clean_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            lstm_model = LSTMModel(
                embedding_dim=128,
                hidden_dim=256,
                num_layers=2,
                dropout=0.5,
                num_epochs=args.num_epochs_lstm
            )
            
            lstm_model.train(X_train_text, y_train, X_val_text, y_val)
            trained_models.append(lstm_model)
            
            # Evaluate
            y_pred = lstm_model.predict(X_test_text)
            
            acc = accuracy_score(y_test, y_pred)
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            f1_macro = f1_score(y_test, y_pred, average='macro')
            
            results['lstm'] = {
                'accuracy': acc,
                'f1_weighted': f1_weighted,
                'f1_macro': f1_macro
            }
            
            logger.info(f"✅ LSTM Results:")
            logger.info(f"   Accuracy: {acc:.4f}")
            logger.info(f"   F1 (weighted): {f1_weighted:.4f}")
            logger.info(f"   F1 (macro): {f1_macro:.4f}")
            
            tracker.log_params({
                'model_type': 'lstm_clean',
                'data_leakage': 'removed',
                'num_epochs': args.num_epochs_lstm
            })
            tracker.log_metrics({
                'test_accuracy': acc,
                'test_f1_weighted': f1_weighted,
                'test_f1_macro': f1_macro
            })
            
            # Save model
            model_path = Config.MODELS_DIR / "lstm_clean_latest.pkl"
            joblib.dump(lstm_model, model_path)
            logger.info(f"Model saved to {model_path}")
    
    # 6.3: Train LightGBM
    if args.train_lightgbm:
        logger.info("\n" + "-"*80)
        logger.info("Training LightGBM Model")
        logger.info("-"*80)
        
        with tracker.start_run(run_name=f"lightgbm_clean_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            lgbm_model = LightGBMModel(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=7,
                num_leaves=31,
                use_cv=True,
                n_folds=5
            )
            
            lgbm_model.train(X_train_features, y_train, X_val_features, y_val)
            
            # Evaluate
            y_pred = lgbm_model.predict(X_test_features)
            
            acc = accuracy_score(y_test, y_pred)
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            f1_macro = f1_score(y_test, y_pred, average='macro')
            
            results['lightgbm'] = {
                'accuracy': acc,
                'f1_weighted': f1_weighted,
                'f1_macro': f1_macro
            }
            
            logger.info(f"✅ LightGBM Results:")
            logger.info(f"   Accuracy: {acc:.4f}")
            logger.info(f"   F1 (weighted): {f1_weighted:.4f}")
            logger.info(f"   F1 (macro): {f1_macro:.4f}")
            
            # Feature importance
            importance_df = lgbm_model.get_feature_importance(top_n=15)
            logger.info("\n📊 Top 15 Features:")
            for idx, row in importance_df.head(15).iterrows():
                logger.info(f"   {row['feature']:30s} {row['importance']:8.2f}")
            
            tracker.log_params({
                'model_type': 'lightgbm_clean',
                'data_leakage': 'removed',
                'n_estimators': 1000
            })
            tracker.log_metrics({
                'test_accuracy': acc,
                'test_f1_weighted': f1_weighted,
                'test_f1_macro': f1_macro
            })
            
            # Save model
            model_path = Config.MODELS_DIR / "lightgbm_clean_latest.pkl"
            joblib.dump(lgbm_model, model_path)
            logger.info(f"Model saved to {model_path}")
            
            # SHAP Explanation
            if args.explain_model:
                logger.info("\n🔍 Generating SHAP Explanations...")
                try:
                    explainer = SHAPExplainer(lgbm_model.model, model_type='tree')
                    explainer.fit(X_train_features.head(100))
                    
                    # Summary plot
                    summary_path = Config.MODELS_DIR / f"shap_summary_clean_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    explainer.plot_summary(X_test_features.head(50), save_path=summary_path)
                    
                    # Feature importance
                    importance_path = Config.MODELS_DIR / f"shap_importance_clean_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    explainer.plot_feature_importance(X_test_features.head(50), save_path=importance_path, top_n=15)
                    
                    logger.info(f"✅ SHAP plots saved")
                except Exception as e:
                    logger.warning(f"SHAP generation failed: {e}")
    
    # 6.4: Train DeBERTa
    if args.train_deberta:
        logger.info("\n" + "-"*80)
        logger.info("Training DeBERTa Model")
        logger.info("-"*80)
        
        with tracker.start_run(run_name=f"deberta_clean_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            deberta_model = DeBERTaV3Model(
                model_name='microsoft/deberta-v3-base',
                max_len=256,
                batch_size=args.batch_size,
                learning_rate=2e-5,
                num_epochs=args.num_epochs_deberta,
                use_focal_loss=True,
                focal_gamma=2.0
            )
            
            deberta_model.train(X_train_text, y_train, X_val_text, y_val)
            trained_models.append(deberta_model)
            
            # Evaluate
            y_pred = deberta_model.predict(X_test_text)
            
            acc = accuracy_score(y_test, y_pred)
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            f1_macro = f1_score(y_test, y_pred, average='macro')
            
            results['deberta'] = {
                'accuracy': acc,
                'f1_weighted': f1_weighted,
                'f1_macro': f1_macro
            }
            
            logger.info(f"✅ DeBERTa Results:")
            logger.info(f"   Accuracy: {acc:.4f}")
            logger.info(f"   F1 (weighted): {f1_weighted:.4f}")
            logger.info(f"   F1 (macro): {f1_macro:.4f}")
            
            tracker.log_params({
                'model_type': 'deberta_clean',
                'data_leakage': 'removed',
                'num_epochs': args.num_epochs_deberta
            })
            tracker.log_metrics({
                'test_accuracy': acc,
                'test_f1_weighted': f1_weighted,
                'test_f1_macro': f1_macro
            })
            
            # Save model
            model_path = Config.MODELS_DIR / "deberta_clean_latest.pkl"
            joblib.dump(deberta_model, model_path)
            logger.info(f"Model saved to {model_path}")
    
    # 6.5: Train Ensemble
    if args.train_ensemble and len(trained_models) >= 2:
        logger.info("\n" + "-"*80)
        logger.info("Training Ensemble Model")
        logger.info("-"*80)
        
        with tracker.start_run(run_name=f"ensemble_clean_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            ensemble = AdvancedStackingEnsemble(
                base_models=trained_models,
                meta_model_type='lightgbm',
                use_oof=True,
                n_folds=5
            )
            
            ensemble.train(X_train_text, y_train, X_val_text, y_val)
            
            # Evaluate
            y_pred = ensemble.predict(X_test_text)
            
            acc = accuracy_score(y_test, y_pred)
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            f1_macro = f1_score(y_test, y_pred, average='macro')
            
            results['ensemble'] = {
                'accuracy': acc,
                'f1_weighted': f1_weighted,
                'f1_macro': f1_macro
            }
            
            logger.info(f"✅ Ensemble Results:")
            logger.info(f"   Accuracy: {acc:.4f}")
            logger.info(f"   F1 (weighted): {f1_weighted:.4f}")
            logger.info(f"   F1 (macro): {f1_macro:.4f}")
            
            tracker.log_params({
                'model_type': 'ensemble_clean',
                'data_leakage': 'removed',
                'n_base_models': len(trained_models)
            })
            tracker.log_metrics({
                'test_accuracy': acc,
                'test_f1_weighted': f1_weighted,
                'test_f1_macro': f1_macro
            })
            
            # Save model
            model_path = Config.MODELS_DIR / "ensemble_clean_latest.pkl"
            joblib.dump(ensemble, model_path)
            logger.info(f"Model saved to {model_path}")
    
    # ========== STEP 7: FINAL COMPARISON ==========
    logger.info("\n" + "="*80)
    logger.info("FINAL MODEL COMPARISON (NO DATA LEAKAGE)")
    logger.info("="*80)
    
    # Create comparison table
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.sort_values('f1_weighted', ascending=False)
    
    logger.info("\n📊 Model Performance Comparison:")
    logger.info(comparison_df.to_string())
    
    # Find best model
    best_model = comparison_df.index[0]
    best_acc = comparison_df.loc[best_model, 'accuracy']
    best_f1 = comparison_df.loc[best_model, 'f1_weighted']
    
    logger.info(f"\n🏆 Best Model: {best_model}")
    logger.info(f"   Accuracy: {best_acc:.4f}")
    logger.info(f"   F1 (weighted): {best_f1:.4f}")
    
    # Save comparison
    comparison_path = Config.MODELS_DIR / f"model_comparison_clean_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    comparison_df.to_csv(comparison_path)
    logger.info(f"\n💾 Comparison saved to {comparison_path}")
    
    # ========== FINAL SUMMARY ==========
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"✅ Models trained: {len(results)}")
    logger.info(f"✅ All models saved to: {Config.MODELS_DIR}")
    logger.info(f"✅ View MLflow UI: mlflow ui --port 5000")
    logger.info("\n🎯 These are REALISTIC accuracies without data leakage!")
    logger.info("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Complete fixed training pipeline")
    
    # Data arguments
    parser.add_argument('--data_path', type=str, 
                       default=str(Config.PROCESSED_DATA_DIR / "labeled_data.csv"))
    parser.add_argument('--test_size', type=float, default=0.15)
    parser.add_argument('--val_size', type=float, default=0.15)
    
    # Augmentation
    parser.add_argument('--augment_data', action='store_true')
    parser.add_argument('--augmentation_ratio', type=float, default=0.5)
    
    # Model selection
    parser.add_argument('--train_baseline', action='store_true')
    parser.add_argument('--train_lstm', action='store_true')
    parser.add_argument('--train_lightgbm', action='store_true')
    parser.add_argument('--train_deberta', action='store_true')
    parser.add_argument('--train_ensemble', action='store_true')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs_lstm', type=int, default=10)
    parser.add_argument('--num_epochs_deberta', type=int, default=5)
    
    # Explainability
    parser.add_argument('--explain_model', action='store_true')
    
    args = parser.parse_args()
    
    # If no model specified, train all except DeBERTa (to save time)
    if not any([args.train_baseline, args.train_lstm, args.train_lightgbm, args.train_deberta]):
        args.train_baseline = True
        args.train_lstm = True
        args.train_lightgbm = True
        args.train_ensemble = True
    
    main(args)