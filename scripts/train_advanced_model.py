# File: scripts/train_advanced_model.py
"""
Advanced Model Training Script
Complete pipeline with all advanced features
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
from sklearn.metrics import accuracy_score, f1_score
warnings.filterwarnings('ignore')

from src.data.advanced_augmentation import AdvancedTextAugmenter
from src.features.advanced_feature_engineer import AdvancedFeatureEngineer
from src.models.deberta_model import DeBERTaV3Model
from src.models.lightgbm_model import LightGBMModel
from src.models.advanced_ensemble import AdvancedStackingEnsemble
from src.models.baseline_model import BaselineModel
from src.models.lstm_model import LSTMModel
from src.optimization.hyperparameter_tuner import HyperparameterTuner
from src.explainability.shap_explainer import SHAPExplainer
from src.mlflow_tracking.experiment_tracker import ExperimentTracker
from src.evaluation.evaluator import ModelEvaluator
from src.utils.logger import setup_logger
from src.utils.config import Config

logger = setup_logger(__name__)


def main(args):
    """Main training pipeline"""
    logger.info("="*80)
    logger.info("ADVANCED CRYPTOCURRENCY SENTIMENT ANALYSIS - TRAINING PIPELINE")
    logger.info("="*80)
    
    # ========== STEP 1: LOAD DATA ==========
    logger.info("\n" + "="*80)
    logger.info("STEP 1: LOADING DATA")
    logger.info("="*80)
    
    df = pd.read_csv(args.data_path)
    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"Label distribution:\n{df['auto_label_id'].value_counts()}")
    
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
        
        logger.info(f"After augmentation: {len(df)} samples")
        logger.info(f"New distribution:\n{df['auto_label_id'].value_counts()}")
    
    # ========== STEP 3: ADVANCED FEATURE ENGINEERING ==========
    logger.info("\n" + "="*80)
    logger.info("STEP 3: ADVANCED FEATURE ENGINEERING")
    logger.info("="*80)
    
    feature_engineer = AdvancedFeatureEngineer()
    df = feature_engineer.create_all_advanced_features(df)
    
    logger.info(f"Total features created: {len(df.columns)}")
    
    # ========== STEP 4: TRAIN/VAL/TEST SPLIT ==========
    logger.info("\n" + "="*80)
    logger.info("STEP 4: DATA SPLITTING")
    logger.info("="*80)
    
    from sklearn.model_selection import train_test_split
    
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        stratify=df['auto_label_id'],
        random_state=42
    )
    
    # Second split: train vs val
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=args.val_size / (1 - args.test_size),
        stratify=train_val_df['auto_label_id'],
        random_state=42
    )
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Prepare data for different models
    X_train_text = train_df['preprocessed_text']
    X_val_text = val_df['preprocessed_text']
    X_test_text = test_df['preprocessed_text']
    
    X_train_features = train_df
    X_val_features = val_df
    X_test_features = test_df
    
    y_train = train_df['auto_label_id'].values
    y_val = val_df['auto_label_id'].values
    y_test = test_df['auto_label_id'].values
    
    # ========== STEP 5: HYPERPARAMETER TUNING (OPTIONAL) ==========
    if args.tune_hyperparameters:
        logger.info("\n" + "="*80)
        logger.info("STEP 5: HYPERPARAMETER TUNING")
        logger.info("="*80)
        
        tuner = HyperparameterTuner(n_trials=args.n_trials, n_folds=5)
        
        # Tune LightGBM
        logger.info("Tuning LightGBM...")
        best_lgbm_params = tuner.tune_lightgbm(X_train_features, y_train)
        
        # Save tuning results
        tuning_results_path = Config.MODELS_DIR / f"tuning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(tuning_results_path, 'w') as f:
            f.write(f"Best LightGBM Parameters:\n{best_lgbm_params}\n")
        
        logger.info(f"Tuning results saved to {tuning_results_path}")
    
    # ========== STEP 6: TRAIN BASE MODELS ==========
    logger.info("\n" + "="*80)
    logger.info("STEP 6: TRAINING BASE MODELS")
    logger.info("="*80)
    
    base_models = []
    
    # Initialize MLflow tracker
    tracker = ExperimentTracker()
    
    # 6.1: Train Baseline Model
    if args.train_baseline:
        logger.info("\n--- Training Baseline Model ---")
        
        with tracker.start_run(run_name=f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            baseline_model = BaselineModel(
                max_features=5000,
                ngram_range=(1, 2),
                C=1.0
            )
            
            baseline_model.train(X_train_text, y_train, X_val_text, y_val)
            base_models.append(baseline_model)
            
            # Evaluate
            y_pred = baseline_model.predict(X_test_text)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            logger.info(f"Baseline - Test Accuracy: {acc:.4f}, F1: {f1:.4f}")
            
            tracker.log_params({'model_type': 'baseline'})
            tracker.log_metrics({'test_accuracy': acc, 'test_f1_weighted': f1})
    
    # 6.2: Train LSTM Model
    if args.train_lstm:
        logger.info("\n--- Training LSTM Model ---")
        
        with tracker.start_run(run_name=f"lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            lstm_model = LSTMModel(
                embedding_dim=128,
                hidden_dim=256,
                num_layers=2,
                dropout=0.5,
                num_epochs=args.num_epochs_lstm
            )
            
            lstm_model.train(X_train_text, y_train, X_val_text, y_val)
            base_models.append(lstm_model)
            
            # Evaluate
            y_pred = lstm_model.predict(X_test_text)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            logger.info(f"LSTM - Test Accuracy: {acc:.4f}, F1: {f1:.4f}")
            
            tracker.log_params({'model_type': 'lstm'})
            tracker.log_metrics({'test_accuracy': acc, 'test_f1_weighted': f1})
    
    # 6.3: Train DeBERTaV3 Model
    if args.train_deberta:
        logger.info("\n--- Training DeBERTaV3 Model ---")
        
        with tracker.start_run(run_name=f"deberta_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
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
            base_models.append(deberta_model)
            
            # Evaluate
            y_pred = deberta_model.predict(X_test_text)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            logger.info(f"DeBERTa - Test Accuracy: {acc:.4f}, F1: {f1:.4f}")
            
            tracker.log_params({'model_type': 'deberta_v3'})
            tracker.log_metrics({'test_accuracy': acc, 'test_f1_weighted': f1})
    
    # 6.4: Train LightGBM Model
    if args.train_lightgbm:
        logger.info("\n--- Training LightGBM Model ---")
        
        with tracker.start_run(run_name=f"lightgbm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            lgbm_model = LightGBMModel(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=7,
                use_cv=True,
                n_folds=5
            )
            
            lgbm_model.train(X_train_features, y_train, X_val_features, y_val)
            base_models.append(lgbm_model)
            
            # Evaluate
            y_pred = lgbm_model.predict(X_test_features)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            logger.info(f"LightGBM - Test Accuracy: {acc:.4f}, F1: {f1:.4f}")
            
            tracker.log_params({'model_type': 'lightgbm'})
            tracker.log_metrics({'test_accuracy': acc, 'test_f1_weighted': f1})
            
            # Plot feature importance
            importance_df = lgbm_model.get_feature_importance(top_n=20)
            logger.info(f"\nTop 10 Features:\n{importance_df.head(10)}")
    
    # ========== STEP 7: TRAIN ENSEMBLE ==========
    if args.train_ensemble and len(base_models) >= 2:
        logger.info("\n" + "="*80)
        logger.info("STEP 7: TRAINING STACKING ENSEMBLE")
        logger.info("="*80)
        
        with tracker.start_run(run_name=f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Note: Ensemble needs consistent input format
            # For simplicity, we'll use text-based models only
            text_models = [m for m in base_models if hasattr(m, 'tokenizer') or 'baseline' in m.model_name or 'lstm' in m.model_name]
            
            if len(text_models) >= 2:
                ensemble = AdvancedStackingEnsemble(
                    base_models=text_models,
                    meta_model_type='lightgbm',
                    use_oof=True,
                    n_folds=5
                )
                
                ensemble.train(X_train_text, y_train, X_val_text, y_val)
                
                # Evaluate
                y_pred = ensemble.predict(X_test_text)
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                logger.info(f"Ensemble - Test Accuracy: {acc:.4f}, F1: {f1:.4f}")
                
                tracker.log_params({'model_type': 'stacking_ensemble', 'n_base_models': len(text_models)})
                tracker.log_metrics({'test_accuracy': acc, 'test_f1_weighted': f1})
                
                # Predict with uncertainty
                y_pred_unc, uncertainties = ensemble.predict_with_uncertainty(X_test_text, n_iterations=10)
                logger.info(f"Mean prediction uncertainty: {np.mean(uncertainties):.4f}")
    
    # ========== STEP 8: MODEL EXPLAINABILITY ==========
    if args.explain_model and args.train_lightgbm:
        logger.info("\n" + "="*80)
        logger.info("STEP 8: MODEL EXPLAINABILITY (SHAP)")
        logger.info("="*80)
        
        explainer = SHAPExplainer(lgbm_model.model, model_type='tree')
        explainer.fit(X_train_features, max_samples=100)
        
        # Plot summary
        summary_path = Config.MODELS_DIR / f"shap_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        explainer.plot_summary(X_test_features.head(100), save_path=summary_path)
        
        # Plot feature importance
        importance_path = Config.MODELS_DIR / f"shap_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        importance_df = explainer.plot_feature_importance(X_test_features.head(100), save_path=importance_path, top_n=20)
        
        logger.info(f"SHAP plots saved to {Config.MODELS_DIR}")
    
    # ========== FINAL SUMMARY ==========
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"Models trained: {len(base_models)}")
    logger.info(f"All models saved to: {Config.MODELS_DIR}")
    logger.info(f"View MLflow UI: mlflow ui --port 5000")
    logger.info("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train advanced sentiment models")
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default=str(Config.PROCESSED_DATA_DIR / "labeled_data.csv"))
    parser.add_argument('--test_size', type=float, default=0.15)
    parser.add_argument('--val_size', type=float, default=0.15)
    
    # Augmentation arguments
    parser.add_argument('--augment_data', action='store_true', help="Apply data augmentation")
    parser.add_argument('--augmentation_ratio', type=float, default=0.5, help="Target ratio for minority classes")
    
    # Model selection
    parser.add_argument('--train_baseline', action='store_true', help="Train baseline model")
    parser.add_argument('--train_lstm', action='store_true', help="Train LSTM model")
    parser.add_argument('--train_deberta', action='store_true', help="Train DeBERTa model")
    parser.add_argument('--train_lightgbm', action='store_true', help="Train LightGBM model")
    parser.add_argument('--train_ensemble', action='store_true', help="Train ensemble model")
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs_lstm', type=int, default=10)
    parser.add_argument('--num_epochs_deberta', type=int, default=5)
    
    # Hyperparameter tuning
    parser.add_argument('--tune_hyperparameters', action='store_true', help="Run hyperparameter tuning")
    parser.add_argument('--n_trials', type=int, default=50, help="Number of Optuna trials")
    
    # Explainability
    parser.add_argument('--explain_model', action='store_true', help="Generate SHAP explanations")
    
    args = parser.parse_args()
    
    # If no model specified, train all
    if not any([args.train_baseline, args.train_lstm, args.train_deberta, args.train_lightgbm]):
        args.train_baseline = True
        args.train_lstm = True
        args.train_deberta = True
        args.train_lightgbm = True
        args.train_ensemble = True
    
    main(args)