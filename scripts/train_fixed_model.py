# File: scripts/train_fixed_model.py
"""
Fixed Training Pipeline - No Data Leakage
Train models using only legitimate features
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
warnings.filterwarnings('ignore')

from src.data.advanced_augmentation import AdvancedTextAugmenter
from src.features.advanced_feature_engineer import AdvancedFeatureEngineer
from src.models.deberta_model import DeBERTaV3Model
from src.models.lightgbm_model import LightGBMModel
from src.models.baseline_model import BaselineModel
from src.models.lstm_model import LSTMModel
from src.mlflow_tracking.experiment_tracker import ExperimentTracker
from src.evaluation.evaluator import ModelEvaluator
from src.utils.logger import setup_logger
from src.utils.config import Config

logger = setup_logger(__name__)


# ========== FEATURE BLACKLIST (PREVENT DATA LEAKAGE) ==========
LEAKAGE_FEATURES = [
    # Auto-labeling metadata (REMOVE!)
    'label_confidence',
    'label_agreement',
    'auto_label',
    'auto_label_id',
    
    # Ensemble model outputs (REMOVE!)
    'score_negative',
    'score_neutral',
    'score_positive',
    
    # Individual model predictions (REMOVE!)
    'textblob_prediction',
    'vader_prediction',
    'finbert_prediction',
    
    # VADER sentiment scores (REMOVE!)
    'vader_compound',
    'vader_pos',
    'vader_neg',
    'vader_neu',
    
    # Any column containing 'label' or 'sentiment' in name
]


def remove_leakage_features(df: pd.DataFrame) -> pd.DataFrame:
    """Remove features that cause data leakage"""
    logger.info("Removing data leakage features...")
    
    original_cols = len(df.columns)
    
    # Remove explicit leakage features
    cols_to_remove = [col for col in LEAKAGE_FEATURES if col in df.columns]
    
    # Remove any column with 'label' or 'prediction' in name
    cols_to_remove += [col for col in df.columns if 'label' in col.lower() or 'prediction' in col.lower()]
    
    # Remove duplicates
    cols_to_remove = list(set(cols_to_remove))
    
    if cols_to_remove:
        logger.info(f"Removing {len(cols_to_remove)} leakage features:")
        for col in cols_to_remove:
            logger.info(f"  - {col}")
        
        df = df.drop(columns=cols_to_remove, errors='ignore')
    
    logger.info(f"Features before: {original_cols}, after: {len(df.columns)}")
    
    return df


def get_legitimate_features(df: pd.DataFrame) -> list:
    """Get list of legitimate features (no leakage)"""
    
    # Legitimate feature categories
    legitimate_features = []
    
    # 1. Reddit metadata (legitimate)
    reddit_features = ['score', 'upvote_ratio', 'num_comments', 'is_self']
    legitimate_features.extend([f for f in reddit_features if f in df.columns])
    
    # 2. Text statistics (legitimate)
    text_features = [
        'text_length', 'word_count', 'char_count', 'sentence_count',
        'avg_word_length', 'unique_word_count', 'lexical_diversity',
        'avg_sentence_length', 'text_density', 'readability_score'
    ]
    legitimate_features.extend([f for f in text_features if f in df.columns])
    
    # 3. Linguistic features (legitimate)
    linguistic_features = [
        'bullish_keyword_count', 'bearish_keyword_count', 'uncertainty_score',
        'urgency_score', 'bullish_bearish_ratio'
    ]
    legitimate_features.extend([f for f in linguistic_features if f in df.columns])
    
    # 4. Readability (legitimate)
    readability_features = ['flesch_reading_ease', 'flesch_kincaid_grade']
    legitimate_features.extend([f for f in readability_features if f in df.columns])
    
    # 5. Price/Emoji/Crypto mentions (legitimate)
    mention_features = [
        'has_price_mention', 'has_percentage_mention', 'price_mention_count',
        'positive_emoji_count', 'negative_emoji_count', 'emoji_sentiment',
        'mentions_btc', 'mentions_eth', 'mentions_ada', 'mentions_sol', 'mentions_doge',
        'total_crypto_mentions', 'hashtag_count'
    ]
    legitimate_features.extend([f for f in mention_features if f in df.columns])
    
    # 6. Temporal features (legitimate)
    temporal_features = [
        'hour', 'day_of_week', 'day_of_month', 'month',
        'is_weekend', 'is_business_hours'
    ]
    legitimate_features.extend([f for f in temporal_features if f in df.columns])
    
    # 7. Engagement features (legitimate)
    engagement_features = [
        'engagement_ratio', 'controversy_score', 'log_score',
        'log_comments', 'viral_potential'
    ]
    legitimate_features.extend([f for f in engagement_features if f in df.columns])
    
    # Remove duplicates
    legitimate_features = list(set(legitimate_features))
    
    logger.info(f"Selected {len(legitimate_features)} legitimate features")
    
    return legitimate_features


def main(args):
    """Main training pipeline with fixed data leakage"""
    logger.info("="*80)
    logger.info("FIXED TRAINING PIPELINE - NO DATA LEAKAGE")
    logger.info("="*80)
    
    # ========== STEP 1: LOAD DATA ==========
    logger.info("\n" + "="*80)
    logger.info("STEP 1: LOADING DATA")
    logger.info("="*80)
    
    df = pd.read_csv(args.data_path)
    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"Original features: {len(df.columns)}")
    
    # ========== STEP 2: REMOVE DATA LEAKAGE ==========
    logger.info("\n" + "="*80)
    logger.info("STEP 2: REMOVING DATA LEAKAGE FEATURES")
    logger.info("="*80)
    
    # Save label before removing
    y = df['auto_label_id'].values
    
    # Remove leakage features
    df = remove_leakage_features(df)
    
    logger.info(f"Remaining features: {len(df.columns)}")
    
    # ========== STEP 3: DATA AUGMENTATION (OPTIONAL) ==========
    if args.augment_data:
        logger.info("\n" + "="*80)
        logger.info("STEP 3: DATA AUGMENTATION")
        logger.info("="*80)
        
        # Re-add label for augmentation
        df['auto_label_id'] = y
        
        augmenter = AdvancedTextAugmenter()
        df = augmenter.augment_minority_classes(
            df,
            text_column='preprocessed_text',
            label_column='auto_label_id',
            target_ratio=args.augmentation_ratio,
            methods=['synonym', 'insertion', 'swap']
        )
        
        logger.info(f"After augmentation: {len(df)} samples")
        
        # Update y
        y = df['auto_label_id'].values
        
        # Remove label again
        df = df.drop(columns=['auto_label_id'])
    
    # ========== STEP 4: FEATURE ENGINEERING ==========
    logger.info("\n" + "="*80)
    logger.info("STEP 4: FEATURE ENGINEERING (WITHOUT LEAKAGE)")
    logger.info("="*80)
    
    feature_engineer = AdvancedFeatureEngineer()
    
    # Only create features that don't cause leakage
    # Skip VADER features (they were used in auto-labeling)
    df = feature_engineer.create_financial_sentiment_features(df)
    df = feature_engineer.create_readability_features(df)
    df = feature_engineer.create_price_mention_features(df)
    df = feature_engineer.create_emoji_features(df)
    df = feature_engineer.create_crypto_specific_features(df)
    
    logger.info(f"Total features after engineering: {len(df.columns)}")
    
    # Remove any accidentally created leakage features
    df = remove_leakage_features(df)
    
    # ========== STEP 5: SELECT LEGITIMATE FEATURES ==========
    logger.info("\n" + "="*80)
    logger.info("STEP 5: SELECTING LEGITIMATE FEATURES")
    logger.info("="*80)
    
    legitimate_features = get_legitimate_features(df)
    
    # Filter to only numeric features for LightGBM
    numeric_features = df[legitimate_features].select_dtypes(include=[np.number]).columns.tolist()
    
    logger.info(f"Using {len(numeric_features)} numeric features:")
    for i, col in enumerate(numeric_features, 1):
        logger.info(f"  {i:2d}. {col}")
    
    X_features = df[numeric_features]
    X_text = df['preprocessed_text']
    
    # ========== STEP 6: TRAIN/VAL/TEST SPLIT ==========
    logger.info("\n" + "="*80)
    logger.info("STEP 6: DATA SPLITTING")
    logger.info("="*80)
    
    from sklearn.model_selection import train_test_split
    
    # First split: train+val vs test
    indices = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        stratify=y,
        random_state=42
    )
    
    # Second split: train vs val
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=args.val_size / (1 - args.test_size),
        stratify=y[train_val_idx],
        random_state=42
    )
    
    # Split data
    X_train_features = X_features.iloc[train_idx]
    X_val_features = X_features.iloc[val_idx]
    X_test_features = X_features.iloc[test_idx]
    
    X_train_text = X_text.iloc[train_idx]
    X_val_text = X_text.iloc[val_idx]
    X_test_text = X_text.iloc[test_idx]
    
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]
    
    logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # ========== STEP 7: TRAIN MODELS ==========
    logger.info("\n" + "="*80)
    logger.info("STEP 7: TRAINING MODELS (NO LEAKAGE)")
    logger.info("="*80)
    
    tracker = ExperimentTracker()
    results = {}
    
    # 7.1: Train Baseline
    if args.train_baseline:
        logger.info("\n--- Training Baseline Model ---")
        
        with tracker.start_run(run_name=f"baseline_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            baseline_model = BaselineModel(max_features=5000, ngram_range=(1, 2))
            baseline_model.train(X_train_text, y_train, X_val_text, y_val)
            
            # Evaluate
            y_pred = baseline_model.predict(X_test_text)
            from sklearn.metrics import accuracy_score, f1_score
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            logger.info(f"Baseline - Test Accuracy: {acc:.4f}, F1: {f1:.4f}")
            results['baseline'] = {'accuracy': acc, 'f1': f1}
            
            tracker.log_params({'model_type': 'baseline_fixed', 'data_leakage': 'removed'})
            tracker.log_metrics({'test_accuracy': acc, 'test_f1_weighted': f1})
    
    # 7.2: Train LSTM
    if args.train_lstm:
        logger.info("\n--- Training LSTM Model ---")
        
        with tracker.start_run(run_name=f"lstm_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            lstm_model = LSTMModel(
                embedding_dim=128,
                hidden_dim=256,
                num_layers=2,
                num_epochs=args.num_epochs_lstm
            )
            lstm_model.train(X_train_text, y_train, X_val_text, y_val)
            
            # Evaluate
            y_pred = lstm_model.predict(X_test_text)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            logger.info(f"LSTM - Test Accuracy: {acc:.4f}, F1: {f1:.4f}")
            results['lstm'] = {'accuracy': acc, 'f1': f1}
            
            tracker.log_params({'model_type': 'lstm_fixed', 'data_leakage': 'removed'})
            tracker.log_metrics({'test_accuracy': acc, 'test_f1_weighted': f1})
    
    # 7.3: Train LightGBM
    if args.train_lightgbm:
        logger.info("\n--- Training LightGBM Model ---")
        
        with tracker.start_run(run_name=f"lightgbm_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            lgbm_model = LightGBMModel(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=7,
                use_cv=True,
                n_folds=5
            )
            lgbm_model.train(X_train_features, y_train, X_val_features, y_val)
            
            # Evaluate
            y_pred = lgbm_model.predict(X_test_features)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            logger.info(f"LightGBM - Test Accuracy: {acc:.4f}, F1: {f1:.4f}")
            results['lightgbm'] = {'accuracy': acc, 'f1': f1}
            
            # Feature importance
            importance_df = lgbm_model.get_feature_importance(top_n=10)
            logger.info("\nTop 10 Features:")
            logger.info(importance_df.to_string())
            
            tracker.log_params({'model_type': 'lightgbm_fixed', 'data_leakage': 'removed'})
            tracker.log_metrics({'test_accuracy': acc, 'test_f1_weighted': f1})
    
    # 7.4: Train DeBERTa
    if args.train_deberta:
        logger.info("\n--- Training DeBERTa Model ---")
        
        with tracker.start_run(run_name=f"deberta_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            deberta_model = DeBERTaV3Model(
                model_name='microsoft/deberta-v3-base',
                max_len=256,
                batch_size=args.batch_size,
                num_epochs=args.num_epochs_deberta,
                use_focal_loss=True
            )
            deberta_model.train(X_train_text, y_train, X_val_text, y_val)
            
            # Evaluate
            y_pred = deberta_model.predict(X_test_text)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            logger.info(f"DeBERTa - Test Accuracy: {acc:.4f}, F1: {f1:.4f}")
            results['deberta'] = {'accuracy': acc, 'f1': f1}
            
            tracker.log_params({'model_type': 'deberta_fixed', 'data_leakage': 'removed'})
            tracker.log_metrics({'test_accuracy': acc, 'test_f1_weighted': f1})
    
    # ========== FINAL SUMMARY ==========
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE - FIXED PIPELINE (NO DATA LEAKAGE)")
    logger.info("="*80)
    
    logger.info("\nFinal Results:")
    for model_name, metrics in results.items():
        logger.info(f"  {model_name:15s}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
    
    logger.info("\n✅ These are realistic accuracies without data leakage!")
    logger.info("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models without data leakage")
    
    parser.add_argument('--data_path', type=str, default=str(Config.PROCESSED_DATA_DIR / "labeled_data.csv"))
    parser.add_argument('--test_size', type=float, default=0.15)
    parser.add_argument('--val_size', type=float, default=0.15)
    
    parser.add_argument('--augment_data', action='store_true')
    parser.add_argument('--augmentation_ratio', type=float, default=0.5)
    
    parser.add_argument('--train_baseline', action='store_true')
    parser.add_argument('--train_lstm', action='store_true')
    parser.add_argument('--train_lightgbm', action='store_true')
    parser.add_argument('--train_deberta', action='store_true')
    
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs_lstm', type=int, default=10)
    parser.add_argument('--num_epochs_deberta', type=int, default=5)
    
    args = parser.parse_args()
    
    # If no model specified, train all
    if not any([args.train_baseline, args.train_lstm, args.train_lightgbm, args.train_deberta]):
        args.train_baseline = True
        args.train_lstm = True
        args.train_lightgbm = True
    
    main(args)