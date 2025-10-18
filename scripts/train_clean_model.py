#!/usr/bin/env python3
"""
Train models with clean data (no data leakage)
This script trains models using only legitimate features without leakage
"""

import pandas as pd
import numpy as np
import argparse
from datetime import datetime
import warnings
from sklearn.metrics import accuracy_score, f1_score
warnings.filterwarnings('ignore')

# Add src to path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.logger import setup_logger
from src.models.lightgbm_model import LightGBMModel
from src.models.model_trainer import ModelTrainer
from src.explainability.shap_explainer import SHAPExplainer

logger = setup_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train models with clean data (no leakage)')
    parser.add_argument('--data_path', type=str, 
                       default='data/processed/clean_data.csv',
                       help='Path to clean data CSV file')
    parser.add_argument('--text_column', type=str, default='content',
                       help='Text column to use for analysis')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.2,
                       help='Validation set size')
    parser.add_argument('--explain_model', action='store_true',
                       help='Generate SHAP explanations')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("TRAINING WITH CLEAN DATA (NO DATA LEAKAGE)")
    logger.info("="*80)
    
    # ========== STEP 1: LOAD CLEAN DATA ==========
    logger.info("\n" + "="*80)
    logger.info("STEP 1: LOADING CLEAN DATA")
    logger.info("="*80)
    
    df = pd.read_csv(args.data_path)
    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"Features: {len(df.columns)}")
    logger.info("Available features:")
    for i, col in enumerate(df.columns, 1):
        logger.info(f"  {i:2d}. {col}")
    
    # ========== STEP 2: CREATE LABELS FROM TEXT ==========
    logger.info("\n" + "="*80)
    logger.info("STEP 2: CREATING LABELS FROM TEXT ANALYSIS")
    logger.info("="*80)
    
    # Simple sentiment analysis using TextBlob (different from auto-labeler)
    from textblob import TextBlob
    
    def simple_sentiment(text):
        """Simple sentiment analysis using TextBlob"""
        if pd.isna(text) or text == '':
            return 1  # neutral
        
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            return 2  # positive
        elif polarity < -0.1:
            return 0  # negative
        else:
            return 1  # neutral
    
    logger.info("Creating labels using simple TextBlob sentiment...")
    df['simple_label'] = df[args.text_column].apply(simple_sentiment)
    
    logger.info(f"Label distribution:")
    logger.info(df['simple_label'].value_counts().sort_index())
    
    # ========== STEP 3: FEATURE ENGINEERING ==========
    logger.info("\n" + "="*80)
    logger.info("STEP 3: BASIC FEATURE ENGINEERING")
    logger.info("="*80)
    
    # Create basic features from text
    df['text_length'] = df[args.text_column].str.len()
    df['word_count'] = df[args.text_column].str.split().str.len()
    df['has_url'] = df[args.text_column].str.contains('http', na=False).astype(int)
    df['has_emoji'] = df[args.text_column].str.contains('[😀-🙏]', na=False).astype(int)
    df['exclamation_count'] = df[args.text_column].str.count('!')
    df['question_count'] = df[args.text_column].str.count('\\?')
    
    # Reddit-specific features
    df['score_log'] = np.log1p(df['score'].fillna(0))
    df['upvote_ratio_norm'] = df['upvote_ratio'].fillna(0.5)
    df['num_comments_log'] = np.log1p(df['num_comments'].fillna(0))
    
    logger.info(f"Created {len(df.columns) - len(pd.read_csv(args.data_path).columns)} new features")
    
    # ========== STEP 4: PREPARE DATA ==========
    logger.info("\n" + "="*80)
    logger.info("STEP 4: PREPARING DATA FOR TRAINING")
    logger.info("="*80)
    
    # Select numeric features only
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [col for col in numeric_features if col != 'simple_label']
    
    logger.info(f"Using {len(numeric_features)} numeric features:")
    for i, col in enumerate(numeric_features, 1):
        logger.info(f"  {i:2d}. {col}")
    
    X = df[numeric_features]
    y = df['simple_label'].values
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    
    # ========== STEP 5: TRAIN LIGHTGBM ==========
    logger.info("\n" + "="*80)
    logger.info("STEP 5: TRAINING LIGHTGBM MODEL")
    logger.info("="*80)
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # Train LightGBM
    logger.info("Training LightGBM model...")
    lightgbm_model = LightGBMModel()
    lightgbm_model.train(X_train, y_train)
    
    # Evaluate
    y_pred = lightgbm_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    logger.info(f"Test Accuracy: {acc:.4f}")
    logger.info(f"Test F1 Score: {f1:.4f}")
    
    # Show feature importance
    logger.info("\nTop 10 Feature Importance:")
    feature_importance = lightgbm_model.model.feature_importance(importance_type='gain')
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
        logger.info(f"  {i:2d}. {row['feature']:<20} {row['importance']:8.2f}")
    
    # ========== STEP 6: SHAP EXPLANATIONS ==========
    if args.explain_model:
        logger.info("\n" + "="*80)
        logger.info("STEP 6: GENERATING SHAP EXPLANATIONS")
        logger.info("="*80)
        
        logger.info("Creating SHAP explainer...")
        explainer = SHAPExplainer(lightgbm_model.model)
        explainer.fit(X_train)
        
        logger.info("Calculating SHAP values...")
        explainer.explain(X_test)
        
        logger.info("Generating SHAP plots...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Summary plot
        explainer.plot_summary(X_test)
        logger.info(f"SHAP summary plot saved")
        
        # Feature importance plot
        explainer.plot_feature_importance(X_test)
        logger.info(f"SHAP importance plot saved")
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"Final Test Accuracy: {acc:.4f}")
    logger.info(f"Final Test F1 Score: {f1:.4f}")
    logger.info("\nThis is a more realistic accuracy without data leakage!")

if __name__ == "__main__":
    main()
