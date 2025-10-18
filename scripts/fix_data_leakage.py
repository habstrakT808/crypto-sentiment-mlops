#!/usr/bin/env python3
"""
Fix Data Leakage in Crypto Sentiment Analysis
Remove features that cause data leakage from auto-labeling process
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def remove_leakage_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove features that cause data leakage
    
    Args:
        df: DataFrame with leakage features
        
    Returns:
        DataFrame without leakage features
    """
    logger.info("Removing data leakage features...")
    
    # Features that directly leak information about the target
    leakage_features = [
        'label_confidence',      # Confidence from auto-labeler
        'label_agreement',       # Agreement score from auto-labeler
        'auto_label',            # The actual label (if present)
        'auto_label_id',         # The actual label ID (if present)
        'textblob_prediction',   # Individual model predictions
        'vader_prediction',     # Individual model predictions  
        'finbert_prediction',    # Individual model predictions
        'score_negative',        # Scores from auto-labeler
        'score_neutral',         # Scores from auto-labeler
        'score_positive'         # Scores from auto-labeler
    ]
    
    # Find which leakage features exist in the dataframe
    existing_leakage = [col for col in leakage_features if col in df.columns]
    
    if existing_leakage:
        logger.warning(f"Found leakage features: {existing_leakage}")
        df_clean = df.drop(columns=existing_leakage)
        logger.info(f"Removed {len(existing_leakage)} leakage features")
        logger.info(f"Remaining features: {len(df_clean.columns)}")
    else:
        logger.info("No leakage features found")
        df_clean = df.copy()
    
    return df_clean

def main():
    parser = argparse.ArgumentParser(description='Fix data leakage in crypto sentiment analysis')
    parser.add_argument('--input_file', type=str, 
                       default='data/processed/processed_data.csv',
                       help='Input CSV file with leakage features')
    parser.add_argument('--output_file', type=str,
                       default='data/processed/clean_data.csv', 
                       help='Output CSV file without leakage features')
    parser.add_argument('--text_column', type=str, default='content',
                       help='Text column to use for analysis')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("FIXING DATA LEAKAGE IN CRYPTO SENTIMENT ANALYSIS")
    logger.info("="*80)
    
    # Check if input file exists
    if not Path(args.input_file).exists():
        logger.error(f"Input file not found: {args.input_file}")
        logger.info("Please run auto-labeling first:")
        logger.info("python scripts/auto_label_data.py --text_column content")
        return
    
    # Load data
    logger.info(f"Loading data from {args.input_file}...")
    df = pd.read_csv(args.input_file)
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
    
    # Show original feature list
    logger.info("Original features:")
    for i, col in enumerate(df.columns, 1):
        logger.info(f"  {i:2d}. {col}")
    
    # Remove leakage features
    df_clean = remove_leakage_features(df)
    
    # Show cleaned feature list
    logger.info("\nCleaned features:")
    for i, col in enumerate(df_clean.columns, 1):
        logger.info(f"  {i:2d}. {col}")
    
    # Save cleaned data
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving cleaned data to {args.output_file}...")
    df_clean.to_csv(args.output_file, index=False)
    
    logger.info("="*80)
    logger.info("DATA LEAKAGE FIX COMPLETE!")
    logger.info("="*80)
    logger.info(f"Original features: {len(df.columns)}")
    logger.info(f"Cleaned features: {len(df_clean.columns)}")
    logger.info(f"Removed features: {len(df.columns) - len(df_clean.columns)}")
    logger.info(f"Output saved to: {args.output_file}")
    
    logger.info("\nNext steps:")
    logger.info("1. Retrain models without leakage features")
    logger.info("2. Expect more realistic accuracy (60-80%)")
    logger.info("3. Use proper train/validation/test splits")

if __name__ == "__main__":
    main()
