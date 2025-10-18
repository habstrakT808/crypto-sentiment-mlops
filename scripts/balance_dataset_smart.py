# File: scripts/balance_dataset_smart.py
"""
🎯 Smart Dataset Balancing
Balance dataset using smart augmentation + manual labeling hints
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime

from src.data.production_augmentation import ProductionAugmenter
from src.utils.logger import setup_logger
from src.utils.config import Config

logger = setup_logger(__name__)

def heuristic_label(text: str, title: str = "") -> int:
    """
    Heuristic labeling based on keywords
    0 = negative, 1 = neutral, 2 = positive
    """
    combined = (title + " " + text).lower()
    
    # Strong positive keywords
    positive_keywords = [
        'moon', 'bullish', 'pump', 'rally', 'surge', 'breakout', 'gain',
        'profit', 'buy', 'long', 'hodl', 'great', 'amazing', 'excellent',
        'love', 'best', 'awesome', 'fantastic', 'incredible'
    ]
    
    # Strong negative keywords
    negative_keywords = [
        'crash', 'dump', 'bearish', 'drop', 'fall', 'loss', 'sell', 'short',
        'scam', 'rug', 'fail', 'worst', 'terrible', 'horrible', 'disaster',
        'bad', 'decline', 'plummet', 'collapse'
    ]
    
    # Count matches
    pos_count = sum(1 for word in positive_keywords if word in combined)
    neg_count = sum(1 for word in negative_keywords if word in combined)
    
    # Decision
    if pos_count > neg_count and pos_count > 0:
        return 2  # positive
    elif neg_count > pos_count and neg_count > 0:
        return 0  # negative
    else:
        return 1  # neutral

def main():
    """Balance dataset"""
    
    logger.info("="*80)
    logger.info("SMART DATASET BALANCING")
    logger.info("="*80)
    
    # Load raw data
    raw_files = list(Config.RAW_DATA_DIR.glob("reddit_balanced_*.csv"))
    if not raw_files:
        logger.error("❌ No balanced data found! Run collect_balanced_data.py first")
        return
    
    latest_file = max(raw_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading data from {latest_file}...")
    
    df = pd.read_csv(latest_file)
    logger.info(f"Loaded {len(df)} samples")
    
    # Apply heuristic labeling
    logger.info("\n🏷️ Applying heuristic labeling...")
    df['heuristic_label'] = df.apply(
        lambda row: heuristic_label(row['content'], row['title']),
        axis=1
    )
    
    # Filter by confidence (keep only clear cases)
    df['full_text'] = df['title'].fillna('') + ' ' + df['content'].fillna('')
    df['text_length'] = df['full_text'].str.len()
    
    # Keep samples with:
    # 1. Sufficient length (> 50 chars)
    # 2. Clear sentiment indicators
    df_filtered = df[df['text_length'] > 50].copy()
    
    logger.info(f"After filtering: {len(df_filtered)} samples")
    logger.info(f"Label distribution:\n{df_filtered['heuristic_label'].value_counts()}")
    
    # Balance using augmentation
    logger.info("\n🔄 Balancing with augmentation...")
    augmenter = ProductionAugmenter()
    
    df_balanced = augmenter.augment_minority_classes(
        df_filtered,
        text_column='full_text',
        label_column='heuristic_label',
        max_augment_per_sample=2,
        methods=['synonym']
    )
    
    logger.info(f"After augmentation: {len(df_balanced)} samples")
    logger.info(f"New distribution:\n{df_balanced['heuristic_label'].value_counts()}")
    
    # Rename for consistency
    df_balanced = df_balanced.rename(columns={'heuristic_label': 'auto_label_id'})
    df_balanced['auto_label'] = df_balanced['auto_label_id'].map({
        0: 'negative', 1: 'neutral', 2: 'positive'
    })
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Config.PROCESSED_DATA_DIR / f"labeled_balanced_{timestamp}.csv"
    df_balanced.to_csv(output_path, index=False)
    
    logger.info(f"\n✅ Saved balanced dataset to: {output_path}")
    logger.info("="*80)

if __name__ == "__main__":
    main()