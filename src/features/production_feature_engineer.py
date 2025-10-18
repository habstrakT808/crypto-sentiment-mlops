# File: src/features/production_feature_engineer.py
"""
Production Feature Engineering - ZERO Data Leakage
Removes ALL sentiment-related features that could leak from auto-labeling
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import re
from textstat import flesch_reading_ease, flesch_kincaid_grade, syllable_count

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ProductionFeatureEngineer:
    """Production-ready feature engineering with ZERO leakage"""
    
    def __init__(self):
        """Initialize feature engineer"""
        # Financial keywords (domain-specific, NOT sentiment)
        self.bullish_keywords = [
            'moon', 'bullish', 'pump', 'rally', 'surge', 'breakout', 'gain',
            'profit', 'buy', 'long', 'hodl', 'accumulate', 'uptrend', 'bull',
            'green', 'lambo', 'rocket', 'winning', 'success'
        ]
        
        self.bearish_keywords = [
            'crash', 'dump', 'bearish', 'drop', 'fall', 'loss', 'sell',
            'short', 'downtrend', 'bear', 'decline', 'plummet', 'collapse',
            'red', 'rekt', 'scam', 'rug', 'fail'
        ]
        
        self.uncertainty_keywords = [
            'maybe', 'perhaps', 'possibly', 'might', 'could', 'uncertain',
            'doubt', 'question', 'unclear', 'confused', 'unsure', 'idk',
            'not sure', 'dunno'
        ]
        
        logger.info("ProductionFeatureEngineer initialized (ZERO leakage)")
    
    def create_text_statistics(self, df: pd.DataFrame, text_col: str = 'preprocessed_text') -> pd.DataFrame:
        """Create basic text statistics (NO SENTIMENT)"""
        logger.info("Creating text statistics...")
        
        # Length features
        df['text_length'] = df[text_col].str.len()
        df['word_count'] = df[text_col].str.split().str.len()
        
        # Character-level features
        df['uppercase_count'] = df[text_col].str.count(r'[A-Z]')
        df['lowercase_count'] = df[text_col].str.count(r'[a-z]')
        df['digit_count'] = df[text_col].str.count(r'\d')
        df['space_count'] = df[text_col].str.count(r'\s')
        
        # Punctuation features
        df['exclamation_count'] = df[text_col].str.count('!')
        df['question_count'] = df[text_col].str.count(r'\?')
        df['comma_count'] = df[text_col].str.count(',')
        df['period_count'] = df[text_col].str.count(r'\.')
        
        # Derived features
        df['uppercase_ratio'] = df['uppercase_count'] / (df['text_length'] + 1)
        df['digit_ratio'] = df['digit_count'] / (df['text_length'] + 1)
        df['punctuation_ratio'] = (df['exclamation_count'] + df['question_count']) / (df['text_length'] + 1)
        
        logger.info("Text statistics created")
        return df
    
    def create_linguistic_features(self, df: pd.DataFrame, text_col: str = 'preprocessed_text') -> pd.DataFrame:
        """Create linguistic complexity features (NO SENTIMENT)"""
        logger.info("Creating linguistic features...")
        
        def safe_textstat(text, func):
            try:
                if pd.isna(text) or text == '' or len(text.split()) < 3:
                    return 0
                return func(text)
            except:
                return 0
        
        # Readability scores
        df['flesch_reading_ease'] = df[text_col].apply(lambda x: safe_textstat(x, flesch_reading_ease))
        df['flesch_kincaid_grade'] = df[text_col].apply(lambda x: safe_textstat(x, flesch_kincaid_grade))
        df['syllable_count'] = df[text_col].apply(lambda x: safe_textstat(x, syllable_count))
        
        # Lexical diversity
        def lexical_diversity(text):
            if pd.isna(text) or text == '':
                return 0
            words = text.lower().split()
            if len(words) == 0:
                return 0
            return len(set(words)) / len(words)
        
        df['lexical_diversity'] = df[text_col].apply(lexical_diversity)
        
        # Average word length
        def avg_word_length(text):
            if pd.isna(text) or text == '':
                return 0
            words = text.split()
            if len(words) == 0:
                return 0
            return sum(len(word) for word in words) / len(words)
        
        df['avg_word_length'] = df[text_col].apply(avg_word_length)
        
        # Sentence-level features
        df['sentence_count'] = df[text_col].str.count(r'[.!?]+')
        df['avg_sentence_length'] = df['word_count'] / (df['sentence_count'] + 1)
        
        logger.info("Linguistic features created")
        return df
    
    def create_keyword_features(self, df: pd.DataFrame, text_col: str = 'preprocessed_text') -> pd.DataFrame:
        """Create keyword-based features (COUNTS ONLY, NO SENTIMENT INFERENCE)"""
        logger.info("Creating keyword features...")
        
        text_lower = df[text_col].str.lower()
        
        # Bullish keywords COUNT (not sentiment score!)
        df['bullish_keyword_count'] = text_lower.apply(
            lambda x: sum(1 for word in self.bullish_keywords if word in str(x))
        )
        
        # Bearish keywords COUNT
        df['bearish_keyword_count'] = text_lower.apply(
            lambda x: sum(1 for word in self.bearish_keywords if word in str(x))
        )
        
        # Uncertainty keywords COUNT
        df['uncertainty_keyword_count'] = text_lower.apply(
            lambda x: sum(1 for word in self.uncertainty_keywords if word in str(x))
        )
        
        # Keyword ratios (relative counts, NOT sentiment)
        df['bullish_bearish_ratio'] = df['bullish_keyword_count'] / (df['bearish_keyword_count'] + 1)
        df['total_keyword_count'] = df['bullish_keyword_count'] + df['bearish_keyword_count'] + df['uncertainty_keyword_count']
        df['keyword_density'] = df['total_keyword_count'] / (df['word_count'] + 1)
        
        logger.info("Keyword features created")
        return df
    
    def create_crypto_features(self, df: pd.DataFrame, text_col: str = 'preprocessed_text') -> pd.DataFrame:
        """Create cryptocurrency-specific features"""
        logger.info("Creating crypto-specific features...")
        
        text_lower = df[text_col].str.lower()
        
        # Major crypto mentions
        crypto_patterns = {
            'btc': r'\b(bitcoin|btc)\b',
            'eth': r'\b(ethereum|eth)\b',
            'ada': r'\b(cardano|ada)\b',
            'sol': r'\b(solana|sol)\b',
            'doge': r'\b(dogecoin|doge)\b',
        }
        
        for crypto, pattern in crypto_patterns.items():
            df[f'mentions_{crypto}'] = text_lower.str.contains(pattern, na=False, regex=True).astype(int)
        
        df['total_crypto_mentions'] = sum(df[f'mentions_{crypto}'] for crypto in crypto_patterns.keys())
        
        # Price patterns
        df['has_price_mention'] = text_lower.str.contains(r'\$\d+', na=False, regex=True).astype(int)
        df['price_mention_count'] = text_lower.str.count(r'\$\d+')
        df['has_percentage'] = text_lower.str.contains(r'\d+%', na=False, regex=True).astype(int)
        
        logger.info("Crypto-specific features created")
        return df
    
    def create_reddit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create Reddit-specific features"""
        logger.info("Creating Reddit-specific features...")
        
        # Engagement features
        df['score_log'] = np.log1p(df['score'].fillna(0))
        df['num_comments_log'] = np.log1p(df['num_comments'].fillna(0))
        df['upvote_ratio_norm'] = df['upvote_ratio'].fillna(0.5)
        
        # Engagement ratios
        df['engagement_ratio'] = df['num_comments'] / (df['score'] + 1)
        df['controversy_score'] = np.abs(df['upvote_ratio'] - 0.5) * 2
        
        logger.info("Reddit-specific features created")
        return df
    
    def create_temporal_features(self, df: pd.DataFrame, datetime_col: str = 'created_utc') -> pd.DataFrame:
        """Create time-based features"""
        logger.info("Creating temporal features...")
        
        if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
            try:
                df[datetime_col] = pd.to_datetime(df[datetime_col], unit='s')
            except:
                logger.warning(f"Could not parse {datetime_col}, skipping temporal features")
                return df
        
        # Extract time components
        df['hour'] = df[datetime_col].dt.hour
        df['day_of_week'] = df[datetime_col].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        logger.info("Temporal features created")
        return df
    
    def create_all_production_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ALL production features (ZERO LEAKAGE)"""
        logger.info("Creating all production features (ZERO leakage)...")
        
        original_cols = len(df.columns)
        
        # Create all feature types (NO SENTIMENT FEATURES!)
        df = self.create_text_statistics(df)
        df = self.create_linguistic_features(df)
        df = self.create_keyword_features(df)
        df = self.create_crypto_features(df)
        df = self.create_reddit_features(df)
        
        if 'created_utc' in df.columns:
            df = self.create_temporal_features(df)
        
        new_cols = len(df.columns) - original_cols
        logger.info(f"Created {new_cols} production features (total: {len(df.columns)})")
        
        return df
    
    def get_feature_list(self, df: pd.DataFrame) -> List[str]:
        """Get list of all production features - MATCH TRAINING FEATURES"""
        # CRITICAL: Use EXACT same features as training model
        trained_features = [
            'score', 'upvote_ratio', 'num_comments', 'text_length', 'word_count',
            'uppercase_count', 'lowercase_count', 'digit_count', 'space_count',
            'exclamation_count', 'question_count', 'comma_count', 'period_count',
            'uppercase_ratio', 'digit_ratio', 'punctuation_ratio', 'flesch_reading_ease',
            'flesch_kincaid_grade', 'syllable_count', 'lexical_diversity', 'avg_word_length',
            'sentence_count', 'avg_sentence_length', 'bullish_keyword_count', 'bearish_keyword_count',
            'uncertainty_keyword_count', 'bullish_bearish_ratio', 'total_keyword_count',
            'keyword_density', 'mentions_btc', 'mentions_eth', 'mentions_ada', 'mentions_sol',
            'mentions_doge', 'total_crypto_mentions', 'has_price_mention', 'price_mention_count',
            'has_percentage', 'score_log', 'num_comments_log', 'upvote_ratio_norm',
            'engagement_ratio', 'controversy_score'
        ]
        
        # Only return features that exist in dataframe and match training features
        available_features = [col for col in trained_features if col in df.columns]
        
        logger.info(f"Selected {len(available_features)} production features (MATCHED with training)")
        
        return available_features