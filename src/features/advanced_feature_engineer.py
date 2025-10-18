# File: src/features/advanced_feature_engineer.py
"""
Advanced Feature Engineering
Financial domain-specific and linguistic features
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import re
from textstat import flesch_reading_ease, flesch_kincaid_grade
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class AdvancedFeatureEngineer:
    """Advanced feature engineering for financial sentiment"""
    
    def __init__(self):
        """Initialize feature engineer"""
        self.vader = SentimentIntensityAnalyzer()
        
        # Financial keywords
        self.bullish_keywords = [
            'moon', 'bullish', 'pump', 'rally', 'surge', 'breakout', 'gain',
            'profit', 'buy', 'long', 'hodl', 'accumulate', 'uptrend', 'bull'
        ]
        
        self.bearish_keywords = [
            'crash', 'dump', 'bearish', 'drop', 'fall', 'loss', 'sell',
            'short', 'downtrend', 'bear', 'decline', 'plummet', 'collapse'
        ]
        
        self.uncertainty_keywords = [
            'maybe', 'perhaps', 'possibly', 'might', 'could', 'uncertain',
            'doubt', 'question', 'unclear', 'confused', 'unsure'
        ]
        
        self.urgency_keywords = [
            'now', 'immediately', 'urgent', 'quick', 'fast', 'asap',
            'hurry', 'rush', 'soon', 'today'
        ]
    
    def create_financial_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create financial-specific sentiment features"""
        logger.info("Creating financial sentiment features...")
        
        text_col = 'preprocessed_text'
        
        # Bullish/Bearish keyword counts
        df['bullish_keyword_count'] = df[text_col].apply(
            lambda x: sum(1 for word in self.bullish_keywords if word in x.lower())
        )
        
        df['bearish_keyword_count'] = df[text_col].apply(
            lambda x: sum(1 for word in self.bearish_keywords if word in x.lower())
        )
        
        # Sentiment ratio
        df['bullish_bearish_ratio'] = df['bullish_keyword_count'] / (df['bearish_keyword_count'] + 1)
        
        # Uncertainty score
        df['uncertainty_score'] = df[text_col].apply(
            lambda x: sum(1 for word in self.uncertainty_keywords if word in x.lower())
        )
        
        # Urgency score
        df['urgency_score'] = df[text_col].apply(
            lambda x: sum(1 for word in self.urgency_keywords if word in x.lower())
        )
        
        # VADER sentiment (more detailed)
        vader_scores = df['full_text'].apply(lambda x: self.vader.polarity_scores(x))
        df['vader_compound'] = vader_scores.apply(lambda x: x['compound'])
        df['vader_pos'] = vader_scores.apply(lambda x: x['pos'])
        df['vader_neg'] = vader_scores.apply(lambda x: x['neg'])
        df['vader_neu'] = vader_scores.apply(lambda x: x['neu'])
        
        logger.info("Financial sentiment features created")
        return df
    
    def create_readability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create readability features"""
        logger.info("Creating readability features...")
        
        # Flesch Reading Ease
        df['flesch_reading_ease'] = df['full_text'].apply(
            lambda x: flesch_reading_ease(x) if len(x.split()) > 5 else 0
        )
        
        # Flesch-Kincaid Grade
        df['flesch_kincaid_grade'] = df['full_text'].apply(
            lambda x: flesch_kincaid_grade(x) if len(x.split()) > 5 else 0
        )
        
        logger.info("Readability features created")
        return df
    
    def create_price_mention_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect price mentions and targets"""
        logger.info("Creating price mention features...")
        
        # Price patterns
        price_pattern = r'\$\d+[,\d]*\.?\d*[kKmMbB]?'
        percentage_pattern = r'\d+\.?\d*%'
        
        df['has_price_mention'] = df['full_text'].apply(
            lambda x: 1 if re.search(price_pattern, x) else 0
        )
        
        df['has_percentage_mention'] = df['full_text'].apply(
            lambda x: 1 if re.search(percentage_pattern, x) else 0
        )
        
        # Count price mentions
        df['price_mention_count'] = df['full_text'].apply(
            lambda x: len(re.findall(price_pattern, x))
        )
        
        logger.info("Price mention features created")
        return df
    
    def create_emoji_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze emoji usage"""
        logger.info("Creating emoji features...")
        
        # Positive emojis
        positive_emojis = ['🚀', '🌙', '💎', '🙌', '💰', '📈', '🔥', '✨', '⭐', '💪']
        
        # Negative emojis
        negative_emojis = ['📉', '💀', '😢', '😭', '🤡', '⚠️', '❌', '🔻']
        
        df['positive_emoji_count'] = df['full_text'].apply(
            lambda x: sum(1 for emoji in positive_emojis if emoji in x)
        )
        
        df['negative_emoji_count'] = df['full_text'].apply(
            lambda x: sum(1 for emoji in negative_emojis if emoji in x)
        )
        
        df['emoji_sentiment'] = df['positive_emoji_count'] - df['negative_emoji_count']
        
        logger.info("Emoji features created")
        return df
    
    def create_crypto_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cryptocurrency-specific features"""
        logger.info("Creating crypto-specific features...")
        
        # Major crypto mentions
        crypto_patterns = {
            'btc': r'\b(bitcoin|btc)\b',
            'eth': r'\b(ethereum|eth)\b',
            'ada': r'\b(cardano|ada)\b',
            'sol': r'\b(solana|sol)\b',
            'doge': r'\b(dogecoin|doge)\b'
        }
        
        for crypto, pattern in crypto_patterns.items():
            df[f'mentions_{crypto}'] = df['full_text'].apply(
                lambda x: 1 if re.search(pattern, x.lower()) else 0
            )
        
        # Total crypto mentions
        df['total_crypto_mentions'] = sum(df[f'mentions_{crypto}'] for crypto in crypto_patterns.keys())
        
        # Hashtag count
        df['hashtag_count'] = df['full_text'].apply(
            lambda x: len(re.findall(r'#\w+', x))
        )
        
        logger.info("Crypto-specific features created")
        return df
    
    def create_all_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all advanced features"""
        logger.info("Creating all advanced features...")
        
        df = self.create_financial_sentiment_features(df)
        df = self.create_readability_features(df)
        df = self.create_price_mention_features(df)
        df = self.create_emoji_features(df)
        df = self.create_crypto_specific_features(df)
        
        logger.info(f"Advanced feature engineering complete! Total features: {len(df.columns)}")
        
        return df