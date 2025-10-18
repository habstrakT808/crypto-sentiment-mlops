# File: src/features/clean_feature_engineer.py
"""
Clean Feature Engineering - No Data Leakage
Only legitimate features that don't leak target information
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import re
from textstat import flesch_reading_ease, flesch_kincaid_grade, syllable_count
from textblob import TextBlob

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class CleanFeatureEngineer:
    """Feature engineering without data leakage"""
    
    def __init__(self):
        """Initialize feature engineer"""
        # Financial keywords (domain-specific)
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
        
        self.urgency_keywords = [
            'now', 'immediately', 'urgent', 'quick', 'fast', 'asap',
            'hurry', 'rush', 'soon', 'today', 'breaking', 'alert'
        ]
        
        logger.info("CleanFeatureEngineer initialized")
    
    def create_text_statistics(self, df: pd.DataFrame, text_col: str = 'full_text') -> pd.DataFrame:
        """Create basic text statistics"""
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
        
        # URL and mention features
        df['has_url'] = df[text_col].str.contains(r'http|www', na=False, regex=True).astype(int)
        df['url_count'] = df[text_col].str.count(r'http[s]?://\S+')
        df['mention_count'] = df[text_col].str.count(r'@\w+')
        df['hashtag_count'] = df[text_col].str.count(r'#\w+')
        
        # Derived features
        df['uppercase_ratio'] = df['uppercase_count'] / (df['text_length'] + 1)
        df['digit_ratio'] = df['digit_count'] / (df['text_length'] + 1)
        df['punctuation_ratio'] = (df['exclamation_count'] + df['question_count'] + df['comma_count']) / (df['text_length'] + 1)
        
        logger.info("Text statistics created")
        return df
    
    def create_linguistic_features(self, df: pd.DataFrame, text_col: str = 'full_text') -> pd.DataFrame:
        """Create linguistic complexity features"""
        logger.info("Creating linguistic features...")
        
        def safe_textstat(text, func):
            """Safely apply textstat function"""
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
    
    def create_keyword_features(self, df: pd.DataFrame, text_col: str = 'full_text') -> pd.DataFrame:
        """Create keyword-based features"""
        logger.info("Creating keyword features...")
        
        text_lower = df[text_col].str.lower()
        
        # Bullish keywords
        df['bullish_keyword_count'] = text_lower.apply(
            lambda x: sum(1 for word in self.bullish_keywords if word in str(x))
        )
        
        # Bearish keywords
        df['bearish_keyword_count'] = text_lower.apply(
            lambda x: sum(1 for word in self.bearish_keywords if word in str(x))
        )
        
        # Uncertainty keywords
        df['uncertainty_keyword_count'] = text_lower.apply(
            lambda x: sum(1 for word in self.uncertainty_keywords if word in str(x))
        )
        
        # Urgency keywords
        df['urgency_keyword_count'] = text_lower.apply(
            lambda x: sum(1 for word in self.urgency_keywords if word in str(x))
        )
        
        # Keyword ratios
        df['bullish_bearish_ratio'] = df['bullish_keyword_count'] / (df['bearish_keyword_count'] + 1)
        df['total_keyword_count'] = df['bullish_keyword_count'] + df['bearish_keyword_count'] + df['uncertainty_keyword_count'] + df['urgency_keyword_count']
        df['keyword_density'] = df['total_keyword_count'] / (df['word_count'] + 1)
        
        logger.info("Keyword features created")
        return df
    
    def create_crypto_features(self, df: pd.DataFrame, text_col: str = 'full_text') -> pd.DataFrame:
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
            'xrp': r'\b(ripple|xrp)\b',
            'bnb': r'\b(binance|bnb)\b'
        }
        
        for crypto, pattern in crypto_patterns.items():
            df[f'mentions_{crypto}'] = text_lower.str.contains(pattern, na=False, regex=True).astype(int)
        
        df['total_crypto_mentions'] = sum(df[f'mentions_{crypto}'] for crypto in crypto_patterns.keys())
        
        # Price patterns
        df['has_price_mention'] = text_lower.str.contains(r'\$\d+', na=False, regex=True).astype(int)
        df['price_mention_count'] = text_lower.str.count(r'\$\d+')
        df['has_percentage'] = text_lower.str.contains(r'\d+%', na=False, regex=True).astype(int)
        df['percentage_mention_count'] = text_lower.str.count(r'\d+%')
        
        # Trading-related terms
        trading_terms = ['buy', 'sell', 'trade', 'hold', 'hodl', 'dip', 'ath', 'fomo', 'fud']
        df['trading_term_count'] = text_lower.apply(
            lambda x: sum(1 for term in trading_terms if term in str(x))
        )
        
        logger.info("Crypto-specific features created")
        return df
    
    def create_emoji_features(self, df: pd.DataFrame, text_col: str = 'full_text') -> pd.DataFrame:
        """Create emoji-based features"""
        logger.info("Creating emoji features...")
        
        # Positive emojis
        positive_emojis = ['🚀', '🌙', '💎', '🙌', '💰', '📈', '🔥', '✨', '⭐', '💪', '👍', '😊', '😁', '🎉']
        
        # Negative emojis
        negative_emojis = ['📉', '💀', '😢', '😭', '🤡', '⚠️', '❌', '🔻', '👎', '😞', '😔']
        
        df['positive_emoji_count'] = df[text_col].apply(
            lambda x: sum(1 for emoji in positive_emojis if emoji in str(x))
        )
        
        df['negative_emoji_count'] = df[text_col].apply(
            lambda x: sum(1 for emoji in negative_emojis if emoji in str(x))
        )
        
        df['total_emoji_count'] = df['positive_emoji_count'] + df['negative_emoji_count']
        df['emoji_sentiment_score'] = df['positive_emoji_count'] - df['negative_emoji_count']
        df['has_emoji'] = (df['total_emoji_count'] > 0).astype(int)
        
        logger.info("Emoji features created")
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
        df['controversy_score'] = np.abs(df['upvote_ratio'] - 0.5) * 2  # 0 = controversial, 1 = unanimous
        
        # Viral potential
        df['viral_potential'] = (df['num_comments'] > df['score'] * 0.5).astype(int)
        
        # Post type
        df['is_self_post'] = df['is_self'].astype(int) if 'is_self' in df.columns else 0
        
        logger.info("Reddit-specific features created")
        return df
    
    def create_temporal_features(self, df: pd.DataFrame, datetime_col: str = 'created_utc') -> pd.DataFrame:
        """Create time-based features"""
        logger.info("Creating temporal features...")
        
        # Ensure datetime type - handle different formats
        if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
            try:
                # Try parsing as Unix timestamp first
                df[datetime_col] = pd.to_datetime(df[datetime_col], unit='s')
            except (ValueError, TypeError):
                try:
                    # Try parsing as datetime string
                    df[datetime_col] = pd.to_datetime(df[datetime_col])
                except (ValueError, TypeError):
                    # If both fail, skip temporal features
                    logger.warning(f"Could not parse {datetime_col} as datetime, skipping temporal features")
                    return df
        
        # Extract time components
        df['hour'] = df[datetime_col].dt.hour
        df['day_of_week'] = df[datetime_col].dt.dayofweek
        df['day_of_month'] = df[datetime_col].dt.day
        df['month'] = df[datetime_col].dt.month
        
        # Time categories
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        logger.info("Temporal features created")
        return df
    
    def create_textblob_features(self, df: pd.DataFrame, text_col: str = 'full_text') -> pd.DataFrame:
        """Create TextBlob-based features (NOT used in auto-labeling)"""
        logger.info("Creating TextBlob features...")
        
        def get_textblob_features(text):
            """Extract TextBlob features"""
            if pd.isna(text) or text == '':
                return {'polarity': 0.0, 'subjectivity': 0.0}
            
            try:
                blob = TextBlob(str(text))
                return {
                    'polarity': blob.sentiment.polarity,
                    'subjectivity': blob.sentiment.subjectivity
                }
            except:
                return {'polarity': 0.0, 'subjectivity': 0.0}
        
        sentiment_features = df[text_col].apply(get_textblob_features)
        df['textblob_polarity'] = sentiment_features.apply(lambda x: x['polarity'])
        df['textblob_subjectivity'] = sentiment_features.apply(lambda x: x['subjectivity'])
        df['textblob_objectivity'] = 1 - df['textblob_subjectivity']
        
        # Sentiment strength
        df['sentiment_strength'] = np.abs(df['textblob_polarity'])
        
        # Strong opinion indicator
        df['has_strong_opinion'] = ((df['sentiment_strength'] > 0.5) & (df['textblob_subjectivity'] > 0.5)).astype(int)
        
        logger.info("TextBlob features created")
        return df
    
    def create_all_clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all clean features (no leakage)"""
        logger.info("Creating all clean features...")
        
        original_cols = len(df.columns)
        
        # Create all feature types
        df = self.create_text_statistics(df)
        df = self.create_linguistic_features(df)
        df = self.create_keyword_features(df)
        df = self.create_crypto_features(df)
        df = self.create_emoji_features(df)
        df = self.create_reddit_features(df)
        
        if 'created_utc' in df.columns:
            df = self.create_temporal_features(df)
        
        # Add TextBlob features (safe to use as they're different from auto-labeler)
        df = self.create_textblob_features(df)
        
        new_cols = len(df.columns) - original_cols
        logger.info(f"Created {new_cols} clean features (total: {len(df.columns)})")
        
        return df
    
    def get_feature_list(self, df: pd.DataFrame) -> List[str]:
        """Get list of all clean features"""
        # Exclude non-feature columns
        exclude_cols = [
            'post_id', 'title', 'content', 'author', 'subreddit', 'url',
            'permalink', 'link_flair_text', 'collected_at', 'full_text',
            'preprocessed_text', 'created_utc', 'is_self',
            # Exclude any label-related columns
            'auto_label', 'auto_label_id', 'label_confidence', 'label_agreement',
            'score_negative', 'score_neutral', 'score_positive',
            'textblob_prediction', 'vader_prediction', 'finbert_prediction',
            'vader_compound', 'vader_pos', 'vader_neg', 'vader_neu'
        ]
        
        features = [col for col in df.columns if col not in exclude_cols]
        
        # Only numeric features
        numeric_features = df[features].select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"Selected {len(numeric_features)} clean numeric features")
        
        return numeric_features