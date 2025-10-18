"""
Feature Engineering Module
Advanced feature extraction for ML models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import warnings
warnings.filterwarnings('ignore')

from src.utils.logger import setup_logger
from src.utils.config import Config

logger = setup_logger(__name__)

class FeatureEngineer:
    """Engineer features for ML models"""
    
    def __init__(self):
        """Initialize feature engineer"""
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.lda_model = None
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features
        
        Args:
            df: Input dataframe with 'created_utc' column
            
        Returns:
            Dataframe with temporal features
        """
        logger.info("Creating temporal features...")
        
        # Ensure datetime type
        if not pd.api.types.is_datetime64_any_dtype(df['created_utc']):
            df['created_utc'] = pd.to_datetime(df['created_utc'])
        
        # Extract time components
        df['hour'] = df['created_utc'].dt.hour
        df['day_of_week'] = df['created_utc'].dt.dayofweek
        df['day_of_month'] = df['created_utc'].dt.day
        df['month'] = df['created_utc'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Time of day categories
        df['time_of_day'] = pd.cut(
            df['hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening'],
            include_lowest=True
        )
        
        # Business hours
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        logger.info("Temporal features created")
        return df
    
    def create_engagement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engagement-based features
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with engagement features
        """
        logger.info("Creating engagement features...")
        
        # Ensure numeric types
        df['score'] = pd.to_numeric(df['score'], errors='coerce').fillna(0)
        df['num_comments'] = pd.to_numeric(df['num_comments'], errors='coerce').fillna(0)
        
        # Basic engagement metrics
        df['engagement_ratio'] = df['num_comments'] / (df['score'] + 1)  # +1 to avoid division by zero
        df['controversy_score'] = df['num_comments'] - df['score']
        
        # Log transformations for skewed distributions
        df['log_score'] = np.log1p(df['score'])
        df['log_comments'] = np.log1p(df['num_comments'])
        
        # Engagement categories
        df['engagement_level'] = pd.cut(
            df['score'],
            bins=[-np.inf, 0, 10, 100, np.inf],
            labels=['negative', 'low', 'medium', 'high']
        )
        
        # Viral potential (high comments relative to score)
        df['viral_potential'] = (df['num_comments'] > df['score'] * 0.5).astype(int)
        
        logger.info("Engagement features created")
        return df
    
    def create_text_complexity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create text complexity features
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with text complexity features
        """
        logger.info("Creating text complexity features...")
        
        # Lexical diversity
        df['lexical_diversity'] = df['unique_word_count'] / (df['word_count'] + 1)
        
        # Average sentence length
        df['avg_sentence_length'] = df['word_count'] / (df['sentence_count'] + 1)
        
        # Text density (characters per word)
        df['text_density'] = df['char_count'] / (df['word_count'] + 1)
        
        # Readability approximation (Flesch Reading Ease simplified)
        df['readability_score'] = (
            206.835 - 
            1.015 * df['avg_sentence_length'] - 
            84.6 * df['avg_word_length']
        )
        
        # Complexity categories
        df['complexity_level'] = pd.cut(
            df['word_count'],
            bins=[0, 50, 150, 300, np.inf],
            labels=['very_short', 'short', 'medium', 'long']
        )
        
        logger.info("Text complexity features created")
        return df
    
    def create_sentiment_derivative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived sentiment features
        
        Args:
            df: Input dataframe with sentiment features
            
        Returns:
            Dataframe with sentiment derivative features
        """
        logger.info("Creating sentiment derivative features...")
        
        # Sentiment strength (absolute polarity)
        df['sentiment_strength'] = np.abs(df['sentiment_polarity'])
        
        # Sentiment categories
        df['sentiment_category'] = pd.cut(
            df['sentiment_polarity'],
            bins=[-1, -0.1, 0.1, 1],
            labels=['negative', 'neutral', 'positive']
        )
        
        # Objectivity (inverse of subjectivity)
        df['objectivity'] = 1 - df['sentiment_subjectivity']
        
        # Strong opinion indicator
        df['strong_opinion'] = (
            (df['sentiment_strength'] > 0.5) & 
            (df['sentiment_subjectivity'] > 0.5)
        ).astype(int)
        
        # Neutral but subjective (opinion without clear sentiment)
        df['neutral_subjective'] = (
            (df['sentiment_strength'] < 0.1) & 
            (df['sentiment_subjectivity'] > 0.5)
        ).astype(int)
        
        logger.info("Sentiment derivative features created")
        return df
    
    def create_subreddit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create subreddit-based features
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with subreddit features
        """
        logger.info("Creating subreddit features...")
        
        # Subreddit statistics
        subreddit_stats = df.groupby('subreddit').agg({
            'score': ['mean', 'std', 'median'],
            'num_comments': ['mean', 'std', 'median'],
            'sentiment_polarity': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        subreddit_stats.columns = [
            'subreddit',
            'subreddit_avg_score', 'subreddit_std_score', 'subreddit_median_score',
            'subreddit_avg_comments', 'subreddit_std_comments', 'subreddit_median_comments',
            'subreddit_avg_sentiment', 'subreddit_std_sentiment'
        ]
        
        # Merge back to original dataframe
        df = df.merge(subreddit_stats, on='subreddit', how='left')
        
        # Relative metrics (how this post compares to subreddit average)
        df['score_vs_subreddit_avg'] = df['score'] / (df['subreddit_avg_score'] + 1)
        df['sentiment_vs_subreddit_avg'] = df['sentiment_polarity'] - df['subreddit_avg_sentiment']
        
        logger.info("Subreddit features created")
        return df
    
    def create_tfidf_features(
        self, 
        df: pd.DataFrame, 
        text_column: str = 'preprocessed_text',
        max_features: int = 100
    ) -> Tuple[pd.DataFrame, TfidfVectorizer]:
        """
        Create TF-IDF features
        
        Args:
            df: Input dataframe
            text_column: Column containing preprocessed text
            max_features: Maximum number of features to extract
            
        Returns:
            Tuple of (dataframe with TF-IDF features, fitted vectorizer)
        """
        logger.info(f"Creating TF-IDF features (max_features={max_features})...")
        
        # Initialize or use existing vectorizer
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),  # unigrams and bigrams
                min_df=2,  # minimum document frequency
                max_df=0.95  # maximum document frequency
            )
            
            # Fit and transform
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(df[text_column])
        else:
            # Transform only
            tfidf_matrix = self.tfidf_vectorizer.transform(df[text_column])
        
        # Convert to dataframe
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
        )
        
        # Concatenate with original dataframe
        df = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)
        
        logger.info(f"TF-IDF features created: {tfidf_matrix.shape[1]} features")
        return df, self.tfidf_vectorizer
    
    def create_topic_features(
        self,
        df: pd.DataFrame,
        text_column: str = 'preprocessed_text',
        n_topics: int = 10
    ) -> pd.DataFrame:
        """
        Create topic modeling features using LDA
        
        Args:
            df: Input dataframe
            text_column: Column containing preprocessed text
            n_topics: Number of topics to extract
            
        Returns:
            Dataframe with topic features
        """
        logger.info(f"Creating topic features (n_topics={n_topics})...")
        
        # Initialize count vectorizer for LDA
        if self.count_vectorizer is None:
            self.count_vectorizer = CountVectorizer(
                max_features=1000,
                min_df=2,
                max_df=0.95
            )
            count_matrix = self.count_vectorizer.fit_transform(df[text_column])
        else:
            count_matrix = self.count_vectorizer.transform(df[text_column])
        
        # Initialize and fit LDA
        if self.lda_model is None:
            self.lda_model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=10,
                n_jobs=-1
            )
            topic_distributions = self.lda_model.fit_transform(count_matrix)
        else:
            topic_distributions = self.lda_model.transform(count_matrix)
        
        # Create topic feature columns
        topic_df = pd.DataFrame(
            topic_distributions,
            columns=[f'topic_{i}' for i in range(n_topics)]
        )
        
        # Add dominant topic
        topic_df['dominant_topic'] = topic_distributions.argmax(axis=1)
        topic_df['dominant_topic_prob'] = topic_distributions.max(axis=1)
        
        # Concatenate with original dataframe
        df = pd.concat([df.reset_index(drop=True), topic_df], axis=1)
        
        logger.info(f"Topic features created: {n_topics} topics")
        return df
    
    def engineer_all_features(
        self,
        df: pd.DataFrame,
        include_tfidf: bool = False,
        include_topics: bool = False
    ) -> pd.DataFrame:
        """
        Create all features
        
        Args:
            df: Input dataframe
            include_tfidf: Whether to include TF-IDF features
            include_topics: Whether to include topic features
            
        Returns:
            Dataframe with all engineered features
        """
        logger.info("Starting comprehensive feature engineering...")
        
        # Create all feature types
        df = self.create_temporal_features(df)
        df = self.create_engagement_features(df)
        df = self.create_text_complexity_features(df)
        df = self.create_sentiment_derivative_features(df)
        df = self.create_subreddit_features(df)
        
        # Optional: TF-IDF features (can be memory intensive)
        if include_tfidf:
            df, _ = self.create_tfidf_features(df, max_features=50)
        
        # Optional: Topic features
        if include_topics:
            df = self.create_topic_features(df, n_topics=5)
        
        logger.info(f"Feature engineering complete! Total features: {len(df.columns)}")
        
        return df
    
    def get_feature_importance_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary of engineered features
        
        Args:
            df: Dataframe with features
            
        Returns:
            Summary dataframe
        """
        # Identify feature columns (exclude original data columns)
        feature_cols = [col for col in df.columns if any([
            col.startswith('hour'),
            col.startswith('day_'),
            col.startswith('is_'),
            col.startswith('engagement_'),
            col.startswith('log_'),
            col.startswith('lexical_'),
            col.startswith('avg_'),
            col.startswith('sentiment_'),
            col.startswith('subreddit_'),
            col.startswith('tfidf_'),
            col.startswith('topic_')
        ])]
        
        # Create summary
        summary = pd.DataFrame({
            'feature': feature_cols,
            'dtype': [df[col].dtype for col in feature_cols],
            'missing_pct': [(df[col].isna().sum() / len(df)) * 100 for col in feature_cols],
            'unique_values': [df[col].nunique() for col in feature_cols]
        })
        
        return summary.sort_values('feature')