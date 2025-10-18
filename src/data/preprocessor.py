"""
Text Preprocessing Module
Comprehensive text cleaning and preprocessing for NLP
"""

import re
import string
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
from textblob import TextBlob

from src.utils.logger import setup_logger
from src.utils.config import Config

logger = setup_logger(__name__)

class TextPreprocessor:
    """Advanced text preprocessing for sentiment analysis"""
    
    def __init__(self, remove_stopwords: bool = True, lemmatize: bool = True):
        """
        Initialize preprocessor
        
        Args:
            remove_stopwords: Whether to remove stop words
            lemmatize: Whether to lemmatize words
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        
        # Crypto-specific terms to preserve (DEFINE FIRST)
        self.crypto_terms = {
            'btc', 'bitcoin', 'eth', 'ethereum', 'crypto', 'cryptocurrency',
            'blockchain', 'defi', 'nft', 'altcoin', 'hodl', 'moon', 'lambo',
            'satoshi', 'wei', 'gwei', 'dapp', 'dao', 'ico', 'ido', 'airdrop'
        }
        
        # Sentiment-bearing words to preserve (DEFINE FIRST)
        self.sentiment_words = {
            'not', 'no', 'never', 'nothing', 'nowhere', 'neither', 'nobody',
            'none', 'very', 'too', 'so', 'really', 'quite', 'extremely'
        }
        
        # NOW initialize NLP tools
        self._initialize_nlp_tools()
    
    def _initialize_nlp_tools(self):
        """Initialize NLP libraries and download required data"""
        try:
            # Download required NLTK data
            nltk_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
            for data in nltk_data:
                try:
                    nltk.data.find(f'tokenizers/{data}')
                except LookupError:
                    logger.info(f"Downloading NLTK data: {data}")
                    nltk.download(data, quiet=True)
            
            # Initialize tools
            self.stop_words = set(stopwords.words('english'))
            # Remove sentiment-bearing words from stopwords
            self.stop_words = self.stop_words - self.sentiment_words
            
            self.lemmatizer = WordNetLemmatizer()
            
            # Load spaCy model
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                logger.warning("spaCy model not found. Downloading...")
                import subprocess
                subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
                try:
                    self.nlp = spacy.load('en_core_web_sm')
                except:
                    logger.warning("spaCy model still not available. Some features will be disabled.")
                    self.nlp = None
            
            logger.info("NLP tools initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing NLP tools: {e}")
            # Set defaults to prevent further errors
            self.stop_words = set()
            self.lemmatizer = WordNetLemmatizer()
            self.nlp = None
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str) or not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove user mentions (Reddit/Twitter style)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'u/\w+', '', text)
        text = re.sub(r'r/\w+', '', text)
        
        # Remove hashtags but keep the word
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def remove_special_characters(self, text: str, keep_crypto_symbols: bool = True) -> str:
        """
        Remove special characters
        
        Args:
            text: Input text
            keep_crypto_symbols: Keep $ symbol for crypto tickers
            
        Returns:
            Text without special characters
        """
        if not text:
            return ""
        
        if keep_crypto_symbols:
            # Keep $ for crypto symbols like $BTC
            text = re.sub(r'[^a-zA-Z0-9\s$]', '', text)
        else:
            # Remove all special characters except spaces
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        return text
    
    def remove_stopwords_from_text(self, text: str) -> str:
        """
        Remove stop words while preserving crypto terms
        
        Args:
            text: Input text
            
        Returns:
            Text without stop words
        """
        if not text:
            return ""
        
        words = text.split()
        
        # Keep crypto terms and non-stopwords
        filtered_words = [
            word for word in words
            if word.lower() not in self.stop_words or word.lower() in self.crypto_terms
        ]
        
        return ' '.join(filtered_words)
    
    def lemmatize_text(self, text: str) -> str:
        """
        Lemmatize words in text
        
        Args:
            text: Input text
            
        Returns:
            Lemmatized text
        """
        if not text:
            return ""
        
        try:
            words = text.split()
            lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
            return ' '.join(lemmatized_words)
        except:
            return text
    
    def preprocess(self, text: str, full_pipeline: bool = True) -> str:
        """
        Complete preprocessing pipeline
        
        Args:
            text: Raw text
            full_pipeline: Apply all preprocessing steps
            
        Returns:
            Preprocessed text
        """
        if not isinstance(text, str) or not text:
            return ""
        
        # Basic cleaning (always applied)
        text = self.clean_text(text)
        
        if full_pipeline:
            # Remove special characters
            text = self.remove_special_characters(text)
            
            # Remove stopwords (if enabled)
            if self.remove_stopwords:
                text = self.remove_stopwords_from_text(text)
            
            # Lemmatize (if enabled)
            if self.lemmatize:
                text = self.lemmatize_text(text)
        
        return text
    
    def extract_sentiment_features(self, text: str) -> Dict[str, float]:
        """
        Extract sentiment features using TextBlob
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment features
        """
        try:
            blob = TextBlob(text)
            
            return {
                'polarity': blob.sentiment.polarity,  # -1 to 1
                'subjectivity': blob.sentiment.subjectivity  # 0 to 1
            }
        except Exception as e:
            logger.warning(f"Error extracting sentiment: {e}")
            return {'polarity': 0.0, 'subjectivity': 0.0}
    
    def extract_text_statistics(self, text: str) -> Dict[str, int]:
        """
        Extract basic text statistics
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with text statistics
        """
        if not text:
            return {
                'char_count': 0,
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0.0,
                'unique_word_count': 0
            }
        
        try:
            words = text.split()
            sentences = sent_tokenize(text)
            
            return {
                'char_count': len(text),
                'word_count': len(words),
                'sentence_count': len(sentences),
                'avg_word_length': np.mean([len(word) for word in words]) if words else 0.0,
                'unique_word_count': len(set(words))
            }
        except Exception as e:
            logger.warning(f"Error extracting text statistics: {e}")
            return {
                'char_count': len(text),
                'word_count': len(text.split()),
                'sentence_count': 1,
                'avg_word_length': 0.0,
                'unique_word_count': len(set(text.split()))
            }
    
    def extract_named_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities using spaCy
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with entity types and values
        """
        if not self.nlp or not text:
            return {}
        
        try:
            doc = self.nlp(text)
            
            entities = {}
            for ent in doc.ents:
                if ent.label_ not in entities:
                    entities[ent.label_] = []
                entities[ent.label_].append(ent.text)
            
            return entities
            
        except Exception as e:
            logger.warning(f"Error extracting entities: {e}")
            return {}
    
    def detect_crypto_mentions(self, text: str) -> Dict[str, List[str]]:
        """
        Detect cryptocurrency mentions
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with detected crypto mentions
        """
        text_lower = text.lower()
        
        # Common crypto patterns
        crypto_patterns = {
            'bitcoin': r'\b(bitcoin|btc)\b',
            'ethereum': r'\b(ethereum|eth)\b',
            'cardano': r'\b(cardano|ada)\b',
            'solana': r'\b(solana|sol)\b',
            'polkadot': r'\b(polkadot|dot)\b',
            'dogecoin': r'\b(dogecoin|doge)\b',
            'ripple': r'\b(ripple|xrp)\b'
        }
        
        mentions = {}
        for crypto, pattern in crypto_patterns.items():
            matches = re.findall(pattern, text_lower)
            if matches:
                mentions[crypto] = matches
        
        # Detect $SYMBOL pattern
        ticker_mentions = re.findall(r'\$([A-Z]{2,5})\b', text)
        if ticker_mentions:
            mentions['tickers'] = ticker_mentions
        
        return mentions
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str = 'content') -> pd.DataFrame:
        """
        Process entire dataframe
        
        Args:
            df: Input dataframe
            text_column: Name of text column to process
            
        Returns:
            Dataframe with processed text and features
        """
        logger.info(f"Processing {len(df)} records...")
        
        # Combine title and content
        if 'title' in df.columns and text_column in df.columns:
            df['full_text'] = df['title'].fillna('') + ' ' + df[text_column].fillna('')
        else:
            df['full_text'] = df[text_column].fillna('')
        
        # Clean text
        logger.info("Cleaning text...")
        df['cleaned_text'] = df['full_text'].apply(lambda x: self.clean_text(x))
        
        # Preprocess text
        logger.info("Preprocessing text...")
        df['preprocessed_text'] = df['cleaned_text'].apply(lambda x: self.preprocess(x))
        
        # Extract sentiment features
        logger.info("Extracting sentiment features...")
        sentiment_features = df['cleaned_text'].apply(self.extract_sentiment_features)
        df['sentiment_polarity'] = sentiment_features.apply(lambda x: x['polarity'])
        df['sentiment_subjectivity'] = sentiment_features.apply(lambda x: x['subjectivity'])
        
        # Extract text statistics
        logger.info("Extracting text statistics...")
        text_stats = df['cleaned_text'].apply(self.extract_text_statistics)
        df['char_count'] = text_stats.apply(lambda x: x['char_count'])
        df['word_count'] = text_stats.apply(lambda x: x['word_count'])
        df['sentence_count'] = text_stats.apply(lambda x: x['sentence_count'])
        df['avg_word_length'] = text_stats.apply(lambda x: x['avg_word_length'])
        df['unique_word_count'] = text_stats.apply(lambda x: x['unique_word_count'])
        
        # Detect crypto mentions
        logger.info("Detecting crypto mentions...")
        df['crypto_mentions'] = df['cleaned_text'].apply(self.detect_crypto_mentions)
        df['has_crypto_mention'] = df['crypto_mentions'].apply(lambda x: len(x) > 0)
        
        logger.info("Processing complete!")
        
        return df