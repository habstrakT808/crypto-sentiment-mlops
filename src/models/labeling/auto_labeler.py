"""
Auto Labeler
Ensemble-based automatic labeling using multiple pre-trained models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

from src.utils.logger import setup_logger
from src.utils.config import Config

logger = setup_logger(__name__)


class AutoLabeler:
    """Automatic labeling using ensemble of pre-trained models"""
    
    def __init__(self, device: str = None):
        """
        Initialize auto labeler
        
        Args:
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"AutoLabeler initialized on device: {self.device}")
        
        # Initialize VADER
        self.vader = SentimentIntensityAnalyzer()
        
        # Initialize FinBERT
        try:
            logger.info("Loading FinBERT model...")
            self.finbert_tokenizer = AutoTokenizer.from_pretrained(
                "ProsusAI/finbert"
            )
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained(
                "ProsusAI/finbert"
            ).to(self.device)
            self.finbert_model.eval()
            logger.info("FinBERT loaded successfully")
        except Exception as e:
            logger.error(f"Error loading FinBERT: {e}")
            self.finbert_model = None
            self.finbert_tokenizer = None
        
        # Label mapping
        self.label_map = {
            'negative': 0,
            'neutral': 1,
            'positive': 2
        }
        
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
    
    def _textblob_sentiment(self, text: str) -> Dict[str, float]:
        """
        Get sentiment from TextBlob
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment scores
        """
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            # Convert to 3-class
            if polarity > 0.1:
                label = 'positive'
                scores = {'negative': 0.0, 'neutral': 0.0, 'positive': 1.0}
            elif polarity < -0.1:
                label = 'negative'
                scores = {'negative': 1.0, 'neutral': 0.0, 'positive': 0.0}
            else:
                label = 'neutral'
                scores = {'negative': 0.0, 'neutral': 1.0, 'positive': 0.0}
            
            return {
                'label': label,
                'scores': scores,
                'confidence': abs(polarity)
            }
        except Exception as e:
            logger.warning(f"TextBlob error: {e}")
            return {
                'label': 'neutral',
                'scores': {'negative': 0.0, 'neutral': 1.0, 'positive': 0.0},
                'confidence': 0.0
            }
    
    def _vader_sentiment(self, text: str) -> Dict[str, float]:
        """
        Get sentiment from VADER
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment scores
        """
        try:
            scores = self.vader.polarity_scores(text)
            compound = scores['compound']
            
            # Convert to 3-class
            if compound >= 0.05:
                label = 'positive'
            elif compound <= -0.05:
                label = 'negative'
            else:
                label = 'neutral'
            
            # Normalize scores
            total = scores['neg'] + scores['neu'] + scores['pos']
            normalized_scores = {
                'negative': scores['neg'] / total if total > 0 else 0.0,
                'neutral': scores['neu'] / total if total > 0 else 1.0,
                'positive': scores['pos'] / total if total > 0 else 0.0
            }
            
            return {
                'label': label,
                'scores': normalized_scores,
                'confidence': abs(compound)
            }
        except Exception as e:
            logger.warning(f"VADER error: {e}")
            return {
                'label': 'neutral',
                'scores': {'negative': 0.0, 'neutral': 1.0, 'positive': 0.0},
                'confidence': 0.0
            }
    
    def _finbert_sentiment(self, text: str) -> Dict[str, float]:
        """
        Get sentiment from FinBERT
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment scores
        """
        if self.finbert_model is None or self.finbert_tokenizer is None:
            return {
                'label': 'neutral',
                'scores': {'negative': 0.0, 'neutral': 1.0, 'positive': 0.0},
                'confidence': 0.0
            }
        
        try:
            # Tokenize
            inputs = self.finbert_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.finbert_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                probs = probs.cpu().numpy()[0]
            
            # FinBERT outputs: [positive, negative, neutral]
            scores = {
                'positive': float(probs[0]),
                'negative': float(probs[1]),
                'neutral': float(probs[2])
            }
            
            label = max(scores, key=scores.get)
            confidence = scores[label]
            
            return {
                'label': label,
                'scores': scores,
                'confidence': confidence
            }
        except Exception as e:
            logger.warning(f"FinBERT error: {e}")
            return {
                'label': 'neutral',
                'scores': {'negative': 0.0, 'neutral': 1.0, 'positive': 0.0},
                'confidence': 0.0
            }
    
    def _ensemble_prediction(
        self,
        textblob_result: Dict,
        vader_result: Dict,
        finbert_result: Dict,
        weights: Dict[str, float] = None
    ) -> Dict[str, any]:
        """
        Combine predictions from multiple models
        
        Args:
            textblob_result: TextBlob prediction
            vader_result: VADER prediction
            finbert_result: FinBERT prediction
            weights: Model weights for ensemble
            
        Returns:
            Ensemble prediction dictionary
        """
        if weights is None:
            # Default weights (FinBERT gets highest weight for financial text)
            weights = {
                'textblob': 0.2,
                'vader': 0.3,
                'finbert': 0.5
            }
        
        # Combine scores
        ensemble_scores = {
            'negative': (
                weights['textblob'] * textblob_result['scores']['negative'] +
                weights['vader'] * vader_result['scores']['negative'] +
                weights['finbert'] * finbert_result['scores']['negative']
            ),
            'neutral': (
                weights['textblob'] * textblob_result['scores']['neutral'] +
                weights['vader'] * vader_result['scores']['neutral'] +
                weights['finbert'] * finbert_result['scores']['neutral']
            ),
            'positive': (
                weights['textblob'] * textblob_result['scores']['positive'] +
                weights['vader'] * vader_result['scores']['positive'] +
                weights['finbert'] * finbert_result['scores']['positive']
            )
        }
        
        # Get final label
        final_label = max(ensemble_scores, key=ensemble_scores.get)
        confidence = ensemble_scores[final_label]
        
        # Calculate agreement score (how many models agree)
        predictions = [
            textblob_result['label'],
            vader_result['label'],
            finbert_result['label']
        ]
        agreement_score = predictions.count(final_label) / len(predictions)
        
        return {
            'label': final_label,
            'label_id': self.label_map[final_label],
            'scores': ensemble_scores,
            'confidence': confidence,
            'agreement_score': agreement_score,
            'individual_predictions': {
                'textblob': textblob_result['label'],
                'vader': vader_result['label'],
                'finbert': finbert_result['label']
            },
            'individual_confidences': {
                'textblob': textblob_result['confidence'],
                'vader': vader_result['confidence'],
                'finbert': finbert_result['confidence']
            }
        }
    
    def label_single(self, text: str) -> Dict[str, any]:
        """
        Label a single text
        
        Args:
            text: Input text
            
        Returns:
            Labeling result dictionary
        """
        # Get predictions from all models
        textblob_result = self._textblob_sentiment(text)
        vader_result = self._vader_sentiment(text)
        finbert_result = self._finbert_sentiment(text)
        
        # Ensemble prediction
        ensemble_result = self._ensemble_prediction(
            textblob_result,
            vader_result,
            finbert_result
        )
        
        return ensemble_result
    
    def label_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'preprocessed_text',
        batch_size: int = 32
    ) -> pd.DataFrame:
        """
        Label entire dataframe
        
        Args:
            df: Input dataframe
            text_column: Column containing text to label
            batch_size: Batch size for processing
            
        Returns:
            Dataframe with labels and confidence scores
        """
        logger.info(f"Starting auto-labeling for {len(df)} samples...")
        
        results = []
        
        # Process in batches with progress bar
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Auto-labeling"):
            text = row[text_column]
            
            if pd.isna(text) or text == '':
                # Handle missing text
                result = {
                    'label': 'neutral',
                    'label_id': 1,
                    'scores': {'negative': 0.0, 'neutral': 1.0, 'positive': 0.0},
                    'confidence': 0.0,
                    'agreement_score': 0.0,
                    'individual_predictions': {
                        'textblob': 'neutral',
                        'vader': 'neutral',
                        'finbert': 'neutral'
                    },
                    'individual_confidences': {
                        'textblob': 0.0,
                        'vader': 0.0,
                        'finbert': 0.0
                    }
                }
            else:
                result = self.label_single(text)
            
            results.append(result)
        
        # Add results to dataframe
        df['auto_label'] = [r['label'] for r in results]
        df['auto_label_id'] = [r['label_id'] for r in results]
        df['label_confidence'] = [r['confidence'] for r in results]
        df['label_agreement'] = [r['agreement_score'] for r in results]
        
        # Add individual model predictions
        df['textblob_prediction'] = [r['individual_predictions']['textblob'] for r in results]
        df['vader_prediction'] = [r['individual_predictions']['vader'] for r in results]
        df['finbert_prediction'] = [r['individual_predictions']['finbert'] for r in results]
        
        # Add scores
        df['score_negative'] = [r['scores']['negative'] for r in results]
        df['score_neutral'] = [r['scores']['neutral'] for r in results]
        df['score_positive'] = [r['scores']['positive'] for r in results]
        
        logger.info("Auto-labeling complete!")
        
        # Log label distribution
        label_dist = df['auto_label'].value_counts()
        logger.info(f"Label distribution:\n{label_dist}")
        
        # Log confidence statistics
        logger.info(f"Average confidence: {df['label_confidence'].mean():.4f}")
        logger.info(f"Average agreement: {df['label_agreement'].mean():.4f}")
        
        return df
    
    def get_labeling_stats(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Get statistics about labeling results
        
        Args:
            df: Labeled dataframe
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_samples': len(df),
            'label_distribution': df['auto_label'].value_counts().to_dict(),
            'avg_confidence': float(df['label_confidence'].mean()),
            'avg_agreement': float(df['label_agreement'].mean()),
            'high_confidence_samples': int((df['label_confidence'] > 0.7).sum()),
            'high_agreement_samples': int((df['label_agreement'] >= 0.67).sum()),
            'model_agreement': {
                'all_agree': int((df['label_agreement'] == 1.0).sum()),
                'two_agree': int((df['label_agreement'] >= 0.67).sum()),
                'no_agreement': int((df['label_agreement'] < 0.67).sum())
            }
        }
        
        return stats