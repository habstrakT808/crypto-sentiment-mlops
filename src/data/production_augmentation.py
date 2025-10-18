# File: src/data/production_augmentation.py
"""
Production Data Augmentation
Conservative augmentation to prevent overfitting
"""

import pandas as pd
import numpy as np
from typing import List
import random
from tqdm import tqdm
import nltk
from nltk.corpus import wordnet

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ProductionAugmenter:
    """Conservative augmentation for production"""
    
    def __init__(self):
        """Initialize augmenter"""
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        
        logger.info("ProductionAugmenter initialized")
    
    def _get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word"""
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)
        return list(synonyms)
    
    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """Replace n words with synonyms (CONSERVATIVE)"""
        words = text.split()
        
        if len(words) < 5:  # Don't augment short texts
            return text
        
        replaceable_indices = list(range(len(words)))
        random.shuffle(replaceable_indices)
        
        num_replaced = 0
        for idx in replaceable_indices:
            if num_replaced >= n:
                break
            
            word = words[idx]
            synonyms = self._get_synonyms(word)
            
            if synonyms:
                words[idx] = random.choice(synonyms)
                num_replaced += 1
        
        return ' '.join(words)
    
    def random_deletion(self, text: str, p: float = 0.05) -> str:
        """Randomly delete words (VERY CONSERVATIVE)"""
        words = text.split()
        
        if len(words) < 5:
            return text
        
        # Keep at least 80% of words
        new_words = [word for word in words if random.random() > p]
        
        if len(new_words) < len(words) * 0.8:
            return text
        
        return ' '.join(new_words)
    
    def augment_text(self, text: str, method: str = 'synonym') -> str:
        """Apply conservative augmentation"""
        if not text or len(text.split()) < 5:
            return text
        
        if method == 'synonym':
            return self.synonym_replacement(text, n=1)  # Only 1 replacement
        elif method == 'deletion':
            return self.random_deletion(text, p=0.05)  # Only 5% deletion
        else:
            return text
    
    def augment_minority_classes(
        self,
        df: pd.DataFrame,
        text_column: str = 'preprocessed_text',
        label_column: str = 'auto_label_id',
        max_augment_per_sample: int = 2,  # CONSERVATIVE!
        methods: List[str] = ['synonym']  # Only synonym replacement
    ) -> pd.DataFrame:
        """
        Conservatively augment minority classes
        
        Args:
            df: Input dataframe
            text_column: Text column name
            label_column: Label column name
            max_augment_per_sample: Maximum augmentations per original sample
            methods: Augmentation methods to use
            
        Returns:
            Augmented dataframe
        """
        logger.info(f"Conservatively augmenting minority classes...")
        
        # Get class distribution
        class_counts = df[label_column].value_counts()
        majority_count = class_counts.max()
        
        logger.info(f"Original distribution:\n{class_counts}")
        
        augmented_samples = []
        
        for label, count in class_counts.items():
            # Calculate samples needed (conservative target: 30% of majority)
            target_count = int(majority_count * 0.3)
            samples_needed = max(0, target_count - count)
            
            # Limit augmentation per sample
            class_samples = df[df[label_column] == label]
            max_total_augment = len(class_samples) * max_augment_per_sample
            samples_needed = min(samples_needed, max_total_augment)
            
            if samples_needed == 0:
                continue
            
            logger.info(f"Augmenting class {label}: {count} -> {target_count} (+{samples_needed} samples)")
            
            # Generate augmented samples
            for i in tqdm(range(samples_needed), desc=f"Augmenting class {label}"):
                sample = class_samples.sample(1).iloc[0]
                
                # Choose random method
                method = random.choice(methods)
                
                # Augment text
                augmented_text = self.augment_text(sample[text_column], method=method)
                
                # Skip if augmentation failed
                if augmented_text == sample[text_column]:
                    continue
                
                # Create new sample
                new_sample = sample.copy()
                new_sample[text_column] = augmented_text
                new_sample['is_augmented'] = True
                new_sample['augmentation_method'] = method
                
                augmented_samples.append(new_sample)
        
        # Combine
        if augmented_samples:
            augmented_df = pd.DataFrame(augmented_samples)
            df['is_augmented'] = False
            df['augmentation_method'] = 'original'
            
            combined_df = pd.concat([df, augmented_df], ignore_index=True)
            
            logger.info(f"Augmentation complete!")
            logger.info(f"Original: {len(df)}, Augmented: {len(augmented_samples)}, Total: {len(combined_df)}")
            logger.info(f"New distribution:\n{combined_df[label_column].value_counts()}")
            
            return combined_df
        else:
            df['is_augmented'] = False
            df['augmentation_method'] = 'original'
            return df