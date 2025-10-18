# File: src/data/advanced_augmentation.py
"""
Advanced Data Augmentation
State-of-the-art augmentation techniques for text data
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import random
from tqdm import tqdm
import nltk
from nltk.corpus import wordnet
import torch
from transformers import MarianMTModel, MarianTokenizer
from textattack.augmentation import EmbeddingAugmenter, WordNetAugmenter

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class AdvancedTextAugmenter:
    """Advanced text augmentation for NLP"""
    
    def __init__(self, device: str = None):
        """Initialize augmenter"""
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"AdvancedTextAugmenter initialized on {self.device}")
        
        # Download required NLTK data
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        
        # Initialize augmenters (lazy loading)
        self.back_translation_model = None
        self.back_translation_tokenizer = None
        self.wordnet_augmenter = None
        self.embedding_augmenter = None
    
    def _load_back_translation_models(self):
        """Load back-translation models (EN->DE->EN)"""
        if self.back_translation_model is None:
            logger.info("Loading back-translation models...")
            try:
                # English to German
                self.en_de_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
                self.en_de_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-de').to(self.device)
                
                # German to English
                self.de_en_tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-de-en')
                self.de_en_model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-de-en').to(self.device)
                
                logger.info("Back-translation models loaded")
            except Exception as e:
                logger.error(f"Error loading back-translation models: {e}")
                self.en_de_model = None
    
    def back_translate(self, text: str) -> str:
        """
        Back-translation augmentation (EN -> DE -> EN)
        
        Args:
            text: Input text
            
        Returns:
            Augmented text
        """
        if not text or len(text.split()) < 3:
            return text
        
        self._load_back_translation_models()
        
        if self.en_de_model is None:
            return text
        
        try:
            # English to German
            inputs = self.en_de_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
            translated = self.en_de_model.generate(**inputs)
            german_text = self.en_de_tokenizer.decode(translated[0], skip_special_tokens=True)
            
            # German to English
            inputs = self.de_en_tokenizer(german_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
            translated = self.de_en_model.generate(**inputs)
            back_translated = self.de_en_tokenizer.decode(translated[0], skip_special_tokens=True)
            
            return back_translated
        except Exception as e:
            logger.warning(f"Back-translation error: {e}")
            return text
    
    def synonym_replacement(self, text: str, n: int = 2) -> str:
        """
        Replace n words with synonyms using WordNet
        
        Args:
            text: Input text
            n: Number of words to replace
            
        Returns:
            Augmented text
        """
        words = text.split()
        
        if len(words) < 3:
            return text
        
        # Get replaceable words (nouns, verbs, adjectives)
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
    
    def _get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word using WordNet"""
        synonyms = set()
        
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)
        
        return list(synonyms)
    
    def random_insertion(self, text: str, n: int = 1) -> str:
        """
        Randomly insert n synonyms
        
        Args:
            text: Input text
            n: Number of insertions
            
        Returns:
            Augmented text
        """
        words = text.split()
        
        if len(words) < 3:
            return text
        
        for _ in range(n):
            # Choose random word
            random_word = random.choice(words)
            synonyms = self._get_synonyms(random_word)
            
            if synonyms:
                # Insert synonym at random position
                random_synonym = random.choice(synonyms)
                random_idx = random.randint(0, len(words))
                words.insert(random_idx, random_synonym)
        
        return ' '.join(words)
    
    def random_swap(self, text: str, n: int = 2) -> str:
        """
        Randomly swap n pairs of words
        
        Args:
            text: Input text
            n: Number of swaps
            
        Returns:
            Augmented text
        """
        words = text.split()
        
        if len(words) < 3:
            return text
        
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """
        Randomly delete words with probability p
        
        Args:
            text: Input text
            p: Deletion probability
            
        Returns:
            Augmented text
        """
        words = text.split()
        
        if len(words) < 3:
            return text
        
        # Keep at least 1 word
        new_words = [word for word in words if random.random() > p]
        
        if len(new_words) == 0:
            return random.choice(words)
        
        return ' '.join(new_words)
    
    def mixup_text(self, text1: str, text2: str, alpha: float = 0.5) -> str:
        """
        Mixup two texts (simple concatenation with mixing)
        
        Args:
            text1: First text
            text2: Second text
            alpha: Mixing ratio
            
        Returns:
            Mixed text
        """
        words1 = text1.split()
        words2 = text2.split()
        
        # Calculate number of words to take from each
        n1 = int(len(words1) * alpha)
        n2 = int(len(words2) * (1 - alpha))
        
        # Sample words
        sampled1 = random.sample(words1, min(n1, len(words1)))
        sampled2 = random.sample(words2, min(n2, len(words2)))
        
        # Combine and shuffle
        mixed_words = sampled1 + sampled2
        random.shuffle(mixed_words)
        
        return ' '.join(mixed_words)
    
    def augment_text(self, text: str, method: str = 'all') -> str:
        """
        Apply augmentation method
        
        Args:
            text: Input text
            method: Augmentation method ('synonym', 'insertion', 'swap', 'deletion', 'back_translate', 'all')
            
        Returns:
            Augmented text
        """
        if not text or len(text.split()) < 3:
            return text
        
        if method == 'synonym':
            return self.synonym_replacement(text)
        elif method == 'insertion':
            return self.random_insertion(text)
        elif method == 'swap':
            return self.random_swap(text)
        elif method == 'deletion':
            return self.random_deletion(text)
        elif method == 'back_translate':
            return self.back_translate(text)
        elif method == 'all':
            # Randomly choose a method
            methods = ['synonym', 'insertion', 'swap', 'deletion']
            chosen_method = random.choice(methods)
            return self.augment_text(text, chosen_method)
        else:
            return text
    
    def augment_minority_classes(
        self,
        df: pd.DataFrame,
        text_column: str = 'preprocessed_text',
        label_column: str = 'auto_label_id',
        target_ratio: float = 0.5,
        methods: List[str] = ['synonym', 'insertion', 'swap', 'back_translate']
    ) -> pd.DataFrame:
        """
        Augment minority classes to balance dataset
        
        Args:
            df: Input dataframe
            text_column: Text column name
            label_column: Label column name
            target_ratio: Target ratio for minority classes (relative to majority)
            methods: Augmentation methods to use
            
        Returns:
            Augmented dataframe
        """
        logger.info(f"Augmenting minority classes...")
        
        # Get class distribution
        class_counts = df[label_column].value_counts()
        majority_class = class_counts.idxmax()
        majority_count = class_counts.max()
        
        logger.info(f"Original distribution:\n{class_counts}")
        
        augmented_samples = []
        
        for label, count in class_counts.items():
            if label == majority_class:
                continue
            
            # Calculate target count
            target_count = int(majority_count * target_ratio)
            samples_needed = max(0, target_count - count)
            
            if samples_needed == 0:
                continue
            
            logger.info(f"Augmenting class {label}: {count} -> {target_count} (+{samples_needed} samples)")
            
            # Get samples from this class
            class_samples = df[df[label_column] == label]
            
            # Generate augmented samples
            for i in tqdm(range(samples_needed), desc=f"Augmenting class {label}"):
                # Sample random text from this class
                sample = class_samples.sample(1).iloc[0]
                
                # Choose random augmentation method
                method = random.choice(methods)
                
                # Augment text
                augmented_text = self.augment_text(sample[text_column], method=method)
                
                # Create new sample
                new_sample = sample.copy()
                new_sample[text_column] = augmented_text
                new_sample['is_augmented'] = True
                new_sample['augmentation_method'] = method
                
                augmented_samples.append(new_sample)
        
        # Combine original and augmented
        if augmented_samples:
            augmented_df = pd.DataFrame(augmented_samples)
            df['is_augmented'] = False
            df['augmentation_method'] = 'original'
            
            combined_df = pd.concat([df, augmented_df], ignore_index=True)
            
            logger.info(f"Augmentation complete!")
            logger.info(f"Original samples: {len(df)}")
            logger.info(f"Augmented samples: {len(augmented_samples)}")
            logger.info(f"Total samples: {len(combined_df)}")
            logger.info(f"New distribution:\n{combined_df[label_column].value_counts()}")
            
            return combined_df
        else:
            logger.info("No augmentation needed")
            df['is_augmented'] = False
            df['augmentation_method'] = 'original'
            return df


# Usage example
if __name__ == "__main__":
    # Test augmentation
    augmenter = AdvancedTextAugmenter()
    
    text = "Bitcoin price is rising rapidly today"
    
    print("Original:", text)
    print("Synonym:", augmenter.synonym_replacement(text))
    print("Insertion:", augmenter.random_insertion(text))
    print("Swap:", augmenter.random_swap(text))
    print("Deletion:", augmenter.random_deletion(text))