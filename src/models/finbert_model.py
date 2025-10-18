"""
FinBERT Model
Fine-tuned FinBERT for financial sentiment analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from .base_model import BaseModel
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class FinBERTModel(BaseModel):
    """FinBERT-based sentiment classifier for financial text"""
    
    def __init__(
        self,
        model_name: str = 'ProsusAI/finbert',
        max_len: int = 128,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        warmup_steps: int = 0,
        device: str = None
    ):
        """
        Initialize FinBERT model
        
        Args:
            model_name: Pre-trained FinBERT model name
            max_len: Maximum sequence length
            batch_size: Batch size
            learning_rate: Learning rate
            num_epochs: Number of epochs
            warmup_steps: Warmup steps
            device: Device to use
        """
        super().__init__(model_name="finbert_sentiment", model_version="1.0.0")
        
        self.finbert_model_name = model_name
        self.max_len = max_len
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = None
        self.label_encoder = LabelEncoder()
        
        # Update metadata
        self.metadata.update({
            'finbert_model': model_name,
            'max_len': max_len,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'device': self.device
        })
        
        logger.info(f"FinBERTModel initialized with {model_name} on device: {self.device}")
    
    def _get_framework_name(self) -> str:
        """Get framework name"""
        return "transformers"
    
    def train(
        self,
        X_train: pd.Series,
        y_train: np.ndarray,
        X_val: pd.Series = None,
        y_val: np.ndarray = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train FinBERT model (similar to BERT but with financial domain)"""
        logger.info(f"Training FinBERT model on {len(X_train)} samples...")
        
        from .bert_model import BERTDataset
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.finbert_model_name)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        num_labels = len(self.label_encoder.classes_)
        
        # Initialize model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.finbert_model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        ).to(self.device)
        
        # Create datasets
        train_dataset = BERTDataset(
            X_train.tolist(),
            y_train_encoded,
            self.tokenizer,
            self.max_len
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # Validation loader
        val_loader = None
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            val_dataset = BERTDataset(
                X_val.tolist(),
                y_val_encoded,
                self.tokenizer,
                self.max_len
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
        
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(train_loader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({'loss': loss.item(), 'acc': train_correct/train_total})
            
            epoch_train_loss = train_loss / len(train_loader)
            epoch_train_acc = train_correct / train_total
            
            history['train_loss'].append(epoch_train_loss)
            history['train_acc'].append(epoch_train_acc)
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        
                        loss = outputs.loss
                        logits = outputs.logits
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(logits, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                epoch_val_loss = val_loss / len(val_loader)
                epoch_val_acc = val_correct / val_total
                
                history['val_loss'].append(epoch_val_loss)
                history['val_acc'].append(epoch_val_acc)
                
                logger.info(
                    f"Epoch {epoch+1}: "
                    f"Train Loss={epoch_train_loss:.4f}, Train Acc={epoch_train_acc:.4f}, "
                    f"Val Loss={epoch_val_loss:.4f}, Val Acc={epoch_val_acc:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch+1}: "
                    f"Train Loss={epoch_train_loss:.4f}, Train Acc={epoch_train_acc:.4f}"
                )
        
        self.is_trained = True
        self.training_history = history
        
        # Update metadata
        self.metadata['trained_at'] = pd.Timestamp.now().isoformat()
        self.metadata['training_samples'] = len(X_train)
        
        logger.info("FinBERT training complete!")
        return history
    
    def predict(self, X: pd.Series) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        from .bert_model import BERTDataset
        
        self.model.eval()
        dummy_labels = np.zeros(len(X))
        dataset = BERTDataset(X.tolist(), dummy_labels, self.tokenizer, self.max_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                _, predicted = torch.max(logits, 1)
                predictions.extend(predicted.cpu().numpy())
        
        predictions = self.label_encoder.inverse_transform(predictions)
        return predictions
    
    def predict_proba(self, X: pd.Series) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        from .bert_model import BERTDataset
        
        self.model.eval()
        dummy_labels = np.zeros(len(X))
        dataset = BERTDataset(X.tolist(), dummy_labels, self.tokenizer, self.max_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        probabilities = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(probabilities)