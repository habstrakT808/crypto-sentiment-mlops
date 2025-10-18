# File: src/models/deberta_model.py
"""
DeBERTaV3 Model
State-of-the-art transformer for sentiment analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DebertaV2Tokenizer,
    DebertaV2ForSequenceClassification,
    get_cosine_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from .base_model import BaseModel
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DeBERTaV3Model(BaseModel):
    """DeBERTaV3-based sentiment classifier with advanced training"""
    
    def __init__(
        self,
        model_name: str = 'microsoft/deberta-v3-base',
        max_len: int = 256,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        num_epochs: int = 5,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        device: str = None
    ):
        super().__init__(model_name="deberta_v3_sentiment", model_version="1.0.0")
        
        self.deberta_model_name = model_name
        self.max_len = max_len
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = None
        self.label_encoder = LabelEncoder()
        
        logger.info(f"DeBERTaV3Model initialized with {model_name} on {self.device}")
    
    def _get_framework_name(self) -> str:
        return "transformers"
    
    def train(
        self,
        X_train: pd.Series,
        y_train: np.ndarray,
        X_val: pd.Series = None,
        y_val: np.ndarray = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train DeBERTaV3 with advanced techniques"""
        logger.info(f"Training DeBERTaV3 on {len(X_train)} samples...")
        
        # Initialize tokenizer
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.deberta_model_name)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        num_labels = len(self.label_encoder.classes_)
        
        # Calculate class weights for focal loss
        class_counts = np.bincount(y_train_encoded)
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * num_labels
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        logger.info(f"Class weights: {class_weights}")
        
        # Initialize model
        self.model = DebertaV2ForSequenceClassification.from_pretrained(
            self.deberta_model_name,
            num_labels=num_labels
        ).to(self.device)
        
        # Create datasets
        from .bert_model import BERTDataset
        
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
        
        # Optimizer with weight decay
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Cosine schedule with warmup
        total_steps = len(train_loader) * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Loss function
        if self.use_focal_loss:
            criterion = FocalLoss(alpha=class_weights, gamma=self.focal_gamma)
            logger.info(f"Using Focal Loss with gamma={self.focal_gamma}")
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            logger.info("Using CrossEntropy Loss with class weights")
        
        # Training loop
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        
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
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                loss = criterion(logits, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({
                    'loss': loss.item(),
                    'acc': train_correct/train_total,
                    'lr': scheduler.get_last_lr()[0]
                })
            
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
                            attention_mask=attention_mask
                        )
                        
                        logits = outputs.logits
                        loss = criterion(logits, labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(logits, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                epoch_val_loss = val_loss / len(val_loader)
                epoch_val_acc = val_correct / val_total
                
                history['val_loss'].append(epoch_val_loss)
                history['val_acc'].append(epoch_val_acc)
                
                # Save best model
                if epoch_val_acc > best_val_acc:
                    best_val_acc = epoch_val_acc
                    logger.info(f"New best validation accuracy: {best_val_acc:.4f}")
                
                logger.info(
                    f"Epoch {epoch+1}: "
                    f"Train Loss={epoch_train_loss:.4f}, Train Acc={epoch_train_acc:.4f}, "
                    f"Val Loss={epoch_val_loss:.4f}, Val Acc={epoch_val_acc:.4f}"
                )
        
        self.is_trained = True
        self.training_history = history
        
        logger.info("DeBERTaV3 training complete!")
        return history
    
    def predict(self, X: pd.Series) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
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
        
        return self.label_encoder.inverse_transform(predictions)
    
    def predict_proba(self, X: pd.Series) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
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
                probs = torch.softmax(outputs.logits, dim=1)
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(probabilities)