"""
LSTM Model
Bidirectional LSTM with attention for sentiment analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from .base_model import BaseModel
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SentimentDataset(Dataset):
    """Dataset for sentiment analysis"""
    
    def __init__(self, texts: List[str], labels: np.ndarray, vocab: Dict[str, int], max_len: int = 200):
        """
        Initialize dataset
        
        Args:
            texts: List of texts
            labels: Array of labels
            vocab: Vocabulary dictionary
            max_len: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize and convert to indices
        tokens = text.lower().split()
        indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        # Pad or truncate
        if len(indices) < self.max_len:
            indices = indices + [self.vocab['<PAD>']] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class LSTMClassifier(nn.Module):
    """Bidirectional LSTM with attention"""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.5,
        bidirectional: bool = True
    ):
        """
        Initialize LSTM classifier
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Hidden dimension
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Attention layer
        self.attention = nn.Linear(lstm_output_dim, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(lstm_output_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        self.relu = nn.ReLU()
    
    def attention_net(self, lstm_output):
        """
        Apply attention mechanism
        
        Args:
            lstm_output: LSTM output
            
        Returns:
            Attention-weighted output
        """
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        weighted_output = torch.sum(attention_weights * lstm_output, dim=1)
        return weighted_output
    
    def forward(self, x):
        """Forward pass"""
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)
        
        # Attention
        attn_output = self.attention_net(lstm_out)
        
        # Fully connected
        out = self.relu(self.fc1(attn_output))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class LSTMModel(BaseModel):
    """LSTM-based sentiment classifier"""
    
    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        max_vocab_size: int = 10000,
        max_len: int = 200,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        num_epochs: int = 10,
        device: str = None
    ):
        """
        Initialize LSTM model
        
        Args:
            embedding_dim: Embedding dimension
            hidden_dim: Hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            max_vocab_size: Maximum vocabulary size
            max_len: Maximum sequence length
            batch_size: Batch size
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            device: Device to use
        """
        super().__init__(model_name="lstm_sentiment", model_version="1.0.0")
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_vocab_size = max_vocab_size
        self.max_len = max_len
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.vocab = None
        self.label_encoder = LabelEncoder()
        
        # Update metadata
        self.metadata.update({
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout': dropout,
            'max_vocab_size': max_vocab_size,
            'device': self.device
        })
        
        logger.info(f"LSTMModel initialized on device: {self.device}")
    
    def _get_framework_name(self) -> str:
        """Get framework name"""
        return "pytorch"
    
    def _build_vocab(self, texts: List[str]) -> Dict[str, int]:
        """
        Build vocabulary from texts
        
        Args:
            texts: List of texts
            
        Returns:
            Vocabulary dictionary
        """
        logger.info("Building vocabulary...")
        
        # Count word frequencies
        word_freq = {}
        for text in texts:
            for word in text.lower().split():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Create vocabulary (reserve indices for special tokens)
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for word, _ in sorted_words[:self.max_vocab_size - 2]:
            vocab[word] = len(vocab)
        
        logger.info(f"Vocabulary size: {len(vocab)}")
        return vocab
    
    def train(
        self,
        X_train: pd.Series,
        y_train: np.ndarray,
        X_val: pd.Series = None,
        y_val: np.ndarray = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train LSTM model
        
        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts
            y_val: Validation labels
            
        Returns:
            Training history
        """
        logger.info(f"Training LSTM model on {len(X_train)} samples...")
        
        # Build vocabulary
        self.vocab = self._build_vocab(X_train.tolist())
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        # Create datasets
        train_dataset = SentimentDataset(
            X_train.tolist(),
            y_train_encoded,
            self.vocab,
            self.max_len
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # Validation loader
        val_loader = None
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            val_dataset = SentimentDataset(
                X_val.tolist(),
                y_val_encoded,
                self.vocab,
                self.max_len
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0
            )
        
        # Initialize model
        self.model = LSTMClassifier(
            vocab_size=len(self.vocab),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_classes=len(self.label_encoder.classes_),
            dropout=self.dropout,
            bidirectional=True
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
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
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
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
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
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
        self.metadata['num_epochs'] = self.num_epochs
        
        logger.info("Training complete!")
        return history
    
    def predict(self, X: pd.Series) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input texts
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        self.model.eval()
        
        # Create dataset
        dummy_labels = np.zeros(len(X))
        dataset = SentimentDataset(X.tolist(), dummy_labels, self.vocab, self.max_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
        
        # Decode labels
        predictions = self.label_encoder.inverse_transform(predictions)
        return predictions
    
    def predict_proba(self, X: pd.Series) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Input texts
            
        Returns:
            Class probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        self.model.eval()
        
        # Create dataset
        dummy_labels = np.zeros(len(X))
        dataset = SentimentDataset(X.tolist(), dummy_labels, self.vocab, self.max_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        probabilities = []
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(probabilities)