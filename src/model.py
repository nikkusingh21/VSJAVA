"""
Deep Learning Model Architecture:
PyTorch Bidirectional LSTM (BiLSTM) with Self-Attention mechanism
for multi-class customer support ticket classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    Self-attention layer to weight sequence time-steps dynamically
    based on their contextual importance.
    """
    def __init__(self, hidden_dim: int):
        super(SelfAttention, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, lstm_outputs: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # lstm_outputs shape: [batch_size, seq_len, hidden_dim]
        energy = self.projection(lstm_outputs)  # [batch_size, seq_len, 1]
        
        if mask is not None:
            # Mask out padding tokens (mask: [batch_size, seq_len], True where valid, False where PAD)
            energy = energy.masked_fill(~mask.unsqueeze(-1), -1e9)
            
        weights = F.softmax(energy, dim=1)  # [batch_size, seq_len, 1]
        context_vector = torch.sum(weights * lstm_outputs, dim=1)  # [batch_size, hidden_dim]
        return context_vector, weights

class TicketBiLSTM(nn.Module):
    """
    Deep Neural Network for Support Ticket Classification:
    Word Embeddings -> Bidirectional LSTM -> Self-Attention Pooling -> Dense Classifier
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 8,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0
    ):
        super(TicketBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embed_dropout = nn.Dropout(dropout)
        
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        bilstm_out_dim = hidden_dim * 2
        self.attention = SelfAttention(bilstm_out_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(bilstm_out_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [batch_size, seq_len]
        mask = (input_ids != 0)  # padding_idx is 0
        
        embeds = self.embed_dropout(self.embedding(input_ids))  # [batch_size, seq_len, embed_dim]
        lstm_out, _ = self.bilstm(embeds)  # [batch_size, seq_len, hidden_dim * 2]
        
        context, _ = self.attention(lstm_out, mask=mask)  # [batch_size, hidden_dim * 2]
        logits = self.classifier(context)  # [batch_size, num_classes]
        return logits

    def predict_proba(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities for class distribution."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids)
            return F.softmax(logits, dim=-1)
