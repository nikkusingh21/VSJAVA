"""
NLP Preprocessing module for text normalization, tokenization,
vocabulary building, and PyTorch dataset creation.
"""

import re
import json
import os
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

DEFAULT_CATEGORIES = [
    "Payment Issue",
    "Login Problem",
    "Order Status",
    "Refund Request",
    "Technical Support",
    "Account Issue",
    "Product Complaint",
    "Delivery Issue"
]

def clean_text(text: str) -> str:
    """
    Cleans raw customer ticket text:
    - Lowercases text
    - Replaces URLs with [URL]
    - Replaces emails with [EMAIL]
    - Replaces currency amounts with [AMOUNT]
    - Replaces order IDs / reference numbers with [REF]
    - Strips non-alphanumeric noise while retaining meaningful punctuation
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    # Replace URLs
    text = re.sub(r"https?://\S+|www\.\S+", " [URL] ", text)
    
    # Replace Emails
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " [EMAIL] ", text)
    
    # Replace Currency amounts (e.g. $45.99, €20, £150)
    text = re.sub(r"[\$\€\£]\s?\d+(?:\.\d{2})?", " [AMOUNT] ", text)
    
    # Replace Order/Tracking references like #123456 or REF-9923
    text = re.sub(r"#\d+", " [REF] ", text)
    text = re.sub(r"ref-?\d+", " [REF] ", text)
    
    # Remove special characters but keep brackets for special tokens and alphanumeric
    text = re.sub(r"[^\w\s\[\]]", " ", text)
    
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str) -> List[str]:
    """Tokenize cleaned text by whitespace."""
    return text.split()

class TextVocabulary:
    """Vocabulary mapper between tokens and integer IDs."""
    
    def __init__(self, pad_token: str = PAD_TOKEN, unk_token: str = UNK_TOKEN):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.token2idx = {pad_token: 0, unk_token: 1}
        self.idx2token = {0: pad_token, 1: unk_token}
        
    def build_vocab(self, texts: List[str], min_freq: int = 1, max_vocab_size: int = 10000):
        freqs = {}
        for text in texts:
            tokens = tokenize(clean_text(text))
            for t in tokens:
                freqs[t] = freqs.get(t, 0) + 1
                
        sorted_tokens = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
        
        for token, count in sorted_tokens:
            if count < min_freq:
                break
            if len(self.token2idx) >= max_vocab_size:
                break
            if token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token
                
    def encode(self, text: str, max_len: int = 128) -> List[int]:
        tokens = tokenize(clean_text(text))
        ids = [self.token2idx.get(t, self.token2idx[self.unk_token]) for t in tokens]
        
        # Truncate or pad
        if len(ids) > max_len:
            ids = ids[:max_len]
        else:
            ids = ids + [self.token2idx[self.pad_token]] * (max_len - len(ids))
        return ids

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "token2idx": self.token2idx,
                "pad_token": self.pad_token,
                "unk_token": self.unk_token
            }, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "TextVocabulary":
        vocab = cls()
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab.token2idx = data["token2idx"]
        vocab.idx2token = {int(v): k for k, v in vocab.token2idx.items()}
        vocab.pad_token = data.get("pad_token", PAD_TOKEN)
        vocab.unk_token = data.get("unk_token", UNK_TOKEN)
        return vocab

    def __len__(self):
        return len(self.token2idx)

class LabelEncoder:
    """Encodes category string labels to class integers and vice-versa."""
    
    def __init__(self, categories: Optional[List[str]] = None):
        self.categories = categories or DEFAULT_CATEGORIES
        self.label2idx = {cat: i for i, cat in enumerate(self.categories)}
        self.idx2label = {i: cat for i, cat in enumerate(self.categories)}

    def encode(self, label: str) -> int:
        return self.label2idx.get(label, -1)

    def decode(self, idx: int) -> str:
        return self.idx2label.get(idx, "Unknown")

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "categories": self.categories,
                "label2idx": self.label2idx,
                "idx2label": self.idx2label
            }, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "LabelEncoder":
        encoder = cls()
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        encoder.categories = data["categories"]
        encoder.label2idx = data["label2idx"]
        encoder.idx2label = {int(k): v for k, v in data["idx2label"].items()}
        return encoder

    def __len__(self):
        return len(self.categories)

class TicketDataset(Dataset):
    """PyTorch Dataset for ticket text classification."""
    
    def __init__(self, texts: List[str], labels: List[int], vocab: TextVocabulary, max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        input_ids = torch.tensor(self.vocab.encode(text, self.max_len), dtype=torch.long)
        return input_ids, torch.tensor(label, dtype=torch.long)
