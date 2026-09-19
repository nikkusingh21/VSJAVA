"""
Transformer (DistilBERT) Pipeline for Customer Support Ticket Classification.
Provides an alternative HuggingFace Transformer model architecture
for high-capacity contextual semantic representation.
"""

import os
import torch
import numpy as np
from typing import Dict, Any, List, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from src.preprocessor import DEFAULT_CATEGORIES
from src.inference import ROUTING_CONFIG

DEFAULT_TRANSFORMER_MODEL = "distilbert-base-uncased"

class DistilBertTicketClassifier:
    """
    HuggingFace DistilBERT Classifier for Support Tickets.
    Can be used for zero-shot classification, fine-tuned transformer inference,
    or contextual embeddings.
    """
    def __init__(self, model_name: str = DEFAULT_TRANSFORMER_MODEL, categories: Optional[List[str]] = None):
        self.model_name = model_name
        self.categories = categories or DEFAULT_CATEGORIES
        self.device = 0 if torch.cuda.is_available() else (-1 if not torch.backends.mps.is_available() else "mps")
        self.pipe = None
        self._init_pipeline()

    def _init_pipeline(self):
        try:
            # Initialize zero-shot classification pipeline using lightweight model
            print(f"Initializing Transformer Pipeline ({self.model_name})...")
            # For fast zero-shot/semantic classification on customer queries
            self.pipe = pipeline(
                "zero-shot-classification",
                model="typeform/distilbert-base-uncased-mnli",
                device=self.device
            )
            print("Transformer Pipeline initialized successfully.")
        except Exception as e:
            print(f"Note: Transformer zero-shot pipeline requires online model download: {e}")
            self.pipe = None

    def predict(self, text: str) -> Dict[str, Any]:
        """Classify using transformer pipeline."""
        if self.pipe is None:
            raise RuntimeError("Transformer pipeline is not available or offline.")

        result = self.pipe(text, candidate_labels=self.categories)
        top_category = result["labels"][0]
        confidence = float(result["scores"][0])
        
        prob_distribution = [
            {"category": label, "probability": round(float(score), 4), "percentage": round(float(score) * 100, 2)}
            for label, score in zip(result["labels"], result["scores"])
        ]
        
        routing_info = ROUTING_CONFIG.get(top_category, {
            "department": "General Customer Support",
            "team_email": "support@company.com",
            "default_sla_hours": 24
        })

        return {
            "model": "DistilBERT (Transformer)",
            "predicted_category": top_category,
            "confidence": round(confidence, 4),
            "confidence_percentage": round(confidence * 100, 2),
            "probabilities": prob_distribution,
            "department": routing_info["department"],
            "sla_hours": routing_info["default_sla_hours"]
        }
