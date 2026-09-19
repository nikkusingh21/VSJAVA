"""
Inference Service for Customer Support Ticket Classification.
Provides real-time single ticket prediction, batch processing,
confidence scores, priority/urgency estimation, and department routing.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from src.preprocessor import clean_text, TextVocabulary, LabelEncoder, DEFAULT_CATEGORIES
from src.model import TicketBiLSTM

# Domain mappings for Department Routing & SLA
ROUTING_CONFIG = {
    "Payment Issue": {
        "department": "Billing & Financial Operations",
        "team_email": "billing-ops@support.company.com",
        "default_sla_hours": 4
    },
    "Login Problem": {
        "department": "Identity, Access & Security Support",
        "team_email": "security-support@support.company.com",
        "default_sla_hours": 2
    },
    "Order Status": {
        "department": "Order Management & Fulfillment",
        "team_email": "orders@support.company.com",
        "default_sla_hours": 12
    },
    "Refund Request": {
        "department": "Returns & Refund Processing",
        "team_email": "refunds@support.company.com",
        "default_sla_hours": 8
    },
    "Technical Support": {
        "department": "Engineering & Tier-2 Tech Support",
        "team_email": "tech-tier2@support.company.com",
        "default_sla_hours": 6
    },
    "Account Issue": {
        "department": "Customer Accounts & Profile Management",
        "team_email": "accounts@support.company.com",
        "default_sla_hours": 24
    },
    "Product Complaint": {
        "department": "Product Quality & Warranty Assurance",
        "team_email": "quality@support.company.com",
        "default_sla_hours": 12
    },
    "Delivery Issue": {
        "department": "Logistics & Courier Operations",
        "team_email": "logistics@support.company.com",
        "default_sla_hours": 8
    }
}

class TicketClassifier:
    """Production inference engine for ticket classification."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.device = torch.device("cpu") # Inference defaults to CPU for fast lightweight requests
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")

        self.vocab = None
        self.label_encoder = None
        self.model = None
        self.is_loaded = False
        
        self.load_model()

    def load_model(self) -> bool:
        vocab_path = os.path.join(self.models_dir, "vocab.json")
        encoder_path = os.path.join(self.models_dir, "label_encoder.json")
        weights_path = os.path.join(self.models_dir, "bilstm_ticket_classifier.pth")

        if not (os.path.exists(vocab_path) and os.path.exists(encoder_path) and os.path.exists(weights_path)):
            print(f"Warning: Model assets not found in {self.models_dir}. Need to run training.")
            self.is_loaded = False
            return False

        try:
            self.vocab = TextVocabulary.load(vocab_path)
            self.label_encoder = LabelEncoder.load(encoder_path)

            self.model = TicketBiLSTM(
                vocab_size=len(self.vocab),
                embed_dim=128,
                hidden_dim=128,
                num_classes=len(self.label_encoder),
                num_layers=2,
                dropout=0.0
            )
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.is_loaded = False
            return False

    def estimate_urgency(self, text: str) -> Dict[str, Any]:
        """Heuristic urgency evaluation based on NLP text cues."""
        text_lower = text.lower()
        
        critical_keywords = ["hazard", "burning", "smoke", "legal", "lawsuit", "unauthorized charge", "locked out completely", "stolen", "immediately", "fraud", "hacked"]
        high_keywords = ["urgent", "asap", "chargeback", "charged twice", "crashed", "500 internal", "cant login", "cannot log in", "delayed", "missing"]
        medium_keywords = ["wont work", "broken", "issue", "problem", "defect", "waiting", "cancel", "refund"]
        
        if any(w in text_lower for w in critical_keywords):
            return {"priority": "Critical", "sla_hours": 2, "urgency_score": 0.95}
        elif any(w in text_lower for w in high_keywords):
            return {"priority": "High", "sla_hours": 6, "urgency_score": 0.75}
        elif any(w in text_lower for w in medium_keywords):
            return {"priority": "Medium", "sla_hours": 24, "urgency_score": 0.50}
        else:
            return {"priority": "Low", "sla_hours": 48, "urgency_score": 0.25}

    def predict(self, text: str, max_seq_len: int = 100) -> Dict[str, Any]:
        """
        Classifies single ticket text:
        Returns top category, confidence score, all probabilities, urgency, department.
        """
        if not self.is_loaded:
            if not self.load_model():
                raise RuntimeError("Model is not trained yet! Please train the model first.")

        # Preprocess
        input_ids = torch.tensor([self.vocab.encode(text, max_len=max_seq_len)], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            probs = self.model.predict_proba(input_ids).cpu().numpy()[0]
            
        top_idx = int(np.argmax(probs))
        top_category = self.label_encoder.decode(top_idx)
        confidence = float(probs[top_idx])
        
        # Build full probability distribution sorted descending
        prob_distribution = []
        for idx, prob in enumerate(probs):
            prob_distribution.append({
                "category": self.label_encoder.decode(idx),
                "probability": round(float(prob), 4),
                "percentage": round(float(prob) * 100, 2)
            })
        prob_distribution.sort(key=lambda x: x["probability"], reverse=True)
        
        urgency_info = self.estimate_urgency(text)
        routing_info = ROUTING_CONFIG.get(top_category, {
            "department": "General Customer Support",
            "team_email": "support@company.com",
            "default_sla_hours": 24
        })

        return {
            "predicted_category": top_category,
            "confidence": round(confidence, 4),
            "confidence_percentage": round(confidence * 100, 2),
            "probabilities": prob_distribution,
            "priority": urgency_info["priority"],
            "urgency_score": urgency_info["urgency_score"],
            "department": routing_info["department"],
            "team_email": routing_info["team_email"],
            "sla_hours": min(urgency_info["sla_hours"], routing_info["default_sla_hours"])
        }

    def predict_batch(self, texts: List[str], max_seq_len: int = 100) -> List[Dict[str, Any]]:
        """Batch predictions for multiple tickets."""
        results = []
        for t in texts:
            results.append(self.predict(t, max_seq_len=max_seq_len))
        return results

    def predict_dataframe(self, df: pd.DataFrame, text_column: str = "ticket_text") -> pd.DataFrame:
        """Enriches an entire dataframe with predictions."""
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in dataframe.")
            
        results = self.predict_batch(df[text_column].tolist())
        df = df.copy()
        df["predicted_category"] = [r["predicted_category"] for r in results]
        df["confidence"] = [r["confidence"] for r in results]
        df["confidence_pct"] = [r["confidence_percentage"] for r in results]
        df["priority"] = [r["priority"] for r in results]
        df["department"] = [r["department"] for r in results]
        df["sla_hours"] = [r["sla_hours"] for r in results]
        return df
