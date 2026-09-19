"""
Script to create the Jupyter Notebook customer_support_ticket_classification.ipynb
"""
import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🎫 AI-Powered Customer Support Ticket Classifier Using Deep Learning and NLP\n",
    "\n",
    "## Project Overview\n",
    "Customer support departments receive thousands of tickets each day across varied channels. Manually reading, assessing urgency, and routing these tickets to the right team introduces substantial delay.\n",
    "\n",
    "This notebook demonstrates an end-to-end Deep Learning & NLP solution:\n",
    "1. **Exploratory Data Analysis (EDA)** on customer tickets\n",
    "2. **NLP Text Cleaning and Normalization**\n",
    "3. **Tokenization, Vocabulary Indexing, and PyTorch Datasets**\n",
    "4. **Bidirectional LSTM with Self-Attention Architecture**\n",
    "5. **Model Training, Validation, and Metric Tracking**\n",
    "6. **Multi-Class Evaluation (Accuracy, F1-Score, Confusion Matrix)**\n",
    "7. **Urgency Assessment & Departmental Routing Dispatch**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 1: Imports\n",
    "import os\n",
    "import sys\n",
    "import re\n",
    "import json\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import classification_report, confusion_matrix\n",
    "\n",
    "# Set seeds for reproducibility\n",
    "torch.manual_seed(42)\n",
    "np.random.seed(42)\n",
    "\n",
    "device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))\n",
    "print(f\"Using computational device: {device}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Load & Explore the Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_path = '../data/customer_support_tickets.csv'\n",
    "df = pd.read_csv(data_path)\n",
    "print(f\"Total ticket records: {len(df)}\")\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Category Class Distribution\n",
    "plt.figure(figsize=(10, 4))\n",
    "sns.countplot(data=df, y='ticket_type', order=df['ticket_type'].value_counts().index, palette='viridis')\n",
    "plt.title('Support Ticket Distribution Across 8 Business Categories')\n",
    "plt.xlabel('Ticket Count')\n",
    "plt.ylabel('Category')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Text Preprocessing & Cleaning\n",
    "Normalize URLs, Emails, Currency Amounts, and Order Numbers."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def clean_ticket_text(text):\n",
    "    if not isinstance(text, str):\n",
    "        return ''\n",
    "    text = text.lower()\n",
    "    text = re.sub(r'https?://\\S+|www\\.\\S+', ' [URL] ', text)\n",
    "    text = re.sub(r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b', ' [EMAIL] ', text)\n",
    "    text = re.sub(r'[\\$\\€\\£]\\s?\\d+(?:\\.\\d{2})?', ' [AMOUNT] ', text)\n",
    "    text = re.sub(r'#\\d+', ' [REF] ', text)\n",
    "    text = re.sub(r'[^\\w\\s\\[\\]]', ' ', text)\n",
    "    text = re.sub(r'\\s+', ' ', text).strip()\n",
    "    return text\n",
    "\n",
    "sample_raw = \"Charged twice $89.50 on card for order #98213! Check www.mybank.com\"\n",
    "print(\"Raw Text:    \", sample_raw)\n",
    "print(\"Cleaned Text:\", clean_ticket_text(sample_raw))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Deep Learning Model: PyTorch BiLSTM with Attention"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "class SelfAttention(nn.Module):\n",
    "    def __init__(self, hidden_dim):\n",
    "        super().__init__()\n",
    "        self.projection = nn.Sequential(\n",
    "            nn.Linear(hidden_dim, 64),\n",
    "            nn.Tanh(),\n",
    "            nn.Linear(64, 1)\n",
    "        )\n",
    "    def forward(self, lstm_outputs, mask=None):\n",
    "        energy = self.projection(lstm_outputs)\n",
    "        if mask is not None:\n",
    "            energy = energy.masked_fill(~mask.unsqueeze(-1), -1e9)\n",
    "        weights = F.softmax(energy, dim=1)\n",
    "        context = torch.sum(weights * lstm_outputs, dim=1)\n",
    "        return context, weights\n",
    "\n",
    "class TicketBiLSTM(nn.Module):\n",
    "    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=8, dropout=0.3):\n",
    "        super().__init__()\n",
    "        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)\n",
    "        self.embed_dropout = nn.Dropout(dropout)\n",
    "        self.bilstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True, dropout=dropout)\n",
    "        self.attention = SelfAttention(hidden_dim * 2)\n",
    "        self.classifier = nn.Sequential(\n",
    "            nn.Linear(hidden_dim * 2, 128),\n",
    "            nn.LayerNorm(128),\n",
    "            nn.ReLU(),\n",
    "            nn.Dropout(dropout),\n",
    "            nn.Linear(128, num_classes)\n",
    "        )\n",
    "    def forward(self, input_ids):\n",
    "        mask = (input_ids != 0)\n",
    "        embeds = self.embed_dropout(self.embedding(input_ids))\n",
    "        lstm_out, _ = self.bilstm(embeds)\n",
    "        context, _ = self.attention(lstm_out, mask=mask)\n",
    "        return self.classifier(context)\n",
    "\n",
    "print(\"Model architecture defined successfully.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Live Inference & Routing Prediction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load production inference engine\n",
    "sys.path.append('..')\n",
    "from src.inference import TicketClassifier\n",
    "\n",
    "classifier = TicketClassifier(models_dir='../models')\n",
    "\n",
    "test_inquiries = [\n",
    "    \"I was charged twice on my credit card for order #88921\",\n",
    "    \"SMS verification 2FA code is not coming to my smartphone\",\n",
    "    \"Package was marked delivered but front door porch was empty\",\n",
    "    \"The application crashes with segmentation fault on macOS Sonoma\"\n",
    "]\n",
    "\n",
    "for inquiry in test_inquiries:\n",
    "    res = classifier.predict(inquiry)\n",
    "    print(f\"\\n📝 Query: '{inquiry}'\")\n",
    "    print(f\"   ➔ Category: {res['predicted_category']} ({res['confidence_percentage']}%), Priority: {res['priority']}\")\n",
    "    print(f\"   ➔ Routing:  {res['department']} (Target SLA: {res['sla_hours']} Hours)\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open("notebooks/customer_support_ticket_classification.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Created notebooks/customer_support_ticket_classification.ipynb successfully!")
