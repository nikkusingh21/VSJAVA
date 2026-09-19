# 🎫 AI-Powered Customer Support Ticket Classifier Using Deep Learning and NLP

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C.svg)](https://pytorch.org)
[![Vercel](https://img.shields.io/badge/Deploy-Vercel-black.svg?logo=vercel)](VERCEL_DEPLOYMENT.md)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.64-FF4B4B.svg)](https://streamlit.io)
[![Accuracy](https://img.shields.io/badge/Test_Accuracy-100%25-success.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

An enterprise-grade Deep Learning and Natural Language Processing (NLP) system designed to automatically classify customer support tickets into predefined operational categories, quantify confidence scores, detect customer urgency/priority, and route tickets to appropriate departments with SLA enforcement. **Pre-configured for 1-click deployment on Vercel.**

---

## 📌 Problem Statement & Objectives

Enterprises receive thousands of unstructured customer support requests daily across multiple touchpoints (Email, Web Portal, Live Chat, Mobile App). Manual triaging, reading, and assigning tickets is labor-intensive, error-prone, and leads to delayed responses.

### Key Objectives
- **Automatic Classification**: Classify raw text into 8 core enterprise categories using Deep Learning.
- **Urgency & SLA Assignment**: Estimate issue priority (`Critical`, `High`, `Medium`, `Low`) and automatically assign departmental resolution SLAs.
- **Department Routing**: Route customer issues directly to dedicated support queues (e.g. Billing, Identity/Security, Logistics).
- **Interactive Web Interface**: Provide an intuitive, modern Streamlit dashboard for real-time single ticket classification, bulk CSV processing, and operational analytics.
- **Persistent Ticket Log**: Store and manage classified tickets in a local SQLite database with status workflows (`Open`, `In Progress`, `Resolved`).

---

## 🏷️ Supported Ticket Categories

The classifier categorizes tickets across 8 business domains:

| # | Category | Target Department | Default SLA | Example Query |
|---|---|---|---|---|
| 1 | **Payment Issue** | Billing & Financial Operations | 4 Hours | *"My credit card was charged twice for order #98213"* |
| 2 | **Login Problem** | Identity, Access & Security Support | 2 Hours | *"2FA verification SMS code never arrives, locked out"* |
| 3 | **Order Status** | Order Management & Fulfillment | 12 Hours | *"Order has been stuck in transit for 8 days, need tracking"* |
| 4 | **Refund Request** | Returns & Refund Processing | 8 Hours | *"Returned defective item 10 days ago, where is my refund?"* |
| 5 | **Technical Support** | Engineering & Tier-2 Tech Support | 6 Hours | *"App crashes with HTTP 500 error on exporting PDF report"* |
| 6 | **Account Issue** | Customer Accounts & Profile Management | 24 Hours | *"Need to update my registered email and merge company licenses"* |
| 7 | **Product Complaint** | Product Quality & Warranty Assurance | 12 Hours | *"Monitor arrived with scratches and dead pixels on display"* |
| 8 | **Delivery Issue** | Logistics & Courier Operations | 8 Hours | *"Package marked delivered but nothing was left on front porch"* |

---

## 🧠 Deep Learning Architecture

```
[Raw Customer Ticket Text]
           │
           ▼
[NLP Normalization & Preprocessing]
(Lowercasing, Regex Regex [URL]/[EMAIL]/[AMOUNT]/[REF], Whitespace Tokenization)
           │
           ▼
[Embedding Layer (128-dim, vocab_size ~1,000)]
           │
           ▼
[Bidirectional LSTM (2 Layers, 128 hidden_dim, Dropout=0.3)]
           │
           ▼
[Multi-Head Self-Attention Pooling]
(Dynamically weights sequence time-steps based on contextual importance)
           │
           ▼
[Dense Classification Head]
(LayerNorm -> ReLU -> Dropout -> Linear(8 classes))
           │
           ▼
[Softmax Output Probabilities & Confidence Score]
```

### Why BiLSTM with Self-Attention?
- **Bidirectional Context**: Captures customer intent from both preceding and succeeding words (e.g., distinguishing *"refund for cancelled item"* vs *"cancel my refund request"*).
- **Self-Attention Pooling**: Standard LSTMs suffer from bottlenecking the entire message into the final hidden state. Attention allows the network to focus on high-impact problem keywords regardless of where they appear in long complaint texts.
- **Hardware Acceleration**: Natively runs on Apple Silicon GPU (`mps`), CUDA (`cuda`), or fallback `cpu`.

---

## 📂 Project Structure

```
.
├── app.py                             # Interactive Streamlit Web Application
├── predict_cli.py                     # Command-line prediction & batch tool
├── requirements.txt                   # Project dependencies
├── data/
│   ├── customer_support_tickets.csv   # 1,600 realistic training tickets across 8 classes
│   ├── sample_batch_tickets.csv      # Sample batch CSV for demonstration
│   ├── generate_dataset.py            # Dataset generation script
│   └── tickets.db                     # SQLite persistence database
├── models/
│   ├── bilstm_ticket_classifier.pth   # Trained PyTorch model weights
│   ├── vocab.json                     # Vocabulary mapping
│   ├── label_encoder.json             # Category class mapping
│   └── model_metadata.json           # Accuracy curves, confusion matrix, metrics
├── src/
│   ├── __init__.py
│   ├── preprocessor.py               # Text cleaning, vocab builder, PyTorch dataset
│   ├── model.py                      # PyTorch BiLSTM + Self-Attention architecture
│   ├── train.py                      # Training loop with validation & early stopping
│   ├── inference.py                  # Predictor engine, urgency, and routing rules
│   └── database.py                   # SQLite CRUD operations and statistics
└── tests/
    └── test_inference.py              # Automated test suite across all 8 classes
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup

Ensure Python 3.10+ is installed:

```bash
# Clone or navigate to the directory
cd "DLNLP project"

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web App

```bash
source .venv/bin/activate
streamlit run app.py
```

Open your browser at **`http://localhost:8501`**.

### 3. Features in the Web App
1. **🎫 Single Ticket Classifier**:
   - Select from realistic presets or type any custom customer inquiry.
   - View predicted category, confidence score, urgency priority, and target SLA.
   - View full horizontal probability breakdown bar chart.
   - Automatically logs classified tickets to the local database.
2. **📂 Batch CSV Processing**:
   - Upload any custom CSV or load the bundled 25-ticket Kaggle sample.
   - Process hundreds of tickets in seconds.
   - Export enriched CSV with predicted categories, confidence %, priority, and routing.
3. **📊 Analytics Operations Dashboard**:
   - Executive KPIs: Total tickets, Average confidence, Open vs Resolved count.
   - Category distribution donut chart.
   - Urgency breakdown stacked bar chart.
   - Support channel distribution and departmental SLAs.
4. **🗃️ Ticket History & Logs**:
   - Filter tickets by category, priority, or status.
   - Update ticket lifecycle: `Open` ➔ `In Progress` ➔ `Resolved`.
   - Export history to CSV.
5. **🧠 Model Metrics & Insights**:
   - View test accuracy and F1 score (100% on hold-out test set).
   - Interactive training vs. validation loss and accuracy curves.
   - Multi-class Confusion Matrix heatmap.
   - Per-class Precision, Recall, and F1 table.

---

## 💻 Command-Line Interface (CLI)

Classify tickets directly from your terminal:

```bash
# Single ticket prediction
python predict_cli.py "My credit card was charged twice for order #88123"

# Batch classify a CSV file
python predict_cli.py --batch data/sample_batch_tickets.csv

# Output as JSON
python predict_cli.py "App crashes with 500 internal server error" --json
```

---

## 🌐 Flask REST API Microservice

Run headless API server for external microservices or bot integrations:

```bash
# Start Flask API server (runs on port 5050)
PYTHONPATH=. python api.py
```

### Endpoints
- `GET  /api/health` - Health check & model loaded status
- `POST /api/predict` - Classify a support ticket
- `POST /api/predict_batch` - Classify multiple tickets in one call
- `GET  /api/stats` - Total tickets and category counts
- `GET  /api/history` - Query ticket history with filters
- `PATCH /api/tickets/<ticket_id>` - Update resolution status

#### Example cURL Request:
```bash
curl -X POST http://localhost:5050/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I was charged twice on my credit card for order #88123"}'
```

---

## 📓 Interactive Jupyter Notebook

Run the step-by-step EDA, training, and evaluation notebook:

```bash
jupyter notebook notebooks/customer_support_ticket_classification.ipynb
```

---

## 🧪 Running Automated Tests & Re-training

```bash
# Run model inference test suite (8 classes)
PYTHONPATH=. python tests/test_inference.py

# Run Flask REST API test suite
PYTHONPATH=. python tests/test_api.py

# Re-train the deep learning model from scratch
PYTHONPATH=. python src/train.py
```

---

## 📊 Evaluation & Performance

| Metric | Score |
|---|---|
| **Test Accuracy** | **100.00%** |
| **Weighted F1-Score** | **100.00%** |
| **Test Loss** | **0.0004** |
| **Inference Latency** | **< 15 ms / ticket** |
| **Hardware Acceleration** | Apple Silicon MPS / NVIDIA CUDA / CPU |
