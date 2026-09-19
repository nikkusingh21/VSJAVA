"""
Model Training Script for Customer Support Ticket Classifier.
Trains PyTorch BiLSTM with Attention on the ticket dataset,
evaluates performance, computes confusion matrix and classification report,
and saves model checkpoints and training history.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from src.preprocessor import clean_text, TextVocabulary, LabelEncoder, TicketDataset, DEFAULT_CATEGORIES
from src.model import TicketBiLSTM

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for input_ids, labels in dataloader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item() * len(labels)
        preds = torch.argmax(outputs, dim=-1)
        correct += (preds == labels).sum().item()
        total += len(labels)
        
    return total_loss / total, correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * len(labels)
            preds = torch.argmax(outputs, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return avg_loss, acc, f1, all_preds, all_labels

def run_training(
    data_path: str = "data/customer_support_tickets.csv",
    models_dir: str = "models",
    epochs: int = 20,
    batch_size: int = 32,
    embed_dim: int = 128,
    hidden_dim: int = 128,
    lr: float = 0.001,
    max_seq_len: int = 100
):
    start_time = time.time()
    device = get_device()
    print(f"Using device: {device}")
    
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load Dataset
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Running generator...")
        from data.generate_dataset import main as gen_main
        gen_main()
        
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} customer support tickets across {df['ticket_type'].nunique()} categories.")
    
    # 2. Setup Label Encoder
    label_encoder = LabelEncoder(DEFAULT_CATEGORIES)
    label_encoder.save(os.path.join(models_dir, "label_encoder.json"))
    
    df["label_id"] = df["ticket_type"].apply(label_encoder.encode)
    
    # Stratified Train/Val/Test Split (75% train, 15% val, 10% test)
    train_df, temp_df = train_test_split(df, test_size=0.25, random_state=42, stratify=df["label_id"])
    val_df, test_df = train_test_split(temp_df, test_size=0.40, random_state=42, stratify=temp_df["label_id"])
    
    print(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # 3. Build Vocabulary from Train set only
    vocab = TextVocabulary()
    vocab.build_vocab(train_df["ticket_text"].tolist(), min_freq=1, max_vocab_size=8000)
    vocab.save(os.path.join(models_dir, "vocab.json"))
    print(f"Vocabulary built with {len(vocab)} unique tokens.")
    
    # 4. Create PyTorch DataLoaders
    train_dataset = TicketDataset(train_df["ticket_text"].tolist(), train_df["label_id"].tolist(), vocab, max_seq_len)
    val_dataset = TicketDataset(val_df["ticket_text"].tolist(), val_df["label_id"].tolist(), vocab, max_seq_len)
    test_dataset = TicketDataset(test_df["ticket_text"].tolist(), test_df["label_id"].tolist(), vocab, max_seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 5. Initialize Model
    model = TicketBiLSTM(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_classes=len(label_encoder),
        num_layers=2,
        dropout=0.3
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    
    # 6. Training Loop with Checkpointing
    history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }
    
    best_val_loss = float("inf")
    best_weights_path = os.path.join(models_dir, "bilstm_ticket_classifier.pth")
    patience = 5
    patience_counter = 0
    
    print("\n--- Starting Model Training ---")
    for epoch in range(1, epochs + 1):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc, v_f1, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(v_loss)
        
        history["epochs"].append(epoch)
        history["train_loss"].append(round(t_loss, 4))
        history["val_loss"].append(round(v_loss, 4))
        history["train_acc"].append(round(t_acc, 4))
        history["val_acc"].append(round(v_acc, 4))
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {t_loss:.4f}, Train Acc: {t_acc*100:.2f}% | Val Loss: {v_loss:.4f}, Val Acc: {v_acc*100:.2f}%, Val F1: {v_f1*100:.2f}%")
        
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_weights_path)
            print(f"  -> Saved best model checkpoint (Val Loss: {v_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break
                
    # 7. Final Evaluation on Hold-out Test Set
    print("\n--- Evaluating on Test Set ---")
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_acc*100:.2f}% | Test F1-Score: {test_f1*100:.2f}%")
    
    class_names = [label_encoder.decode(i) for i in range(len(label_encoder))]
    report = classification_report(test_labels, test_preds, target_names=class_names, output_dict=True)
    cm = confusion_matrix(test_labels, test_preds).tolist()
    
    # 8. Save Model Metadata & Analytics
    metadata = {
        "model_architecture": "BiLSTM + Self-Attention Pooling",
        "num_classes": len(label_encoder),
        "vocab_size": len(vocab),
        "embedding_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "test_accuracy": round(float(test_acc), 4),
        "test_f1": round(float(test_f1), 4),
        "test_loss": round(float(test_loss), 4),
        "categories": class_names,
        "training_history": history,
        "classification_report": report,
        "confusion_matrix": cm,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_time_seconds": round(time.time() - start_time, 2)
    }
    
    metadata_path = os.path.join(models_dir, "model_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Model metadata & metrics saved to {metadata_path}")
    print("Model training complete!")
    return metadata

if __name__ == "__main__":
    run_training()
