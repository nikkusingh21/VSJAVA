"""
Train lightweight model for Vercel Serverless deployment.
Uses TF-IDF Vectorizer with Logistic Regression / Calibrated Classifier
to provide fast, compact, high-accuracy inference within Vercel's 250MB limit.
"""

import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from src.preprocessor import clean_text

def train_lightweight_model(data_path="data/customer_support_tickets.csv", output_path="models/lightweight_model.joblib"):
    print("Training lightweight model for Vercel deployment...")
    df = pd.read_csv(data_path)
    
    # Preprocess text
    df["cleaned_text"] = df["ticket_text"].apply(clean_text)
    
    X = df["cleaned_text"]
    y = df["ticket_type"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
        ("clf", LogisticRegression(C=10.0, max_iter=1000, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Lightweight Model Test Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, preds))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(pipeline, output_path, compress=3)
    file_size_kb = os.path.getsize(output_path) / 1024
    print(f"Saved lightweight model to {output_path} (Size: {file_size_kb:.2f} KB)")

if __name__ == "__main__":
    train_lightweight_model()
