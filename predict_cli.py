#!/usr/bin/env python3
"""
CLI Tool for AI Support Ticket Classification.
Usage:
    python predict_cli.py "My credit card was charged twice for order #88123"
    python predict_cli.py --batch data/sample_batch_tickets.csv
"""

import sys
import os
import argparse
import json

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.inference import TicketClassifier
from src.database import save_ticket

def main():
    parser = argparse.ArgumentParser(description="AI Customer Support Ticket Classifier CLI")
    parser.add_argument("text", nargs="?", type=str, help="Ticket message text to classify")
    parser.add_argument("--batch", type=str, help="Path to CSV file for batch classification")
    parser.add_argument("--save", action="store_true", help="Save prediction to SQLite database")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()

    clf = TicketClassifier()
    if not clf.is_loaded:
        print("Error: Trained model weights not found in models/. Run 'python src/train.py' first.", file=sys.stderr)
        sys.exit(1)

    if args.batch:
        import pandas as pd
        if not os.path.exists(args.batch):
            print(f"Error: Batch file {args.batch} not found.", file=sys.stderr)
            sys.exit(1)
        
        df = pd.read_csv(args.batch)
        # Prioritize true text columns over identifier columns
        text_cols = [c for c in df.columns if any(k in c.lower() for k in ["text", "desc", "message", "body", "issue", "content"]) and not any(neg in c.lower() for neg in ["id", "num", "code"])]
        if not text_cols:
            text_cols = [c for c in df.columns if "ticket" in c.lower() and not any(neg in c.lower() for neg in ["id", "num"])]
        text_col = text_cols[0] if text_cols else df.columns[0]
        
        print(f"Classifying {len(df)} tickets using column '{text_col}'...")
        enriched = clf.predict_dataframe(df, text_column=text_col)
        out_path = args.batch.replace(".csv", "_classified.csv")
        enriched.to_csv(out_path, index=False)
        print(f"Saved classified results to: {out_path}")
        print("\nSummary of Predictions:")
        print(enriched["predicted_category"].value_counts().to_string())
        return

    if not args.text:
        parser.print_help()
        sys.exit(1)

    res = clf.predict(args.text)

    if args.save:
        import uuid
        ticket_id = f"TCK-{str(uuid.uuid4())[:8].upper()}"
        save_ticket(
            ticket_id=ticket_id,
            customer_name="CLI User",
            customer_email="cli@internal.system",
            ticket_text=args.text,
            predicted_category=res["predicted_category"],
            confidence=res["confidence"],
            priority=res["priority"],
            department=res["department"],
            status="Open"
        )
        res["saved_ticket_id"] = ticket_id

    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print("\n" + "=" * 60)
        print("🎯 AI TICKET CLASSIFICATION RESULT")
        print("=" * 60)
        print(f"📝 Input Text:       {args.text}")
        print(f"📂 Category:         {res['predicted_category']}")
        print(f"📊 Confidence:       {res['confidence_percentage']}%")
        print(f"⚡ Urgency Priority: {res['priority']}")
        print(f"🏢 Assigned Dept:    {res['department']}")
        print(f"📧 Dispatch Email:   {res['team_email']}")
        print(f"⏱️ Target SLA:       {res['sla_hours']} Hours")
        print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
