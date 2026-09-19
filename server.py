"""
Flask REST API Microservice for Customer Support Ticket Classifier.
Allows external services, bots, and support portals to classify tickets programmatically.
"""

import os
import sys
import json
from flask import Flask, request, jsonify

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.inference import TicketClassifier, ROUTING_CONFIG
from src.database import save_ticket, get_recent_tickets, update_ticket_status, get_ticket_stats
from src.preprocessor import DEFAULT_CATEGORIES

app = Flask(__name__, static_folder="public", static_url_path="")

# Initialize classifier
classifier = TicketClassifier(models_dir="models")

@app.route("/", methods=["GET"])
def home():
    index_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public", "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read(), 200, {"Content-Type": "text/html"}
    return jsonify({"service": "AI Customer Support Ticket Classifier API", "version": "1.0.0"})

@app.route("/api", methods=["GET"])
def api_info():
    return jsonify({
        "service": "AI Customer Support Ticket Classifier API",
        "version": "1.0.0",
        "model_architecture": "PyTorch BiLSTM with Multi-Head Self-Attention",
        "model_loaded": classifier.is_loaded,
        "supported_categories": DEFAULT_CATEGORIES,
        "endpoints": {
            "GET /api/health": "Healthcheck and model status",
            "POST /api/predict": "Classify a single support ticket",
            "POST /api/predict_batch": "Bulk ticket classification",
            "GET /api/stats": "Overall ticket operations statistics",
            "GET /api/history": "Retrieve logged tickets with filtering",
            "PATCH /api/tickets/<ticket_id>": "Update ticket status"
        }
    })

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": classifier.is_loaded,
        "device": str(classifier.device),
        "categories_count": len(DEFAULT_CATEGORIES)
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request body must be JSON"}), 400

    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Field 'text' is required and cannot be empty"}), 400

    if not classifier.is_loaded:
        return jsonify({"error": "Model weights are not loaded. Ensure models/ directory is populated."}), 503

    try:
        res = classifier.predict(text)
        
        # Optionally persist ticket
        if data.get("save", True):
            import uuid
            ticket_id = data.get("ticket_id") or f"TCK-{str(uuid.uuid4())[:8].upper()}"
            customer_name = data.get("customer_name", "Web API User")
            customer_email = data.get("customer_email", "api@customer.support")
            
            save_ticket(
                ticket_id=ticket_id,
                customer_name=customer_name,
                customer_email=customer_email,
                ticket_text=text,
                predicted_category=res["predicted_category"],
                confidence=res["confidence"],
                priority=res["priority"],
                department=res["department"],
                status="Open"
            )
            res["ticket_id"] = ticket_id

        return jsonify({
            "success": True,
            "data": res
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict_batch", methods=["POST"])
def predict_batch():
    if not request.is_json:
        return jsonify({"error": "Request body must be JSON"}), 400

    data = request.get_json()
    tickets = data.get("tickets", [])
    if not isinstance(tickets, list) or len(tickets) == 0:
        return jsonify({"error": "Field 'tickets' must be a non-empty list of ticket texts"}), 400

    if not classifier.is_loaded:
        return jsonify({"error": "Model weights are not loaded."}), 503

    try:
        results = classifier.predict_batch(tickets)
        return jsonify({
            "success": True,
            "count": len(results),
            "data": results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/stats", methods=["GET"])
def stats():
    try:
        db_stats = get_ticket_stats()
        return jsonify({
            "success": True,
            "data": db_stats
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/history", methods=["GET"])
def history():
    try:
        limit = int(request.args.get("limit", 50))
        category = request.args.get("category", None)
        priority = request.args.get("priority", None)
        status = request.args.get("status", None)
        
        tickets = get_recent_tickets(limit=limit, category=category, priority=priority, status=status)
        return jsonify({
            "success": True,
            "count": len(tickets),
            "data": tickets
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/tickets/<ticket_id>", methods=["PATCH"])
def update_status(ticket_id):
    if not request.is_json:
        return jsonify({"error": "Request body must be JSON"}), 400
        
    data = request.get_json()
    new_status = data.get("status")
    if not new_status or new_status not in ["Open", "In Progress", "Resolved", "Pending Customer"]:
        return jsonify({"error": "Field 'status' must be one of: Open, In Progress, Resolved, Pending Customer"}), 400

    success = update_ticket_status(ticket_id, new_status)
    if success:
        return jsonify({"success": True, "ticket_id": ticket_id, "updated_status": new_status})
    return jsonify({"error": f"Failed to update ticket {ticket_id}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"Starting Flask REST API server on http://localhost:{port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
