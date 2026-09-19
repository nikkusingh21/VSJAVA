"""
Automated tests for Flask REST API microservice.
"""
from server import app

def test_api_endpoints():
    client = app.test_client()
    
    # 1. Health check
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    print("✓ GET /api/health passed")
    
    # 2. Prediction
    test_ticket = "I was charged twice on my credit card for order #55129"
    res = client.post("/api/predict", json={"text": test_ticket, "save": False})
    assert res.status_code == 200
    pdata = res.get_json()["data"]
    assert pdata["predicted_category"] == "Payment Issue"
    assert pdata["confidence"] > 0.8
    print(f"✓ POST /api/predict passed -> {pdata['predicted_category']} ({pdata['confidence_percentage']}%)")
    
    # 3. Batch prediction
    res = client.post("/api/predict_batch", json={
        "tickets": [
            "Cannot log into my portal, 2FA not working",
            "Where is my package? Tracking is delayed"
        ]
    })
    assert res.status_code == 200
    bdata = res.get_json()
    assert bdata["count"] == 2
    assert bdata["data"][0]["predicted_category"] == "Login Problem"
    assert bdata["data"][1]["predicted_category"] == "Order Status"
    print("✓ POST /api/predict_batch passed")
    
    # 4. Stats
    res = client.get("/api/stats")
    assert res.status_code == 200
    sdata = res.get_json()
    assert "total_tickets" in sdata["data"]
    print(f"✓ GET /api/stats passed -> Total Tickets: {sdata['data']['total_tickets']}")
    
    print("\nALL API ENDPOINT TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_api_endpoints()
