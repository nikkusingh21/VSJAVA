"""
Unit test and verification script for TicketClassifier.
"""
from src.inference import TicketClassifier

def run_verification():
    clf = TicketClassifier()
    assert clf.is_loaded, "Model failed to load!"
    
    test_cases = [
        ("I was charged twice on my credit card for order #123456", "Payment Issue"),
        ("Password reset email never arrived and I am locked out of my account", "Login Problem"),
        ("Where is my package? The tracking link says in transit for 10 days", "Order Status"),
        ("I want my money back for the defective item I returned last week", "Refund Request"),
        ("The desktop app crashes with segmentation fault error code 500", "Technical Support"),
        ("How do I change the registered email and shipping address on my account profile?", "Account Issue"),
        ("The screen has scratches and dead pixels right out of the box", "Product Complaint"),
        ("Courier left the parcel in the rain and marked it delivered but no one signed", "Delivery Issue")
    ]
    
    print("=" * 70)
    print("AI CUSTOMER SUPPORT TICKET CLASSIFIER - VERIFICATION TEST")
    print("=" * 70)
    
    all_passed = True
    for text, expected in test_cases:
        res = clf.predict(text)
        cat = res["predicted_category"]
        conf = res["confidence_percentage"]
        prio = res["priority"]
        dept = res["department"]
        passed = (cat == expected)
        status = "PASSED" if passed else "FAILED"
        if not passed:
            all_passed = False
            
        print(f"[{status}] Query: '{text[:45]}...'")
        print(f"         Expected: {expected} | Predicted: {cat} (Confidence: {conf}%)")
        print(f"         Urgency: {prio} | Routing: {dept}")
        print("-" * 70)
        
    print(f"\nFinal Result: {'ALL TESTS PASSED SUCCESSFULLY!' if all_passed else 'SOME TESTS FAILED'}")

if __name__ == "__main__":
    run_verification()
