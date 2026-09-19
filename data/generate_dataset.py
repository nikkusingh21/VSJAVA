"""
Dataset Generator for Customer Support Ticket Classifier
Generates a realistic, diverse Kaggle-style customer support dataset with 8 predefined categories:
- Payment Issue
- Login Problem
- Order Status
- Refund Request
- Technical Support
- Account Issue
- Product Complaint
- Delivery Issue
"""

import os
import random
import pandas as pd

CATEGORIES = [
    "Payment Issue",
    "Login Problem",
    "Order Status",
    "Refund Request",
    "Technical Support",
    "Account Issue",
    "Product Complaint",
    "Delivery Issue"
]

FIRST_NAMES = ["Alex", "Jordan", "Taylor", "Morgan", "Sam", "Chris", "Pat", "Riley", "Casey", "Avery", "David", "Emma", "Michael", "Sophia", "Daniel", "Olivia", "James", "Isabella", "William", "Mia", "Ethan", "Charlotte", "Alexander", "Amelia", "Nikhil", "Priya", "Rahul", "Ananya", "Aarav", "Neha"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Sharma", "Verma", "Patel", "Gupta", "Singh", "Chen", "Wang", "Kim", "Lee", "Tanaka"]

PRIORITIES = ["Low", "Medium", "High", "Critical"]
STATUSES = ["Open", "In Progress", "Resolved", "Pending Customer"]
CHANNELS = ["Email", "Web Portal", "Live Chat", "Mobile App"]

# Rich domain-specific text templates with variations for each category
TEMPLATES = {
    "Payment Issue": [
        "I was charged twice on my credit card for order #{order_id}. Please check your billing system and reverse the duplicate {amount} charge immediately.",
        "My payment failed during checkout using my {card_type} card, but money got deducted from my bank account. Order ID: #{order_id}.",
        "I received an unexpected transaction alert of {amount} from your company that I did not authorize. Please investigate this billing error.",
        "Why is my debit card being declined at the payment screen? There are sufficient funds available and my bank says the merchant gateway is rejecting it.",
        "Attempting to pay via PayPal gives error code PAY_9021. The payment processing loop just hangs without confirmation.",
        "Payment went through and I have bank reference #{ref_id}, but the cart still says 'Awaiting Payment' and did not issue an order receipt.",
        "You charged me an extra {amount} processing fee which was not shown on the checkout summary page. I need this rectified.",
        "My Apple Pay transaction timed out at final review, yet my credit limit shows the hold. Did the payment clear or should I retry?",
        "Automatic renewal charged my expired card somehow and gave an overdue penalty of {amount}. Please fix my payment method.",
        "I cannot add my new credit card in the billing settings. It shows 'Tokenization Failed - invalid CVV' even though CVV is completely correct.",
        "The installment payment option failed to split my bill, and you deducted the full amount of {amount} at once. Please correct the installment plan.",
        "Every time I try to complete payment using Google Pay, the checkout page refreshes and clears my shopping bag without placing the order.",
        "Bank statement shows two identical deductions of {amount} dated yesterday for the same order #{order_id}. Kindly refund the extra deduction.",
        "Payment verification OTP is never delivered to my registered phone number during 3D Secure authentication. Payment times out every time.",
        "I used a promo coupon code offering $30 discount, but my final invoice charged the full price without the promotional deduction."
    ],
    "Login Problem": [
        "I cannot log into my account. When I submit my password, the screen displays 'Invalid Credentials' even though I just reset it.",
        "Two-factor authentication code (2FA) is not arriving on my phone number. I have been locked out of my portal for 4 hours.",
        "I clicked 'Forgot Password' multiple times, but the password reset link email never arrives in my inbox or spam folder.",
        "My account says 'Account Temporarily Locked due to too many failed attempts'. How do I unlock it? I need access for my work urgently.",
        "The mobile app keeps logging me out every 2 minutes. When I attempt to re-login, the login button spins continuously without responding.",
        "Single Sign-On (SSO) via Google returns an error 'OAuth2 callback failed: 403 Forbidden'. Unable to access my dashboard.",
        "The CAPTCHA on the login page does not load, leaving a blank square. Because of this, the login button remains disabled.",
        "I changed my registered email address last week, and now neither the old nor the new email is recognized by the login system.",
        "I get 'Session expired - please login again' immediately after entering my correct username and password. Infinite loop bug.",
        "Biometric Face ID login failed on my iOS device and now asking for a master PIN that I never configured during initial setup.",
        "Unable to sign in on Chrome desktop browser. Clearing cookies and cache did not help, getting 'Error 401 Unauthorized'.",
        "My team member left the company and we need to revoke their login access and transfer the admin account ownership to my email.",
        "Password requirements page won't accept my 16-character secure passphrase, stating it contains illegal characters.",
        "Verification code sent to authenticator app (Google Authenticator) says 'Code expired' every time I type it in.",
        "When I try to login from my home office IP address, it flags 'Suspicious login detected' and blocks my access completely."
    ],
    "Order Status": [
        "Can you please provide the latest tracking status for my order #{order_id}? Placed it 6 days ago and haven't received any dispatch email.",
        "Order #{order_id} has been stuck on 'Processing in Warehouse' for over a week. When will it actually ship out?",
        "I never received an order confirmation email with my tracking number for the purchase made yesterday evening. Card was charged.",
        "The tracking link provided in your email opens a blank page with error 'Tracking number not recognized by carrier'. Order #{order_id}.",
        "Could you confirm if order #{order_id} has been dispatched? The estimated arrival date was today, but no courier updates yet.",
        "I need to check the delivery timeline for purchase #{order_id}. Will it reach before Friday as I am traveling out of state?",
        "Order status shows 'In Transit' for the last 10 days with no location scans since leaving the regional sorting hub.",
        "Where is my package for order #{order_id}? The courier tracking shows 'Manifest Created' but carrier has not picked it up.",
        "I ordered three items under order #{order_id}, but received only one small parcel today. Are the remaining two items shipping separately?",
        "Is there any update on the backordered item in order #{order_id}? It has been 2 weeks since the expected restock date.",
        "Can you send me the invoice and live GPS tracking link for order #{order_id} placed under my business account?",
        "My order status abruptly changed from 'Shipped' back to 'On Hold'. What is the reason for this hold?",
        "Checking if order #{order_id} can still be expedited to express next-day delivery? I am willing to pay the difference.",
        "The tracking dashboard says 'Exception: Delay due to weather conditions'. How many days of delay should I anticipate?",
        "Order #{order_id} is shown as 'Preparing for dispatch' since Monday. Customer care bot is not providing meaningful updates."
    ],
    "Refund Request": [
        "I returned the damaged jacket 10 days ago (return tracking #{ref_id}). When will my refund of {amount} be processed back to my card?",
        "I want my money back for the defective item I returned last week. Please process the credit immediately.",
        "I cancelled order #{order_id} within 30 minutes of placing it. Please confirm when the refund will reflect in my bank account.",
        "You approved my return request last Tuesday, but I have not received the credit note or refund payment confirmation yet.",
        "I was promised a full refund of {amount} due to defective parts, but you only credited a partial amount of $25. Please refund the remainder.",
        "The return courier picked up the parcel from my doorstep on Monday. Kindly initiate the refund to my original payment source.",
        "Please cancel my annual subscription renewal and issue a full refund as per your 30-day money-back guarantee policy.",
        "I was charged for a cancelled order #{order_id}. Please expedite the refund process as this was an error on your store's end.",
        "How long does it take for a refund to show on a Visa debit card? Your support agent said 3-5 business days, but today is day 9.",
        "I want to return this unopened item and get a direct refund rather than store credit vouchers. Please advise on the return label.",
        "Refund status says 'Completed' in my portal, but my bank statement shows zero incoming transfers. Please share the ARN / bank reference number.",
        "The flight ticket was cancelled due to schedule changes by your airline partner. I demand a prompt 100% cash refund.",
        "Received defective electronic headphones. I don't want a replacement, I strictly request a full refund to my original card.",
        "Returned package was delivered to your warehouse according to FedEx tracking #{ref_id}. When will inspection complete and refund issue?",
        "Overcharged by {amount} due to recurring billing glitch after account cancellation. Please reverse and refund this immediately.",
        "Requested a refund 2 weeks ago for order #{order_id}. Still waiting. If not processed today, I will have to file a credit card chargeback.",
        "I am asking for my money back because the service was not provided. Please issue reimbursement.",
        "Sent the package back via postal return. Where is my refund? It has been over two weeks.",
        "Can I get a refund back to my original payment method? I have already shipped the return item."
    ],
    "Technical Support": [
        "The desktop application crashes immediately upon opening on macOS Sonoma. Error log mentions 'Segmentation fault at 0x000003'.",
        "Getting HTTP 500 Internal Server Error whenever I click on the 'Export PDF Report' button in the dashboard analytics module.",
        "The REST API returns 429 Too Many Requests even though our traffic is well under the 60 requests/minute tier quota limit.",
        "Database sync is failing between the mobile client and the cloud server. Offline edits are completely lost upon reconnecting.",
        "Our developers are unable to integrate the Webhook endpoint. Incoming POST payloads are missing the signature verification header.",
        "The browser extension freezes Chrome tabs when scrolling through large document lists. Need an updated patch or workaround.",
        "Audio output does not work during video conferences on the web app. Microphone input is detected, but remote audio is silent.",
        "File upload feature fails for files larger than 15MB with 'Payload Too Large' error, despite our plan allowing up to 100MB uploads.",
        "SSL certificate on your webhook callback domain appears to have expired yesterday, causing all our automated background jobs to fail.",
        "Search functionality in the app is broken; typing any search query returns an unhandled React error in the browser console.",
        "The printer driver integration fails to render barcodes correctly, printing fuzzy unreadable lines instead of 2D data matrix.",
        "After the latest firmware v2.4.1 update, the smart hub disconnects from Wi-Fi every 15 minutes and requires a manual power cycle.",
        "WebSocket connection drops repeatedly with code 1006. Reconnection attempts cause UI lag and high CPU consumption.",
        "Exported CSV files have corrupted text formatting and broken UTF-8 encoding for international non-English characters.",
        "Mobile app push notifications are completely failing on Android 14 devices. Permissions are enabled in system settings."
    ],
    "Account Issue": [
        "I need to change my primary registered email address from old@example.com to new@example.com because I lost access to the old email provider.",
        "Please help me update the legal company name and tax ID (VAT number) on our corporate subscription account.",
        "I would like to permanently delete my account and erase all associated personal data in accordance with GDPR regulations.",
        "How can I merge two accounts created under different email addresses so all my purchased licenses are under one master profile?",
        "I upgraded my plan from Basic to Pro, but my account dashboard is still displaying Basic tier limits and restrictions.",
        "Unable to modify my shipping address in saved profile settings. The 'Save Address' button stays greyed out.",
        "Need to remove a former administrator from our organization workspace and reassign all project ownership to my user ID.",
        "My account profile picture and username will not update. It says 'Profile updated successfully', but old details remain.",
        "We are receiving marketing emails and promotional newsletters even though all communication preferences are disabled in my profile.",
        "Please transfer ownership of our company team account to our new operations director email address.",
        "I want to change my account phone number for security verification, but the system prompts me to verify via the old number which is deactivated.",
        "How do I download a complete archive copy of my account data, purchase history, and uploaded invoices for tax auditing?",
        "My student discount verification was approved by SheerID, but my subscription billing still shows standard commercial price.",
        "I accidentally created two duplicate profiles for the same customer. Can your support team merge their loyalty points?",
        "Our team seat count is capped at 5, but our invoice shows 10 active licenses. Please audit our user seats and adjust the billing tier."
    ],
    "Product Complaint": [
        "The smartphone display screen arrived with multiple visible scratches and a dead pixel in the center. Very poor quality control.",
        "The material of this winter coat is extremely cheap and thin, completely different from what was advertised in your product photos.",
        "Missing essential accessories: the package did not include the power charging adapter and USB cable specified on the box.",
        "The blender motor started smoking and smelled like burning plastic on its very first usage at normal speed setting. Dangerous hazard.",
        "You sent me the wrong color! I ordered Midnight Blue, but you shipped bright yellow. I need the correct item sent ASAP.",
        "The wireless earbuds have terrible battery life, dying in less than 45 minutes instead of the advertised 8 hours of playback.",
        "The zipper on this leather backpack broke off on day two of use. This is unacceptable for a luxury price product.",
        "Product description claimed this monitor has built-in speakers and HDR support, but the unit received lacks both features entirely.",
        "The size chart provided on your website is completely inaccurate. The XL shirt fits like a small medium. Requesting an exchange.",
        "One of the dining table legs is two inches shorter than the others, making the table wobble violently. Defective manufacturing.",
        "The software license key printed on the inside card says 'Already Redeemed by another user'. I bought this brand new sealed.",
        "The camera lens has internal dust particles sealed beneath the glass element which blurs every photo taken.",
        "Fragile glass vase was shipped without adequate bubble wrap or protective padding, arriving completely shattered into pieces.",
        "The non-stick frying pan coating began peeling and flaking into food after just three gentle hand-wash cleanings.",
        "Shoes look like they were previously worn; the soles are scuffed and dirty right out of the sealed shipping box."
    ],
    "Delivery Issue": [
        "Courier marked my package as 'Delivered to resident', but no one was home and there is no package on the porch or with neighbors.",
        "My delivery is delayed by over 6 business days with zero explanations from the courier service. Placed order #{order_id}.",
        "The delivery driver literally threw the heavy parcel over my high front fence, denting the outer shipping box.",
        "The courier attempted delivery at 6:30 AM without ringing the doorbell, left a 'Failed Attempt' slip, and returned it to the depot.",
        "Delivery address was updated to my new apartment, but the package was dispatched to my old address in another city.",
        "Package is held at customs clearance depot for 8 days because the shipping invoice documentation was missing from the parcel.",
        "The courier driver refused to bring the heavy 45kg parcel up to my apartment floor, leaving it on the street sidewalk.",
        "The delivery status has said 'Out for Delivery' for three consecutive days from 8 AM to 9 PM, but driver never shows up.",
        "Courier says delivery address is 'Incomplete / Missing Apt Number' even though my order confirmation shows complete details.",
        "Requested weekend delivery when placing order #{order_id}, but carrier attempted delivery on Wednesday during work hours.",
        "The parcel was left out in heavy pouring rain on my uncovered driveway instead of under the covered front porch.",
        "My parcel was sent to the wrong delivery hub across the country due to a barcode sorting misroute. Need urgent reroute.",
        "Signature was required for this high-value laptop delivery, yet the carrier signed on my behalf with fake initials and dumped it.",
        "Estimated delivery date has changed 4 times this week. First Monday, then Wednesday, now next week. Highly unreliable service.",
        "The courier tracking shows 'Returned to Sender - Recipient Not Found' without making any delivery attempt or phone call."
    ]
}

def generate_dataset(num_samples_per_cat=200):
    records = []
    ticket_counter = 10001
    
    for category, templates in TEMPLATES.items():
        for i in range(num_samples_per_cat):
            # Choose a template
            tpl = random.choice(templates)
            
            # Fill dynamic slots
            order_id = random.randint(100000, 999999)
            ref_id = f"REF-{random.randint(10000, 99999)}"
            amount = f"${random.randint(15, 850)}.{random.choice(['00', '50', '99', '25', '75'])}"
            card_type = random.choice(["Visa", "Mastercard", "Amex", "Discover", "Debit"])
            
            text = tpl.format(order_id=order_id, ref_id=ref_id, amount=amount, card_type=card_type)
            
            # Add slight realistic noise / variations
            prefixes = [
                "", "Hello, ", "Hi Support Team, ", "Urgent: ", "Dear Customer Service, ",
                "Good day, ", "Hey, ", "Help needed: ", "To whom it may concern: "
            ]
            suffixes = [
                "", " Thanks.", " Please help me resolve this ASAP.", " Looking forward to your prompt response.",
                " This is very frustrating.", " Kindly advise.", " Appreciate your quick assistance.",
                " Let me know how to proceed.", " Thank you!"
            ]
            
            pfx = random.choice(prefixes)
            sfx = random.choice(suffixes)
            full_text = f"{pfx}{text}{sfx}".strip()
            
            # Determine urgency based on keywords or category
            if any(w in full_text.lower() for w in ["urgent", "immediately", "hazard", "stolen", "unauthorized", "chargeback", "asap", "locked out"]):
                priority = random.choice(["High", "Critical"])
            elif any(w in full_text.lower() for w in ["frustrating", "delayed", "broken", "crashing", "twice", "failed", "smoking"]):
                priority = random.choice(["Medium", "High"])
            else:
                priority = random.choice(["Low", "Medium"])
                
            first = random.choice(FIRST_NAMES)
            last = random.choice(LAST_NAMES)
            customer_name = f"{first} {last}"
            customer_email = f"{first.lower()}.{last.lower()}{random.randint(10, 99)}@example.com"
            
            channel = random.choice(CHANNELS)
            status = random.choice(STATUSES)
            
            records.append({
                "ticket_id": f"TCK-{ticket_counter}",
                "customer_name": customer_name,
                "customer_email": customer_email,
                "ticket_type": category,
                "ticket_text": full_text,
                "priority": priority,
                "status": status,
                "channel": channel
            })
            ticket_counter += 1
            
    random.shuffle(records)
    df = pd.DataFrame(records)
    return df

def main():
    os.makedirs("/Users/nikusingh/Desktop/DLNLP project/data", exist_ok=True)
    os.makedirs("/Users/nikusingh/Desktop/DLNLP project/models", exist_ok=True)
    os.makedirs("/Users/nikusingh/Desktop/DLNLP project/src", exist_ok=True)
    
    print("Generating training & evaluation dataset (1,600 realistic tickets across 8 classes)...")
    df = generate_dataset(num_samples_per_cat=200)
    csv_path = "/Users/nikusingh/Desktop/DLNLP project/data/customer_support_tickets.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved dataset with {len(df)} records to {csv_path}")
    print("Category breakdown:\n", df['ticket_type'].value_counts())
    
    # Also generate a smaller sample batch file for demoing batch classification
    sample_batch = df.sample(25, random_state=42)[["ticket_id", "customer_name", "ticket_text"]].copy()
    sample_path = "/Users/nikusingh/Desktop/DLNLP project/data/sample_batch_tickets.csv"
    sample_batch.to_csv(sample_path, index=False)
    print(f"Saved sample batch file with {len(sample_batch)} records to {sample_path}")

if __name__ == "__main__":
    main()
