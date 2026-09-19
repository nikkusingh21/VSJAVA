"""
Database module for storing and querying customer support ticket history.
Uses SQLite for robust, zero-configuration local persistence.
"""

import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tickets.db")

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ticket_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id TEXT UNIQUE,
            customer_name TEXT,
            customer_email TEXT,
            ticket_text TEXT,
            predicted_category TEXT,
            confidence REAL,
            priority TEXT,
            department TEXT,
            status TEXT DEFAULT 'Open',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_ticket(
    ticket_id: str,
    customer_name: str,
    customer_email: str,
    ticket_text: str,
    predicted_category: str,
    confidence: float,
    priority: str,
    department: str,
    status: str = "Open"
) -> bool:
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO ticket_history 
            (ticket_id, customer_name, customer_email, ticket_text, predicted_category, confidence, priority, department, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ticket_id,
            customer_name,
            customer_email,
            ticket_text,
            predicted_category,
            round(confidence, 4),
            priority,
            department,
            status,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving ticket {ticket_id}: {e}")
        return False

def get_recent_tickets(
    limit: int = 200,
    category: Optional[str] = None,
    priority: Optional[str] = None,
    status: Optional[str] = None
) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    query = "SELECT * FROM ticket_history WHERE 1=1"
    params = []
    
    if category and category != "All":
        query += " AND predicted_category = ?"
        params.append(category)
    if priority and priority != "All":
        query += " AND priority = ?"
        params.append(priority)
    if status and status != "All":
        query += " AND status = ?"
        params.append(status)
        
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def update_ticket_status(ticket_id: str, new_status: str) -> bool:
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE ticket_history SET status = ? WHERE ticket_id = ?", (new_status, ticket_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error updating status for {ticket_id}: {e}")
        return False

def get_ticket_stats() -> Dict[str, Any]:
    conn = get_connection()
    cursor = conn.cursor()
    
    # Total count
    cursor.execute("SELECT COUNT(*), AVG(confidence) FROM ticket_history")
    row = cursor.fetchone()
    total_tickets = row[0] or 0
    avg_confidence = round(row[1] or 0.0, 3)
    
    # Counts by category
    cursor.execute("SELECT predicted_category, COUNT(*) as cnt FROM ticket_history GROUP BY predicted_category")
    category_counts = {r["predicted_category"]: r["cnt"] for r in cursor.fetchall()}
    
    # Counts by priority
    cursor.execute("SELECT priority, COUNT(*) as cnt FROM ticket_history GROUP BY priority")
    priority_counts = {r["priority"]: r["cnt"] for r in cursor.fetchall()}
    
    # Counts by status
    cursor.execute("SELECT status, COUNT(*) as cnt FROM ticket_history GROUP BY status")
    status_counts = {r["status"]: r["cnt"] for r in cursor.fetchall()}
    
    conn.close()
    return {
        "total_tickets": total_tickets,
        "avg_confidence": avg_confidence,
        "category_counts": category_counts,
        "priority_counts": priority_counts,
        "status_counts": status_counts
    }

# Auto-initialize on import
init_db()
