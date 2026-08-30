"""
app/api/feedback.py
-------------------
Human-in-the-loop (HITL) endpoint for tracking false positives/negatives.
Connects to the same SQLite database used by LangGraph for seamless thread mapping.
"""
import os 
import sqlite3
from fastapi import APIRouter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.core.schemas import FeedbackPayload

router = APIRouter(prefix="/v1/feedback", tags=["Human-in-the-Loop"])

# Initialize the feedback table in the existing database
def init_feedback_db():
    conn = sqlite3.connect("chatbot.db", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS human_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT,
            feedback_value INTEGER,
            correction TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    return conn

db_conn = init_feedback_db()

@router.post("/")
def submit_feedback(payload: FeedbackPayload):
    """API Endpoint to log human feedback for model retraining/auditing."""
    cursor = db_conn.cursor()
    cursor.execute(
        "INSERT INTO human_feedback (thread_id, feedback_value, correction) VALUES (?, ?, ?)",
        (payload.thread_id, payload.feedback_value, payload.correction)
    )
    db_conn.commit()
    return {"status": "success", "message": "Feedback recorded for RLHF/Retraining."}

def save_feedback_sync(thread_id: str, feedback_value: int, correction: str = None):
    """Direct helper function for Streamlit to avoid network overhead."""
    cursor = db_conn.cursor()
    cursor.execute(
        "INSERT INTO human_feedback (thread_id, feedback_value, correction) VALUES (?, ?, ?)",
        (thread_id, feedback_value, correction)
    )
    db_conn.commit()
