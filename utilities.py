import sqlite3
import bcrypt
from datetime import datetime
import streamlit as st

def get_db_connection():
    return sqlite3.connect('users.db')

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

def log_activity(user_id, action_type, details=None):
    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT INTO activity_logs (user_id, action_type, details) VALUES (?, ?, ?)",
            (user_id, action_type, details)
        )
        conn.commit()
    finally:
        conn.close()

def get_user_tier(username):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT tier FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        return result[0] if result else None
    finally:
        conn.close()
