import sqlite3
from datetime import datetime

DB_PATH = "banking_auth.db"  # Update if your DB path is different

def create_location_table():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_locations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def add_user_location(user_id, latitude, longitude):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO user_locations (user_id, latitude, longitude, timestamp) VALUES (?, ?, ?, ?)",
        (user_id, latitude, longitude, datetime.now())
    )
    conn.commit()
    conn.close()

def fetch_user_locations(user_id, limit=10):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT latitude, longitude FROM user_locations WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
        (user_id, limit)
    )
    rows = c.fetchall()
    conn.close()
    return [(row[0], row[1]) for row in rows]

create_location_table() 