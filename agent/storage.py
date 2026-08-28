import sqlite3
import os

DB_PATH = "agent_state.db"

def get_connection():
    """Establish a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the append-only event log and idempotency tables."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # R2: The append-only event log for durability
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            step INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # R2: Exactly-once guarantee for tools (like send_email)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tool_intents (
            intent_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            status TEXT NOT NULL, -- 'PENDING', 'COMPLETED', or 'FAILED'
            result_payload TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")