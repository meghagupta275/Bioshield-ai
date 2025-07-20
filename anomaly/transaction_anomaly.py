import sqlite3
from datetime import datetime

# Database setup
DB_NAME = 'transactions.db'

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            amount REAL NOT NULL,
            type TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def add_transaction(user_id, amount, type_):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    c.execute('INSERT INTO transactions (user_id, amount, type, timestamp) VALUES (?, ?, ?, ?)',
              (user_id, amount, type_, timestamp))
    conn.commit()
    conn.close()

def fetch_user_transactions(user_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT amount, type, timestamp FROM transactions WHERE user_id = ?', (user_id,))
    rows = c.fetchall()
    conn.close()
    return rows

def is_transaction_anomalous(user_transactions, new_transaction, max_amount=100000, max_repeats=10):
    # Block if amount is too high (now 100K)
    if new_transaction["amount"] > max_amount:
        return True, f"Amount {new_transaction['amount']:,} exceeds threshold of {max_amount:,}"
    
    # Block if repeated too many times (now 10)
    count = sum(
        t[0] == new_transaction["amount"] and t[1] == new_transaction["type"]
        for t in user_transactions
    )
    if count >= max_repeats:
        return True, "Repeated transaction detected"
    
    # Remove/block the 50000 threshold for additional security
    # (No additional block for >50000, only >100000)
    
    return False, "No anomaly"

# --- DEMO USAGE ---
if __name__ == "__main__":
    init_db()
    user_id = "user123"
    new_transaction = {"amount": 20000000, "type": "rent"}  # 2 crore

    # Fetch user's past transactions
    user_transactions = fetch_user_transactions(user_id)

    # Check for anomaly
    is_anomaly, reason = is_transaction_anomalous(user_transactions, new_transaction)
    print(f"Anomaly: {is_anomaly}, Reason: {reason}")

    if not is_anomaly:
        add_transaction(user_id, new_transaction["amount"], new_transaction["type"])
        print("Transaction added.")
    else:
        print("Transaction blocked due to anomaly.") 