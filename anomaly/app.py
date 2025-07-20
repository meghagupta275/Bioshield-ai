from geopy.distance import geodesic
import geocoder
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

# --- TRANSACTION ANOMALY DETECTION SECTION ---
from transaction_anomaly import (
    init_db, add_transaction, fetch_user_transactions, is_transaction_anomalous
)

# Initialize the transaction database
init_db()

user_id = "user123"
new_transaction = {"amount": 20000000, "type": "rent"}  # 2 crore

# Fetch user's past transactions
user_transactions = fetch_user_transactions(user_id)

# Check for anomaly
is_anomaly, reason = is_transaction_anomalous(user_transactions, new_transaction)
print(f"[Transaction] Anomaly: {is_anomaly}, Reason: {reason}")

if not is_anomaly:
    add_transaction(user_id, new_transaction["amount"], new_transaction["type"])
    print("[Transaction] Transaction added.")
else:
    print("[Transaction] Transaction blocked due to anomaly.")


def is_gps_anomaly(user_history, new_location, threshold_km=50):
    for old_location in user_history:
        distance = geodesic(old_location, new_location).km
        if distance <= threshold_km:
            return False  # Not an anomaly
    return True  # Anomaly

# Get current location
g = geocoder.ip('me')
if not g.ok:
    print("Could not determine location.")
    exit()

current_location = tuple(g.latlng)
print("Your current location:", current_location)

# Example user history
user_history = [
    (40.7128, -74.0060),  # New York
    (40.7130, -74.0055),  # New York (nearby)
    (40.7127, -74.0059),  # New York (nearby)
   
]

# Prepare data for the model
X = np.array(user_history)

# Train Isolation Forest
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X)

# Save the model
joblib.dump(model, 'anomaly_model.joblib')
print("Model saved as anomaly_model.joblib")

# Predict anomaly for the current location (ML model)
current_location_np = np.array(current_location).reshape(1, -1)
prediction = model.predict(current_location_np)
is_ml_anomaly = prediction[0] == -1

# Distance-based anomaly detection (50km rule)
is_distance_anomaly = is_gps_anomaly(user_history, current_location, threshold_km=50)

if is_distance_anomaly:
    print("Anomaly detected by distance rule (>50km): True")
elif is_ml_anomaly:
    print("Anomaly detected by ML model: True")
else:
    print("Anomaly detected: False")