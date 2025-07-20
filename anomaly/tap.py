import time
import math
from datetime import datetime
import numpy as np
from sklearn.ensemble import IsolationForest

# Store legitimate users (features)
legitimate_users = {}
# Store bot detection logs
detection_logs = []
# ML model (global)
ml_model = None

def get_tap_timestamps():
    """Get tap timestamps from user input."""
    timestamps = []
    print("\n️  HUMAN VERIFICATION TEST")
    print("Tap naturally (like a human would)...")
    print("Enter timestamps for each tap (e.g., 0.0, 0.8, 1.5, 2.3)")
    print("Type 'done' when finished\n")
    
    tap_count = 1
    while True:
        try:
            user_input = input(f"Tap {tap_count} timestamp: ")
            if user_input.lower() == 'done':
                break
            timestamp = float(user_input)
            timestamps.append(timestamp)
            tap_count += 1
        except ValueError:
            print("Invalid input. Please enter a number or 'done'.")
    
    return timestamps

def extract_features(timestamps):
    """Extract features from tap timestamps for ML model."""
    if len(timestamps) < 3:
        return None
    intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
    avg_interval = sum(intervals) / len(intervals)
    min_interval = min(intervals)
    max_interval = max(intervals)
    variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
    tap_speed = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0]) if timestamps[-1] > timestamps[0] else 0
    return [avg_interval, min_interval, max_interval, variance, tap_speed]

def train_ml_model():
    """Train the ML model on all registered human user patterns."""
    global ml_model
    if not legitimate_users:
        ml_model = None
        return
    X = np.array(list(legitimate_users.values()))
    ml_model = IsolationForest(contamination='auto', random_state=42)
    ml_model.fit(X)

def ml_predict(features):
    """Use the ML model to predict if the pattern is an anomaly (bot)."""
    if ml_model is None:
        return False, "ML model not trained. Assuming human."
    X = np.array(features).reshape(1, -1)
    prediction = ml_model.predict(X)
    if prediction[0] == -1:
        return True, "ML model: Anomaly detected (bot-like pattern)"
    else:
        return False, "ML model: Human-like pattern"

def verify_user_or_bot():
    """Main verification function (ML enhanced)."""
    print("\n" + "="*50)
    print("    🤖 BOT DETECTION SYSTEM (ML Enhanced)")
    print("="*50)
    
    timestamps = get_tap_timestamps()
    if len(timestamps) < 3:
        print("❌ Need at least 3 taps for verification.")
        return
    
    features = extract_features(timestamps)
    if not features:
        print("❌ Failed to extract features.")
        return
    
    is_bot_user, reason = ml_predict(features)
    
    print(f"\n=== VERIFICATION RESULTS ===")
    print(f"Features: {features}")
    
    if is_bot_user:
        print(f"\n🚨 BOT DETECTED!")
        print(f"Reason: {reason}")
        print("\nAction: BLOCKED")
        detection_logs.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'BOT',
            'features': features,
            'reason': reason
        })
        return False
    else:
        print(f"\n✅ HUMAN VERIFIED!")
        print(f"Reason: {reason}")
        print("\nAction: ALLOWED")
        detection_logs.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'HUMAN',
            'features': features,
            'reason': reason
        })
        return True

def register_human_user():
    """Register a human user."""
    user_id = input("Enter user ID: ")
    print(f"\n🔐 Registering human user: {user_id}")
    
    timestamps = get_tap_timestamps()
    if len(timestamps) < 3:
        print("❌ Need at least 3 taps for registration.")
        return
    
    features = extract_features(timestamps)
    if not features:
        print("❌ Failed to extract features.")
        return
    
    legitimate_users[user_id] = features
    train_ml_model()
    print(f"\n✅ Human user {user_id} registered successfully!")
    print("Pattern saved for future verification.")

def verify_registered_user():
    """Verify against registered user pattern (classic logic)."""
    user_id = input("Enter user ID to verify: ")
    
    if user_id not in legitimate_users:
        print(f"❌ User {user_id} not found.")
        return
    
    print(f"\n🔍 Verifying user: {user_id}")
    
    timestamps = get_tap_timestamps()
    if len(timestamps) < 3:
        print("❌ Need at least 3 taps for verification.")
        return
    
    current_features = extract_features(timestamps)
    if not current_features:
        print("❌ Failed to extract current features.")
        return
    
    stored_features = legitimate_users[user_id]
    interval_diff = abs(current_features[0] - stored_features[0])
    speed_diff = abs(current_features[4] - stored_features[4])
    similarity = max(0, 1 - (interval_diff + speed_diff) / 2)
    print(f"\n=== USER VERIFICATION RESULTS ===")
    print(f"Similarity Score: {similarity:.3f}")
    if similarity >= 0.6:
        print("✅ USER VERIFIED: Pattern matches registered user")
        print("Action: ALLOWED")
    else:
        print("❌ USER VERIFICATION FAILED: Pattern doesn't match")
        print("Action: BLOCKED")

def view_detection_logs():
    """View all detection logs."""
    if not detection_logs:
        print("No detection logs yet.")
        return
    print(f"\n=== DETECTION LOGS ({len(detection_logs)} entries) ===")
    for i, log in enumerate(detection_logs, 1):
        print(f"\n{i}. Time: {log['timestamp']}")
        print(f"   Type: {log['type']}")
        print(f"   Features: {log['features']}")
        print(f"   Reason: {log['reason']}")

def menu():
    """Main menu."""
    while True:
        print("\n" + "="*50)
        print("    🤖 BOT DETECTION SYSTEM (ML Enhanced)")
        print("="*50)
        print("1. Verify User or Bot (ML)")
        print("2. Register Human User")
        print("3. Verify Registered User (Classic)")
        print("4. View Detection Logs")
        print("5. Exit")
        
        choice = input("\nChoose an option (1-5): ")
        
        if choice == "1":
            verify_user_or_bot()
        elif choice == "2":
            register_human_user()
        elif choice == "3":
            verify_registered_user()
        elif choice == "4":
            view_detection_logs()
        elif choice == "5":
            print("Goodbye! Stay safe from bots! 🤖")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    print("🤖 Welcome to Bot Detection System (ML Enhanced)!")
    print("This system verifies if you're human or a bot using machine learning.")
    menu()