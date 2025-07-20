import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime
import geocoder

# Store user location history
user_history = []
# Store anomaly detection logs
anomaly_logs = []


def add_location():
    print("\nAdd a new location:")
    print("1. Use current IP location")
    print("2. Enter latitude and longitude manually")
    choice = input("Choose an option (1-2): ")
    if choice == "1":
        g = geocoder.ip('me')
        if not g.ok:
            print("Could not determine location.")
            return
        location = tuple(g.latlng)
        print(f"Current location: {location}")
    elif choice == "2":
        try:
            lat = float(input("Enter latitude: "))
            lng = float(input("Enter longitude: "))
            location = (lat, lng)
        except ValueError:
            print("Invalid input.")
            return
    else:
        print("Invalid choice.")
        return
    user_history.append(location)
    print(f"Location {location} added to history.")


def check_anomaly():
    if len(user_history) < 3:
        print("\nNeed at least 3 locations in history for anomaly detection.")
        return
    print("\nCheck if a location is an anomaly:")
    print("1. Use current IP location")
    print("2. Enter latitude and longitude manually")
    choice = input("Choose an option (1-2): ")
    if choice == "1":
        g = geocoder.ip('me')
        if not g.ok:
            print("Could not determine location.")
            return
        location = tuple(g.latlng)
        print(f"Current location: {location}")
    elif choice == "2":
        try:
            lat = float(input("Enter latitude: "))
            lng = float(input("Enter longitude: "))
            location = (lat, lng)
        except ValueError:
            print("Invalid input.")
            return
    else:
        print("Invalid choice.")
        return
    # Prepare data for the model
    X = np.array(user_history)
    # Use 'auto' for contamination to avoid linter error
    model = IsolationForest(contamination='auto', random_state=42)
    model.fit(X)
    location_np = np.array(location).reshape(1, -1)
    prediction = model.predict(location_np)
    is_anomaly = prediction[0] == -1
    print(f"\nAnomaly detected: {is_anomaly}")
    anomaly_logs.append({
        'timestamp': datetime.now().isoformat(),
        'location': location,
        'is_anomaly': is_anomaly
    })


def view_logs():
    if not anomaly_logs:
        print("\nNo anomaly detection logs yet.")
        return
    print(f"\n=== ANOMALY DETECTION LOGS ({len(anomaly_logs)} entries) ===")
    for i, log in enumerate(anomaly_logs, 1):
        print(f"\n{i}. Time: {log['timestamp']}")
        print(f"   Location: {log['location']}")
        print(f"   Anomaly: {log['is_anomaly']}")


def menu():
    while True:
        print("\n" + "="*50)
        print("    📍 GPS ANOMALY DETECTION SYSTEM")
        print("="*50)
        print("1. Add new location to history")
        print("2. Check if a location is an anomaly")
        print("3. View anomaly detection logs")
        print("4. Exit")
        choice = input("\nChoose an option (1-4): ")
        if choice == "1":
            add_location()
        elif choice == "2":
            check_anomaly()
        elif choice == "3":
            view_logs()
        elif choice == "4":
            print("Goodbye! Stay safe!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    print("\n📍 Welcome to GPS Anomaly Detection System!")
    print("This system detects if a GPS location is an anomaly based on your history using machine learning.")
    menu()