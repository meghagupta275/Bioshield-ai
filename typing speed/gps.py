from geopy.distance import geodesic
import geocoder

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
    (23.2547, 77.4029),
]

ANOMALY_DISTANCE_KM = 100

def is_gps_anomaly(user_history, new_location, threshold_km=100):
    for old_location in user_history:
        distance = geodesic(old_location, new_location).km
        if distance <= threshold_km:
            return False  # Not an anomaly
    return True  # Anomaly

anomaly = is_gps_anomaly(user_history, current_location, ANOMALY_DISTANCE_KM)
print("Anomaly detected:", anomaly)