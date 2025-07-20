#!/usr/bin/env python3
"""
GPS Anomaly Detection Model Training Script
This script trains an Isolation Forest model for GPS anomaly detection
and saves it for use in the main banking application.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

def train_gps_anomaly_model():
    """Train and save GPS anomaly detection model."""
    
    # Generate synthetic GPS location data for training
    # In production, this would be real user location history
    
    # Normal locations (within reasonable travel distances)
    normal_locations = [
        # User's typical locations (home, work, etc.)
        [40.7128, -74.0060],  # New York
        [40.7130, -74.0055],  # New York (nearby)
        [40.7127, -74.0059],  # New York (nearby)
        [40.7135, -74.0065],  # New York (nearby)
        [40.7120, -74.0050],  # New York (nearby)
        [40.7140, -74.0070],  # New York (nearby)
        [40.7115, -74.0045],  # New York (nearby)
        [40.7138, -74.0068],  # New York (nearby)
        [40.7122, -74.0052],  # New York (nearby)
        [40.7132, -74.0062],  # New York (nearby)
    ]
    
    # Anomalous locations (far from normal patterns)
    anomalous_locations = [
        [51.5074, -0.1278],   # London (very far)
        [35.6762, 139.6503],  # Tokyo (very far)
        [48.8566, 2.3522],    # Paris (very far)
        [-33.8688, 151.2093], # Sydney (very far)
        [55.7558, 37.6176],   # Moscow (very far)
    ]
    
    # Combine normal and anomalous data
    X_train = np.array(normal_locations + anomalous_locations)
    
    # Train Isolation Forest model
    model = IsolationForest(
        contamination=0.1,  # Expect 10% anomalies
        random_state=42,
        n_estimators=100
    )
    
    model.fit(X_train)
    
    # Save the model
    model_path = "gps_anomaly_model.joblib"
    joblib.dump(model, model_path)
    
    print(f"GPS anomaly model trained and saved to {model_path}")
    print(f"Training data shape: {X_train.shape}")
    print(f"Model contamination: {model.contamination}")
    
    # Test the model
    print("\nTesting model predictions:")
    for i, location in enumerate(normal_locations[:3]):
        prediction = model.predict([location])
        print(f"Normal location {i+1}: {location} -> {'Normal' if prediction[0] == 1 else 'Anomaly'}")
    
    for i, location in enumerate(anomalous_locations[:3]):
        prediction = model.predict([location])
        print(f"Anomalous location {i+1}: {location} -> {'Normal' if prediction[0] == 1 else 'Anomaly'}")
    
    return model

if __name__ == "__main__":
    print("Training GPS Anomaly Detection Model...")
    model = train_gps_anomaly_model()
    print("Training completed successfully!") 