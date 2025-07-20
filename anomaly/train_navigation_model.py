#!/usr/bin/env python3
"""
Navigation Anomaly Detection Model Training Script
This script trains an Isolation Forest model for navigation anomaly detection
and saves it for use in the main banking application.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

def train_navigation_anomaly_model():
    """Train and save navigation anomaly detection model."""
    
    # Generate synthetic navigation data for training
    # In production, this would be real user navigation patterns
    
    # Normal navigation patterns (realistic user behavior)
    normal_patterns = [
        # [number_of_accesses, avg_time_between_accesses]
        [5, 60],   # 5 accesses, 60s avg between (normal browsing)
        [10, 45],  # 10 accesses, 45s avg between (active user)
        [7, 70],   # 7 accesses, 70s avg between (casual browsing)
        [8, 55],   # 8 accesses, 55s avg between (normal activity)
        [6, 65],   # 6 accesses, 65s avg between (steady user)
        [12, 40],  # 12 accesses, 40s avg between (very active user)
        [4, 80],   # 4 accesses, 80s avg between (slow browsing)
        [9, 50],   # 9 accesses, 50s avg between (normal activity)
        [11, 35],  # 11 accesses, 35s avg between (fast browsing)
        [3, 90],   # 3 accesses, 90s avg between (minimal activity)
    ]
    
    # Anomalous navigation patterns (bot-like or suspicious behavior)
    anomalous_patterns = [
        [50, 2],   # 50 accesses, 2s avg between (bot-like rapid access)
        [100, 1],  # 100 accesses, 1s avg between (extreme bot behavior)
        [20, 5],   # 20 accesses, 5s avg between (suspicious rapid access)
        [30, 3],   # 30 accesses, 3s avg between (automated behavior)
        [15, 10],  # 15 accesses, 10s avg between (suspicious pattern)
    ]
    
    # Combine normal and anomalous data
    X_train = np.array(normal_patterns + anomalous_patterns)
    
    # Train Isolation Forest model
    model = IsolationForest(
        contamination=0.1,  # Expect 10% anomalies
        random_state=42,
        n_estimators=100
    )
    
    model.fit(X_train)
    
    # Save the model
    model_path = "navigation_anomaly_model.joblib"
    joblib.dump(model, model_path)
    
    print(f"Navigation anomaly model trained and saved to {model_path}")
    print(f"Training data shape: {X_train.shape}")
    print(f"Model contamination: {model.contamination}")
    
    # Test the model
    print("\nTesting model predictions:")
    print("Normal patterns:")
    for i, pattern in enumerate(normal_patterns[:3]):
        prediction = model.predict([pattern])
        print(f"  Pattern {i+1}: {pattern} -> {'Normal' if prediction[0] == 1 else 'Anomaly'}")
    
    print("\nAnomalous patterns:")
    for i, pattern in enumerate(anomalous_patterns[:3]):
        prediction = model.predict([pattern])
        print(f"  Pattern {i+1}: {pattern} -> {'Normal' if prediction[0] == 1 else 'Anomaly'}")
    
    return model

if __name__ == "__main__":
    print("Training Navigation Anomaly Detection Model...")
    model = train_navigation_anomaly_model()
    print("Training completed successfully!") 