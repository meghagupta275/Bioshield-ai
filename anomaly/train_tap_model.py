#!/usr/bin/env python3
"""
🤖 Tap Speed Anomaly Detection Model Trainer

This script trains a machine learning model for tap speed anomaly detection
using synthetic data and real user patterns.

Features:
- Generate synthetic human and bot tap patterns
- Train Isolation Forest model
- Save trained model for use in banking app
- Validate model performance
"""

import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import json

# Import functions from tap.py
from tap import extract_features, legitimate_users

def generate_human_tap_patterns(num_samples=100):
    """Generate synthetic human tap patterns."""
    human_patterns = []
    
    print(f"🔄 Generating {num_samples} human tap patterns...")
    
    for i in range(num_samples):
        # Human-like tap patterns with natural variation
        base_interval = np.random.uniform(0.3, 1.2)  # 300ms to 1.2s intervals
        num_taps = np.random.randint(5, 15)  # 5-15 taps
        
        timestamps = [0.0]  # Start at 0
        current_time = 0.0
        
        for tap in range(num_taps - 1):
            # Add natural variation to intervals
            variation = np.random.normal(0, 0.1)  # ±100ms variation
            interval = max(0.1, base_interval + variation)  # Minimum 100ms
            current_time += interval
            timestamps.append(current_time)
        
        # Extract features
        features = extract_features(timestamps)
        if features:
            human_patterns.append(features)
    
    print(f"✅ Generated {len(human_patterns)} human patterns")
    return human_patterns

def generate_bot_tap_patterns(num_samples=50):
    """Generate synthetic bot tap patterns."""
    bot_patterns = []
    
    print(f"🔄 Generating {num_samples} bot tap patterns...")
    
    for i in range(num_samples):
        # Bot-like patterns with suspicious characteristics
        pattern_type = np.random.choice(['perfect', 'too_fast', 'too_consistent', 'irregular'])
        
        if pattern_type == 'perfect':
            # Perfect timing (no variation)
            base_interval = np.random.uniform(0.2, 0.8)
            num_taps = np.random.randint(5, 12)
            timestamps = [i * base_interval for i in range(num_taps)]
            
        elif pattern_type == 'too_fast':
            # Unnaturally fast taps
            base_interval = np.random.uniform(0.02, 0.08)  # 20-80ms intervals
            num_taps = np.random.randint(8, 20)
            timestamps = [i * base_interval for i in range(num_taps)]
            
        elif pattern_type == 'too_consistent':
            # Too consistent rhythm
            base_interval = np.random.uniform(0.3, 0.6)
            num_taps = np.random.randint(6, 15)
            timestamps = [0.0]
            current_time = 0.0
            for tap in range(num_taps - 1):
                # Very small variation
                variation = np.random.uniform(-0.01, 0.01)
                interval = base_interval + variation
                current_time += interval
                timestamps.append(current_time)
                
        else:  # irregular
            # Highly irregular pattern
            num_taps = np.random.randint(5, 12)
            timestamps = [0.0]
            current_time = 0.0
            for tap in range(num_taps - 1):
                # Large random intervals
                interval = np.random.uniform(0.05, 2.0)
                current_time += interval
                timestamps.append(current_time)
        
        # Extract features
        features = extract_features(timestamps)
        if features:
            bot_patterns.append(features)
    
    print(f"✅ Generated {len(bot_patterns)} bot patterns")
    return bot_patterns

def create_training_dataset():
    """Create training dataset with human and bot patterns."""
    print("🚀 Creating training dataset...")
    
    # Generate synthetic patterns
    human_patterns = generate_human_tap_patterns(150)
    bot_patterns = generate_bot_tap_patterns(75)
    
    # Combine patterns
    X = np.array(human_patterns + bot_patterns)
    
    # Create labels (0 for human, 1 for bot)
    y = np.array([0] * len(human_patterns) + [1] * len(bot_patterns))
    
    print(f"📊 Dataset created:")
    print(f"   - Human patterns: {len(human_patterns)}")
    print(f"   - Bot patterns: {len(bot_patterns)}")
    print(f"   - Total samples: {len(X)}")
    print(f"   - Features per sample: {X.shape[1]}")
    
    return X, y

def train_isolation_forest_model(X, y):
    """Train Isolation Forest model for anomaly detection."""
    print("🤖 Training Isolation Forest model...")
    
    # Split data for validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Isolation Forest
    model = IsolationForest(
        contamination=0.1,  # Expect 10% anomalies
        random_state=42,
        n_estimators=100,
        max_samples='auto'
    )
    
    # Fit model on training data
    model.fit(X_train)
    
    # Make predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    # Convert predictions (1 for normal, -1 for anomaly)
    train_labels = (train_predictions == -1).astype(int)
    test_labels = (test_predictions == -1).astype(int)
    
    # Evaluate model
    print("\n📈 Model Performance:")
    print("Training Set:")
    print(classification_report(y_train, train_labels, target_names=['Human', 'Bot']))
    print("Test Set:")
    print(classification_report(y_test, test_labels, target_names=['Human', 'Bot']))
    
    # Confusion matrix
    print("\n🔍 Confusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, test_labels)
    print("Predicted:")
    print("         Human  Bot")
    print(f"Actual Human  {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"      Bot     {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    return model, X_test, y_test

def save_model_and_metadata(model, X_test, y_test, model_path="tap_anomaly_model.joblib"):
    """Save trained model and metadata."""
    print(f"💾 Saving model to {model_path}...")
    
    # Save model
    joblib.dump(model, model_path)
    
    # Create metadata
    metadata = {
        "model_type": "IsolationForest",
        "training_date": datetime.now().isoformat(),
        "features": ["avg_interval", "min_interval", "max_interval", "variance", "tap_speed"],
        "test_samples": len(X_test),
        "test_humans": int(sum(y_test == 0)),
        "test_bots": int(sum(y_test == 1)),
        "contamination": 0.1,
        "n_estimators": 100
    }
    
    # Save metadata
    metadata_path = model_path.replace('.joblib', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Model saved successfully!")
    print(f"   - Model file: {model_path}")
    print(f"   - Metadata: {metadata_path}")
    
    return model_path, metadata_path

def validate_model(model, X_test, y_test):
    """Validate model with test data."""
    print("🔍 Validating model...")
    
    # Make predictions
    predictions = model.predict(X_test)
    anomaly_labels = (predictions == -1).astype(int)
    
    # Calculate metrics
    accuracy = np.mean(anomaly_labels == y_test)
    precision = np.sum((anomaly_labels == 1) & (y_test == 1)) / np.sum(anomaly_labels == 1) if np.sum(anomaly_labels == 1) > 0 else 0
    recall = np.sum((anomaly_labels == 1) & (y_test == 1)) / np.sum(y_test == 1) if np.sum(y_test == 1) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"📊 Validation Results:")
    print(f"   - Accuracy: {accuracy:.3f}")
    print(f"   - Precision: {precision:.3f}")
    print(f"   - Recall: {recall:.3f}")
    print(f"   - F1-Score: {f1_score:.3f}")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

def test_model_with_examples(model):
    """Test model with example patterns."""
    print("\n🧪 Testing model with example patterns...")
    
    # Example 1: Human-like pattern
    human_timestamps = [0.0, 0.8, 1.6, 2.5, 3.3, 4.2, 5.1]
    human_features = extract_features(human_timestamps)
    
    if human_features:
        human_prediction = model.predict([human_features])
        human_result = "ANOMALY" if human_prediction[0] == -1 else "NORMAL"
        print(f"Human-like pattern: {human_result}")
        print(f"Features: {human_features}")
    
    # Example 2: Bot-like pattern (perfect timing)
    bot_timestamps = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    bot_features = extract_features(bot_timestamps)
    
    if bot_features:
        bot_prediction = model.predict([bot_features])
        bot_result = "ANOMALY" if bot_prediction[0] == -1 else "NORMAL"
        print(f"Bot-like pattern: {bot_result}")
        print(f"Features: {bot_features}")

def main():
    """Main training function."""
    print("🤖 Tap Speed Anomaly Detection Model Trainer")
    print("=" * 50)
    
    try:
        # Step 1: Create training dataset
        X, y = create_training_dataset()
        
        # Step 2: Train model
        model, X_test, y_test = train_isolation_forest_model(X, y)
        
        # Step 3: Validate model
        validation_results = validate_model(model, X_test, y_test)
        
        # Step 4: Save model
        model_path, metadata_path = save_model_and_metadata(model, X_test, y_test)
        
        # Step 5: Test with examples
        test_model_with_examples(model)
        
        print("\n🎉 Training completed successfully!")
        print(f"Model saved to: {model_path}")
        print(f"Metadata saved to: {metadata_path}")
        
        # Instructions for integration
        print("\n📋 Integration Instructions:")
        print("1. Copy the model file to your banking app directory")
        print("2. Load the model in your banking app:")
        print("   model = joblib.load('tap_anomaly_model.joblib')")
        print("3. Use the model for tap speed analysis")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 