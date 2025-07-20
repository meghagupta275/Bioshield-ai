# 👆 Tap Speed Anomaly Detection Training Guide

## 🎯 Overview

This guide explains how to train and use the tap speed anomaly detection model that combines machine learning (from the anomaly folder) with rule-based detection (from the main banking app).

## 🚀 Quick Start

### **1. Train the Model**
```bash
# Navigate to anomaly folder
cd anoamly

# Run the training script
python train_tap_model.py
```

### **2. Expected Output**
```
🤖 Tap Speed Anomaly Detection Model Trainer
==================================================
🚀 Creating training dataset...
🔄 Generating 150 human tap patterns...
✅ Generated 150 human patterns
🔄 Generating 75 bot tap patterns...
✅ Generated 75 bot patterns
📊 Dataset created:
   - Human patterns: 150
   - Bot patterns: 75
   - Total samples: 225
   - Features per sample: 5

🤖 Training Isolation Forest model...

📈 Model Performance:
Training Set:
              precision    recall  f1-score   support
       Human       0.74      1.00      0.85       120
         Bot       1.00      0.30      0.46        60
    accuracy                           0.77       180
   macro avg       0.87      0.65      0.66       180
weighted avg       0.83      0.77      0.72       180

Test Set:
              precision    recall  f1-score   support
       Human       0.72      0.97      0.83        30
         Bot       0.80      0.27      0.40        15
    accuracy                           0.73        45
   macro avg       0.76      0.62      0.61        45

🔍 Confusion Matrix (Test Set):
Predicted:
         Human  Bot
Actual Human    29     1
      Bot       11     4

💾 Saving model to tap_anomaly_model.joblib...
✅ Model saved successfully!
   - Model file: tap_anomaly_model.joblib
   - Metadata: tap_anomaly_model_metadata.json

🧪 Testing model with example patterns...
Human-like pattern: NORMAL
Bot-like pattern: ANOMALY

🎉 Training completed successfully!
```

### **3. Copy Model to Banking App**
```bash
# Copy model files to banking app directory
copy tap_anomaly_model.joblib "../typing speed/"
copy tap_anomaly_model_metadata.json "../typing speed/"
```

## 📊 Model Details

### **Features Extracted (5 features):**
1. **avg_interval**: Average time between taps
2. **min_interval**: Minimum time between taps
3. **max_interval**: Maximum time between taps
4. **variance**: Variation in tap intervals
5. **tap_speed**: Overall tap speed (taps per second)

### **Detection Methods:**
1. **Rule-Based Detection** (7 rules):
   - Too perfect timing (no natural variation)
   - Unnaturally fast taps (< 50ms)
   - Too consistent rhythm
   - Unrealistic tap speed (> 8 taps/sec)
   - Perfect intervals (machine-like)
   - Unusually slow tapping (< 0.5 taps/sec)
   - Highly irregular pattern

2. **ML-Based Detection**:
   - Isolation Forest algorithm
   - Trained on 225 synthetic patterns
   - 150 human patterns + 75 bot patterns
   - 73% accuracy on test set

## 🔧 Training Process

### **Step 1: Generate Synthetic Data**
```python
def generate_human_tap_patterns(num_samples=150):
    """Generate human-like tap patterns with natural variation."""
    for i in range(num_samples):
        base_interval = np.random.uniform(0.3, 1.2)  # 300ms to 1.2s
        num_taps = np.random.randint(5, 15)
        
        timestamps = [0.0]
        current_time = 0.0
        
        for tap in range(num_taps - 1):
            # Add natural variation
            variation = np.random.normal(0, 0.1)  # ±100ms
            interval = max(0.1, base_interval + variation)
            current_time += interval
            timestamps.append(current_time)
        
        features = extract_features(timestamps)
        human_patterns.append(features)
```

### **Step 2: Generate Bot Patterns**
```python
def generate_bot_tap_patterns(num_samples=75):
    """Generate bot-like patterns with suspicious characteristics."""
    pattern_types = ['perfect', 'too_fast', 'too_consistent', 'irregular']
    
    for i in range(num_samples):
        pattern_type = np.random.choice(pattern_types)
        
        if pattern_type == 'perfect':
            # Perfect timing (no variation)
            base_interval = np.random.uniform(0.2, 0.8)
            timestamps = [i * base_interval for i in range(num_taps)]
            
        elif pattern_type == 'too_fast':
            # Unnaturally fast taps
            base_interval = np.random.uniform(0.02, 0.08)  # 20-80ms
            timestamps = [i * base_interval for i in range(num_taps)]
            
        # ... other pattern types
```

### **Step 3: Train Isolation Forest**
```python
def train_isolation_forest_model(X, y):
    """Train Isolation Forest model for anomaly detection."""
    # Split data for validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = IsolationForest(
        contamination=0.1,  # Expect 10% anomalies
        random_state=42,
        n_estimators=100,
        max_samples='auto'
    )
    
    model.fit(X_train)
    return model, X_test, y_test
```

## 🎯 Model Performance

### **Training Results:**
- **Accuracy**: 73.3%
- **Precision**: 80.0%
- **Recall**: 26.7%
- **F1-Score**: 40.0%

### **Confusion Matrix:**
```
Predicted:
         Human  Bot
Actual Human    29     1
      Bot       11     4
```

### **Interpretation:**
- **29 True Negatives**: Correctly identified human patterns
- **4 True Positives**: Correctly identified bot patterns
- **1 False Positive**: Human pattern flagged as bot
- **11 False Negatives**: Bot pattern missed

## 🔄 Integration with Banking App

### **1. Model Loading**
```python
# In banking_auth_app.py
tap_anomaly_model = None
TAP_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'tap_anomaly_model.joblib'))
if os.path.exists(TAP_MODEL_PATH):
    tap_anomaly_model = joblib.load(TAP_MODEL_PATH)
    logger.info("Tap anomaly model loaded successfully")
```

### **2. Enhanced Analysis**
```python
def analyze_tap_speed_anomaly(timestamps: List[float]) -> dict:
    # Rule-based detection (existing)
    # ... 7 detection rules ...
    
    # ML-based detection (new)
    ml_anomaly = False
    ml_confidence = 1.0
    if tap_anomaly_model is not None and len(timestamps) >= 3:
        features = extract_tap_features(timestamps)
        if features:
            X = np.array(features).reshape(1, -1)
            prediction = tap_anomaly_model.predict(X)
            ml_anomaly = prediction[0] == -1
            
            if ml_anomaly:
                flags.append("ML model: Tap pattern anomaly detected")
                anomaly_score += 0.5
                ml_confidence = 0.3
            else:
                ml_confidence = 0.9
    
    # Combine rule-based and ML confidence
    combined_confidence = (confidence + ml_confidence) / 2
    
    return {
        "is_anomaly": is_anomaly,
        "confidence": combined_confidence,
        "ml_anomaly": ml_anomaly,
        "ml_confidence": ml_confidence,
        # ... other fields ...
    }
```

### **3. API Endpoint**
```python
@app.post("/api/v2/tap-speed/analyze")
async def analyze_tap_speed(
    timestamps: List[float],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Analyze tap speed using both rule-based and ML detection."""
    analysis = analyze_tap_speed_anomaly(timestamps)
    
    # Log analysis
    await log_security_event(
        "tap_speed_analysis",
        current_user.id,
        json.dumps(analysis),
        db=db
    )
    
    return analysis
```

## 🧪 Testing the Model

### **1. Test with Human Pattern**
```python
# Human-like pattern (should be NORMAL)
human_timestamps = [0.0, 0.8, 1.6, 2.5, 3.3, 4.2, 5.1]
features = extract_tap_features(human_timestamps)
prediction = model.predict([features])
result = "ANOMALY" if prediction[0] == -1 else "NORMAL"
print(f"Human-like pattern: {result}")
```

### **2. Test with Bot Pattern**
```python
# Bot-like pattern (should be ANOMALY)
bot_timestamps = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # Perfect timing
features = extract_tap_features(bot_timestamps)
prediction = model.predict([features])
result = "ANOMALY" if prediction[0] == -1 else "NORMAL"
print(f"Bot-like pattern: {result}")
```

## 📈 Model Customization

### **1. Adjust Contamination**
```python
# For more sensitive detection (higher false positives)
model = IsolationForest(contamination=0.2)  # Expect 20% anomalies

# For less sensitive detection (higher false negatives)
model = IsolationForest(contamination=0.05)  # Expect 5% anomalies
```

### **2. Add More Training Data**
```python
# Increase training samples
human_patterns = generate_human_tap_patterns(300)  # 300 human patterns
bot_patterns = generate_bot_tap_patterns(150)      # 150 bot patterns
```

### **3. Feature Engineering**
```python
def extract_tap_features_enhanced(timestamps):
    """Enhanced feature extraction with additional features."""
    # Basic features
    intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
    avg_interval = sum(intervals) / len(intervals)
    min_interval = min(intervals)
    max_interval = max(intervals)
    variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
    tap_speed = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
    
    # Additional features
    consistency = max_interval - min_interval
    rhythm_variation = variance / avg_interval if avg_interval > 0 else 0
    acceleration = np.diff(intervals).mean() if len(intervals) > 1 else 0
    
    return [avg_interval, min_interval, max_interval, variance, tap_speed, 
            consistency, rhythm_variation, acceleration]
```

## 🔍 Monitoring and Maintenance

### **1. Model Performance Tracking**
```python
def track_model_performance(predictions, actual_labels):
    """Track model performance over time."""
    accuracy = np.mean(predictions == actual_labels)
    precision = np.sum((predictions == 1) & (actual_labels == 1)) / np.sum(predictions == 1)
    recall = np.sum((predictions == 1) & (actual_labels == 1)) / np.sum(actual_labels == 1)
    
    # Log performance metrics
    logger.info(f"Model Performance - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
    
    return {"accuracy": accuracy, "precision": precision, "recall": recall}
```

### **2. Retraining Schedule**
```python
# Retrain model monthly with new data
def schedule_retraining():
    """Schedule monthly model retraining."""
    current_date = datetime.now()
    last_training = get_last_training_date()
    
    if (current_date - last_training).days > 30:
        logger.info("Scheduling model retraining...")
        retrain_model()
```

### **3. A/B Testing**
```python
def ab_test_models(new_model, old_model, test_data):
    """A/B test new model against old model."""
    new_predictions = new_model.predict(test_data)
    old_predictions = old_model.predict(test_data)
    
    new_performance = evaluate_model(new_predictions, test_labels)
    old_performance = evaluate_model(old_predictions, test_labels)
    
    if new_performance["f1_score"] > old_performance["f1_score"]:
        deploy_new_model(new_model)
        logger.info("New model deployed successfully")
    else:
        logger.info("Keeping old model - new model didn't improve performance")
```

## 🚨 Troubleshooting

### **Common Issues:**

#### **1. Model Not Loading**
```bash
# Check if model file exists
ls -la tap_anomaly_model.joblib

# Check file permissions
chmod 644 tap_anomaly_model.joblib

# Verify model integrity
python -c "import joblib; model = joblib.load('tap_anomaly_model.joblib'); print('Model loaded successfully')"
```

#### **2. Poor Performance**
```python
# Check training data quality
print(f"Training samples: {len(X_train)}")
print(f"Feature distribution: {X_train.mean(axis=0)}")
print(f"Feature variance: {X_train.var(axis=0)}")

# Adjust model parameters
model = IsolationForest(
    contamination=0.15,  # Try different contamination
    n_estimators=200,    # More trees
    max_samples=100      # Larger sample size
)
```

#### **3. High False Positives**
```python
# Reduce sensitivity
model = IsolationForest(contamination=0.05)  # Lower contamination

# Add more human training data
human_patterns = generate_human_tap_patterns(500)  # More human patterns
```

## 📋 Best Practices

### **1. Data Quality**
- Use diverse human tap patterns
- Include various bot attack patterns
- Balance training data (more humans than bots)
- Validate feature distributions

### **2. Model Selection**
- Isolation Forest for unsupervised learning
- Consider ensemble methods for better performance
- Regular retraining with new data
- A/B testing before deployment

### **3. Production Deployment**
- Monitor model performance
- Set up alerts for performance degradation
- Maintain model versioning
- Document model decisions

### **4. Security Considerations**
- Validate input data
- Handle edge cases gracefully
- Log all predictions for audit
- Implement rate limiting

## 🎉 Success Metrics

### **Target Performance:**
- **Accuracy**: > 80%
- **Precision**: > 85%
- **Recall**: > 70%
- **F1-Score**: > 75%

### **Business Impact:**
- Reduced false positives (better user experience)
- Improved bot detection (enhanced security)
- Faster response times (real-time analysis)
- Lower manual review workload

## 📞 Support

For questions or issues with tap speed anomaly detection:

1. **Check the logs**: Look for error messages in the application logs
2. **Verify model files**: Ensure model files are in the correct location
3. **Test with examples**: Use the provided test patterns
4. **Review performance**: Monitor accuracy and adjust parameters

The integrated tap speed anomaly detection system provides the best of both worlds: advanced ML-based detection with real-time production integration! 🚀 