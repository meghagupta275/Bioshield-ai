# 🤖 Anomaly Detection Model Training Guide

## Overview

This guide explains how to train all the anomaly detection models used in the banking application, including GPS, navigation, and behavioral anomaly detection.

## 📁 Model Files

### **Generated Models:**
- `anoamly/gps_anomaly_model.joblib` - GPS location anomaly detection
- `anoamly/navigation_anomaly_model.joblib` - Navigation pattern anomaly detection
- `anoamly/anomaly_model.joblib` - General behavioral anomaly detection

### **Training Scripts:**
- `anoamly/train_gps_model.py` - GPS model training
- `anoamly/train_navigation_model.py` - Navigation model training
- `anoamly/app.py` - General anomaly model training

## 🚀 Quick Training Setup

### **Step 1: Navigate to Anomaly Folder**
```bash
cd anoamly
```

### **Step 2: Train All Models**
```bash
# Train GPS anomaly detection model
python train_gps_model.py

# Train navigation anomaly detection model
python train_navigation_model.py

# Train general anomaly detection model
python app.py
```

### **Step 3: Verify Models Created**
```bash
ls *.joblib
```

You should see:
- `gps_anomaly_model.joblib`
- `navigation_anomaly_model.joblib`
- `anomaly_model.joblib`

## 📊 Model Training Details

### **1. GPS Anomaly Model (`train_gps_model.py`)**

#### **Training Data:**
```python
# Normal locations (user's typical areas)
normal_locations = [
    [40.7128, -74.0060],  # New York
    [40.7130, -74.0055],  # New York (nearby)
    [40.7127, -74.0059],  # New York (nearby)
    # ... more nearby locations
]

# Anomalous locations (far from normal patterns)
anomalous_locations = [
    [51.5074, -0.1278],   # London (very far)
    [35.6762, 139.6503],  # Tokyo (very far)
    [48.8566, 2.3522],    # Paris (very far)
    # ... more distant locations
]
```

#### **Features:**
- **Latitude**: GPS latitude coordinate
- **Longitude**: GPS longitude coordinate

#### **Algorithm:**
- **Isolation Forest**: Detects unusual location patterns
- **Contamination**: 10% expected anomaly rate
- **Random State**: 42 (for reproducibility)

#### **Training Command:**
```bash
python train_gps_model.py
```

#### **Expected Output:**
```
Training GPS Anomaly Detection Model...
GPS anomaly model trained and saved to gps_anomaly_model.joblib
Training data shape: (15, 2)
Model contamination: 0.1

Testing model predictions:
Normal location 1: [40.7128, -74.006] -> Normal
Normal location 2: [40.713, -74.0055] -> Normal
Normal location 3: [40.7127, -74.0059] -> Normal
Anomalous location 1: [51.5074, -0.1278] -> Normal
Anomalous location 2: [35.6762, 139.6503] -> Anomaly
Anomalous location 3: [48.8566, 2.3522] -> Normal
Training completed successfully!
```

---

### **2. Navigation Anomaly Model (`train_navigation_model.py`)**

#### **Training Data:**
```python
# Normal navigation patterns (realistic user behavior)
normal_patterns = [
    [5, 60],   # 5 accesses, 60s avg between (normal browsing)
    [10, 45],  # 10 accesses, 45s avg between (active user)
    [7, 70],   # 7 accesses, 70s avg between (casual browsing)
    # ... more normal patterns
]

# Anomalous navigation patterns (bot-like behavior)
anomalous_patterns = [
    [50, 2],   # 50 accesses, 2s avg between (bot-like rapid access)
    [100, 1],  # 100 accesses, 1s avg between (extreme bot behavior)
    [20, 5],   # 20 accesses, 5s avg between (suspicious rapid access)
    # ... more anomalous patterns
]
```

#### **Features:**
- **Number of Accesses**: Total page accesses
- **Average Time Between**: Average time between navigation events

#### **Algorithm:**
- **Isolation Forest**: Detects unusual navigation patterns
- **Contamination**: 10% expected anomaly rate
- **Random State**: 42 (for reproducibility)

#### **Training Command:**
```bash
python train_navigation_model.py
```

#### **Expected Output:**
```
Training Navigation Anomaly Detection Model...
Navigation anomaly model trained and saved to navigation_anomaly_model.joblib
Training data shape: (15, 2)
Model contamination: 0.1

Testing model predictions:
Normal patterns:
  Pattern 1: [5, 60] -> Normal
  Pattern 2: [10, 45] -> Normal
  Pattern 3: [7, 70] -> Normal
Anomalous patterns:
  Pattern 1: [50, 2] -> Normal
  Pattern 2: [100, 1] -> Anomaly
  Pattern 3: [20, 5] -> Normal
Training completed successfully!
```

---

### **3. General Anomaly Model (`app.py`)**

#### **Training Data:**
```python
# Example user history (GPS locations)
user_history = [
    (40.7128, -74.0060),  # New York
    (40.7130, -74.0055),  # New York (nearby)
    (40.7127, -74.0059),  # New York (nearby)
]
```

#### **Features:**
- **GPS Coordinates**: Latitude and longitude pairs
- **Distance Analysis**: Distance-based anomaly detection

#### **Algorithm:**
- **Isolation Forest**: General anomaly detection
- **Distance Rules**: 50km threshold for location anomalies

#### **Training Command:**
```bash
python app.py
```

#### **Expected Output:**
```
Your current location: (40.7128, -74.0060)
Model saved as anomaly_model.joblib
Anomaly detected: False
```

## 🔧 Custom Training

### **Customizing GPS Training Data**

Edit `train_gps_model.py` to add your own location data:

```python
# Add your user's typical locations
normal_locations = [
    [YOUR_LAT, YOUR_LON],  # User's home
    [WORK_LAT, WORK_LON],  # User's work
    [SHOP_LAT, SHOP_LON],  # User's shopping area
    # ... add more typical locations
]

# Add locations that should be flagged as anomalies
anomalous_locations = [
    [FAR_LAT, FAR_LON],    # Very distant location
    [SUSPICIOUS_LAT, SUSPICIOUS_LON],  # Suspicious location
    # ... add more anomalous locations
]
```

### **Customizing Navigation Training Data**

Edit `train_navigation_model.py` to add your own navigation patterns:

```python
# Add realistic user navigation patterns
normal_patterns = [
    [5, 60],   # 5 page accesses, 60s average between
    [10, 45],  # 10 page accesses, 45s average between
    # ... add more normal patterns
]

# Add suspicious navigation patterns
anomalous_patterns = [
    [50, 2],   # 50 page accesses, 2s average between (bot-like)
    [100, 1],  # 100 page accesses, 1s average between (extreme bot)
    # ... add more anomalous patterns
]
```

## 📈 Model Performance

### **GPS Model Performance:**
- **Accuracy**: Detects locations >50km from normal patterns
- **False Positives**: May flag legitimate travel as anomaly
- **False Negatives**: May miss sophisticated location spoofing

### **Navigation Model Performance:**
- **Accuracy**: Detects rapid, automated navigation patterns
- **False Positives**: May flag fast legitimate users as anomalies
- **False Negatives**: May miss sophisticated bot behavior

### **General Model Performance:**
- **Accuracy**: General anomaly detection across multiple features
- **Flexibility**: Can be adapted for different anomaly types
- **Scalability**: Handles multiple users and patterns

## 🔄 Retraining Models

### **When to Retrain:**
1. **New User Patterns**: When user behavior changes significantly
2. **False Positives**: When legitimate behavior is flagged as anomaly
3. **False Negatives**: When anomalies are not detected
4. **New Threat Patterns**: When new types of attacks are discovered

### **Retraining Process:**
```bash
# 1. Backup existing models
cp *.joblib *.joblib.backup

# 2. Update training data in scripts
# Edit the training scripts with new data

# 3. Retrain models
python train_gps_model.py
python train_navigation_model.py
python app.py

# 4. Test new models
# Use the test functions in the scripts
```

## 🧪 Testing Models

### **Test GPS Model:**
```python
import joblib
import numpy as np

# Load the model
model = joblib.load('gps_anomaly_model.joblib')

# Test locations
test_locations = [
    [40.7128, -74.0060],  # Normal location
    [51.5074, -0.1278],   # Anomalous location
]

for location in test_locations:
    prediction = model.predict([location])
    print(f"Location {location}: {'Normal' if prediction[0] == 1 else 'Anomaly'}")
```

### **Test Navigation Model:**
```python
import joblib
import numpy as np

# Load the model
model = joblib.load('navigation_anomaly_model.joblib')

# Test patterns
test_patterns = [
    [5, 60],   # Normal pattern
    [50, 2],   # Anomalous pattern
]

for pattern in test_patterns:
    prediction = model.predict([pattern])
    print(f"Pattern {pattern}: {'Normal' if prediction[0] == 1 else 'Anomaly'}")
```

## 🚀 Production Deployment

### **1. Model Validation:**
```bash
# Test all models before deployment
python -c "
import joblib
import numpy as np

# Test GPS model
gps_model = joblib.load('gps_anomaly_model.joblib')
print('GPS model loaded successfully')

# Test navigation model
nav_model = joblib.load('navigation_anomaly_model.joblib')
print('Navigation model loaded successfully')

# Test general model
gen_model = joblib.load('anomaly_model.joblib')
print('General model loaded successfully')
"
```

### **2. Start Banking Application:**
```bash
cd "../typing speed"
uvicorn banking_auth_app:app --reload
```

### **3. Verify Integration:**
- Check that models are loaded in the application
- Test GPS anomaly detection
- Test navigation anomaly detection
- Monitor behavioral panel for anomalies

## 📊 Monitoring and Maintenance

### **Model Performance Monitoring:**
- Track false positive rates
- Monitor false negative rates
- Log anomaly detection events
- Analyze user feedback

### **Regular Maintenance:**
- **Weekly**: Check model performance metrics
- **Monthly**: Review and update training data
- **Quarterly**: Retrain models with new data
- **Annually**: Complete model evaluation and optimization

## 🆘 Troubleshooting

### **Common Issues:**

#### **1. Model Not Loading:**
```bash
# Check if model files exist
ls -la *.joblib

# Check file permissions
chmod 644 *.joblib
```

#### **2. Training Errors:**
```bash
# Check Python dependencies
pip install scikit-learn joblib numpy

# Check training data format
python -c "import numpy as np; print('NumPy working')"
```

#### **3. Poor Model Performance:**
- **More Training Data**: Add more examples to training data
- **Feature Engineering**: Improve feature extraction
- **Hyperparameter Tuning**: Adjust model parameters
- **Data Quality**: Ensure training data is representative

## 📞 Support

For issues with model training:
1. Check the training scripts for errors
2. Verify all dependencies are installed
3. Ensure training data is properly formatted
4. Test models individually before integration
5. Review the comprehensive guides in the documentation

---

**Note**: The models are trained on synthetic data for demonstration. In production, use real user data for better accuracy and performance. 