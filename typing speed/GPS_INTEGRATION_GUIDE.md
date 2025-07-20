# 📍 GPS Anomaly Detection Integration Guide

## Overview

GPS anomaly detection has been successfully integrated into the banking application to provide an additional layer of security by monitoring user location patterns and detecting suspicious location changes.

## 🔧 Integration Components

### 1. Backend Integration (`banking_auth_app.py`)

#### **GPS Anomaly Detection Functions**
- `calculate_distance()`: Calculates distance between GPS coordinates
- `analyze_gps_anomaly()`: Main GPS anomaly analysis function
- `update_user_gps_location()`: Updates user location and checks for anomalies

#### **GPS API Endpoints**
- `POST /api/v2/gps/update`: Update user's GPS location
- `GET /api/v2/gps/status`: Get GPS status and anomaly information
- `POST /api/v2/gps/clear-anomaly`: Clear GPS anomaly flags

#### **Behavioral Analysis Integration**
- GPS anomalies are now included in behavioral pattern analysis
- GPS flags affect overall confidence scores
- GPS anomalies are logged in security events

### 2. Frontend Integration (`banking_dashboard.html`)

#### **GPS Monitoring**
- Real-time GPS location monitoring every 10 seconds
- Automatic anomaly detection and alerts
- GPS status display in behavioral panel

#### **GPS Controls**
- **📍 GPS Status**: Check current GPS status and location
- **🚫 Clear GPS Alert**: Clear false positive GPS anomalies
- GPS status indicator in behavioral panel

#### **Enhanced Behavioral Panel**
- GPS status display
- GPS anomaly warnings
- Location history tracking

### 3. ML Model Integration

#### **GPS Anomaly Model**
- Location: `anoamly/gps_anomaly_model.joblib`
- Training script: `anoamly/train_gps_model.py`
- Uses Isolation Forest algorithm for anomaly detection

## 🚀 How It Works

### 1. **Location Monitoring**
```javascript
// Frontend GPS monitoring
function startGPSMonitoring() {
    navigator.geolocation.getCurrentPosition(function(pos) {
        updateGPSLocation(pos.coords.latitude, pos.coords.longitude);
    });
}
```

### 2. **Anomaly Detection**
```python
# Backend GPS analysis
def analyze_gps_anomaly(user_id, current_lat, current_lon, db):
    # Distance-based detection (50km rule)
    # ML-based detection (Isolation Forest)
    # Rapid location changes (100km in 3 locations)
```

### 3. **Integration with Behavioral Analysis**
```python
# GPS anomalies affect overall confidence
if gps_anomaly and anomaly_score is not None:
    anomaly_score = max(-1, anomaly_score + gps_anomaly_score)
```

## 📊 Detection Methods

### 1. **Distance-Based Detection**
- **50km Rule**: Flags locations more than 50km from previous locations
- **100km Rapid Change**: Flags rapid location changes over 100km

### 2. **ML-Based Detection**
- **Isolation Forest**: Uses machine learning to detect unusual location patterns
- **Training Data**: Normal locations vs. anomalous locations
- **Contamination**: 10% expected anomaly rate

### 3. **Behavioral Integration**
- GPS anomalies affect overall behavioral confidence
- GPS flags are included in security logs
- GPS status is displayed in real-time

## 🔍 API Usage Examples

### Update GPS Location
```javascript
const response = await fetch('/api/v2/gps/update', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${sessionToken}`
    },
    body: JSON.stringify({
        latitude: 40.7128,
        longitude: -74.0060
    })
});
```

### Check GPS Status
```javascript
const response = await fetch('/api/v2/gps/status', {
    headers: {
        'Authorization': `Bearer ${sessionToken}`
    }
});
```

### Clear GPS Anomaly
```javascript
const response = await fetch('/api/v2/gps/clear-anomaly', {
    method: 'POST',
    headers: {
        'Authorization': `Bearer ${sessionToken}`
    }
});
```

## 🛡️ Security Features

### 1. **Real-Time Monitoring**
- GPS location checked every 10 seconds
- Immediate anomaly detection and alerts
- Automatic logging of suspicious activity

### 2. **Multi-Layer Detection**
- Distance-based rules
- ML-based anomaly detection
- Behavioral pattern integration

### 3. **User Controls**
- Users can check their GPS status
- Users can clear false positive alerts
- Transparent anomaly reporting

### 4. **Privacy Protection**
- Location data stored locally in memory
- No permanent location tracking
- User consent required for GPS access

## 📱 Frontend Features

### 1. **GPS Status Panel**
- Real-time GPS monitoring status
- Current location coordinates
- Anomaly detection status

### 2. **GPS Controls**
- **📍 GPS Status Button**: Check current GPS status
- **🚫 Clear GPS Alert Button**: Clear false positives
- GPS status indicator in behavioral panel

### 3. **Anomaly Alerts**
- Visual warnings for GPS anomalies
- Detailed anomaly information
- Color-coded status indicators

## 🔧 Setup Instructions

### 1. **Install Dependencies**
```bash
pip install geopy==2.4.0
```

### 2. **Train GPS Model**
```bash
cd anoamly
python train_gps_model.py
```

### 3. **Start Application**
```bash
cd "typing speed"
uvicorn banking_auth_app:app --reload
```

## 📈 Monitoring and Alerts

### 1. **GPS Anomaly Types**
- **Distance Anomaly**: Location too far from previous
- **ML Anomaly**: Unusual location pattern detected
- **Rapid Change**: Too many location changes quickly

### 2. **Alert Levels**
- **Warning**: Suspicious location detected
- **Critical**: Severe anomaly, account may be locked
- **Info**: Normal location updates

### 3. **Logging**
- All GPS updates logged
- Anomaly detections recorded
- Security events tracked

## 🎯 Benefits

### 1. **Enhanced Security**
- Additional layer of fraud detection
- Real-time location monitoring
- ML-powered anomaly detection

### 2. **User Experience**
- Transparent GPS monitoring
- User control over alerts
- Clear status indicators

### 3. **Compliance**
- Privacy-conscious implementation
- User consent required
- Minimal data retention

## 🔮 Future Enhancements

### 1. **Advanced Features**
- Time-based anomaly detection
- Travel pattern analysis
- Location clustering

### 2. **Integration Options**
- Database storage for location history
- Advanced ML models
- Real-time threat intelligence

### 3. **User Features**
- Location whitelist management
- Travel mode settings
- Custom anomaly thresholds

## 📞 Support

For questions or issues with GPS integration:
1. Check the behavioral panel for GPS status
2. Use the GPS controls to manage alerts
3. Review security logs for detailed information
4. Contact support for technical assistance

---

**Note**: GPS monitoring requires user consent and browser location permissions. The system respects user privacy and only monitors location when explicitly enabled. 