# Banking Behavioral Authentication System - Integration Guide

## Overview
This FastAPI backend provides comprehensive behavioral biometric authentication for banking applications, combining:
- **Typing Pattern Analysis** - Keystroke dynamics
- **Tap Pattern Analysis** - Touch screen interactions  
- **Navigation Pattern Analysis** - User interface navigation
- **GPS Location Analysis** - Geographic anomaly detection
- **Multi-factor Authentication** - Combined behavioral verification

## Quick Start

### 1. Install Dependencies
```bash
cd "C:\Users\megha\OneDrive\Desktop\typing speed"
pip install -r requirements_banking.txt
```

### 2. Run the Server
```bash
python banking_auth_app.py
```

### 3. Access the API
- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/api/health

## API Endpoints

### Authentication Endpoints

#### Register User
```javascript
POST /api/register-user
Content-Type: application/x-www-form-urlencoded

{
  "username": "john_doe",
  "password": "secure_password",
  "auth_type": "typing", // or "tap", "navigation"
  "behavioral_data": "{\"timestamps\": [0.0, 0.5, 1.2, 1.8, 2.5]}"
}
```

#### Authenticate User
```javascript
POST /api/authenticate-user
Content-Type: application/x-www-form-urlencoded

{
  "username": "john_doe",
  "password": "secure_password", 
  "auth_type": "typing",
  "behavioral_data": "{\"timestamps\": [0.0, 0.6, 1.3, 1.9, 2.6]}",
  "gps_lat": 40.7128,
  "gps_lng": -74.0060
}
```

#### Verify Session
```javascript
POST /api/verify-session
Content-Type: application/x-www-form-urlencoded

{
  "session_token": "abc123..."
}
```

### Banking Endpoints

#### Process Transfer
```javascript
POST /api/banking/transfer
Content-Type: application/x-www-form-urlencoded

{
  "session_token": "abc123...",
  "to_account": "1234567890",
  "amount": 1000.00,
  "description": "Rent payment"
}
```

#### Verify Transaction
```javascript
POST /api/banking/verify-transaction
Content-Type: application/x-www-form-urlencoded

{
  "session_token": "abc123...",
  "transaction_id": "txn_123456",
  "behavioral_data": "{\"timestamps\": [0.0, 0.5, 1.1]}"
}
```

## Frontend Integration

### 1. User Registration Flow
```javascript
// Collect behavioral data during registration
let typingTimestamps = [];

function startTypingCapture() {
    typingTimestamps = [];
    const startTime = Date.now();
    
    document.addEventListener('keydown', function(e) {
        const timestamp = (Date.now() - startTime) / 1000;
        typingTimestamps.push(timestamp);
    });
}

async function registerUser(username, password) {
    const behavioralData = JSON.stringify({
        timestamps: typingTimestamps
    });
    
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);
    formData.append('auth_type', 'typing');
    formData.append('behavioral_data', behavioralData);
    
    const response = await fetch('/api/register-user', {
        method: 'POST',
        body: formData
    });
    
    return await response.json();
}
```

### 2. User Authentication Flow
```javascript
async function authenticateUser(username, password) {
    // Collect current behavioral data
    const behavioralData = JSON.stringify({
        timestamps: typingTimestamps
    });
    
    // Get GPS location
    let gpsLat = null, gpsLng = null;
    if (navigator.geolocation) {
        const position = await new Promise((resolve, reject) => {
            navigator.geolocation.getCurrentPosition(resolve, reject);
        });
        gpsLat = position.coords.latitude;
        gpsLng = position.coords.longitude;
    }
    
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);
    formData.append('auth_type', 'typing');
    formData.append('behavioral_data', behavioralData);
    if (gpsLat) formData.append('gps_lat', gpsLat);
    if (gpsLng) formData.append('gps_lng', gpsLng);
    
    const response = await fetch('/api/authenticate-user', {
        method: 'POST',
        body: formData
    });
    
    const result = await response.json();
    
    if (result.status === 'success' && result.session_token) {
        // Store session token
        localStorage.setItem('session_token', result.session_token);
        return result;
    }
    
    return result;
}
```

### 3. Banking Transaction Flow
```javascript
async function processTransfer(toAccount, amount, description) {
    const sessionToken = localStorage.getItem('session_token');
    
    const formData = new FormData();
    formData.append('session_token', sessionToken);
    formData.append('to_account', toAccount);
    formData.append('amount', amount);
    formData.append('description', description);
    
    const response = await fetch('/api/banking/transfer', {
        method: 'POST',
        body: formData
    });
    
    return await response.json();
}

async function verifyTransaction(transactionId) {
    const sessionToken = localStorage.getItem('session_token');
    
    // Collect behavioral data for verification
    const behavioralData = JSON.stringify({
        timestamps: typingTimestamps
    });
    
    const formData = new FormData();
    formData.append('session_token', sessionToken);
    formData.append('transaction_id', transactionId);
    formData.append('behavioral_data', behavioralData);
    
    const response = await fetch('/api/banking/verify-transaction', {
        method: 'POST',
        body: formData
    });
    
    return await response.json();
}
```

## Behavioral Data Collection

### Typing Pattern
```javascript
// Collect keystroke timestamps
let typingStartTime = null;
let typingTimestamps = [];

function startTypingCapture() {
    typingStartTime = Date.now();
    typingTimestamps = [];
    
    const inputField = document.getElementById('password-input');
    inputField.addEventListener('keydown', recordKeystroke);
}

function recordKeystroke(event) {
    const timestamp = (Date.now() - typingStartTime) / 1000;
    typingTimestamps.push(timestamp);
}

function stopTypingCapture() {
    const inputField = document.getElementById('password-input');
    inputField.removeEventListener('keydown', recordKeystroke);
}
```

### Tap Pattern
```javascript
// Collect tap timestamps
let tapStartTime = null;
let tapTimestamps = [];

function startTapCapture() {
    tapStartTime = Date.now();
    tapTimestamps = [];
    
    document.addEventListener('click', recordTap);
}

function recordTap(event) {
    const timestamp = (Date.now() - tapStartTime) / 1000;
    tapTimestamps.push(timestamp);
}

function stopTapCapture() {
    document.removeEventListener('click', recordTap);
}
```

### Navigation Pattern
```javascript
// Collect navigation logs
let navigationLogs = [];

function recordNavigation(screen, transitionType, depth, gesture) {
    const log = {
        screen: screen,
        timestamp: new Date().toISOString(),
        time_spent: 0, // Calculate based on previous navigation
        transition_type: transitionType,
        navigation_depth: depth,
        gesture_type: gesture
    };
    
    navigationLogs.push(log);
}
```

## Security Features

### Bot Detection
- **Too Perfect Timing**: Variance < 0.01
- **Unnaturally Fast**: Intervals < 50ms
- **Too Consistent**: Max-Min interval < 100ms
- **Unrealistic Speed**: > 8 taps/sec

### GPS Anomaly Detection
- **Distance Threshold**: 100km
- **Location History**: Compare with user's historical locations
- **Confidence Scoring**: Based on distance from known locations

### Multi-factor Authentication
- **Password**: Traditional authentication
- **Behavioral**: Typing/tap/navigation patterns
- **Location**: GPS verification
- **Session Management**: Secure token-based sessions

## Response Codes

### Authentication Results
- **authenticated**: Confidence > 85%, proceed
- **suspicious**: Confidence 65-85%, require OTP
- **rejected**: Confidence < 65%, block

### Actions
- **proceed**: Allow access
- **require_otp**: Request additional verification
- **block**: Deny access

## Error Handling

```javascript
async function handleApiError(response) {
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'API Error');
    }
    return response.json();
}

// Usage
try {
    const result = await authenticateUser(username, password);
    if (result.result === 'authenticated') {
        // Proceed to dashboard
        window.location.href = '/dashboard';
    } else if (result.result === 'suspicious') {
        // Show OTP input
        showOTPInput();
    } else {
        // Show error message
        showError('Authentication failed');
    }
} catch (error) {
    showError(error.message);
}
```

## Production Considerations

### Security
- Use HTTPS in production
- Implement rate limiting
- Add request validation
- Use secure session management
- Encrypt sensitive data

### Performance
- Cache behavioral profiles
- Optimize database queries
- Use connection pooling
- Implement load balancing

### Monitoring
- Log authentication attempts
- Monitor success/failure rates
- Track behavioral pattern changes
- Alert on security anomalies

## Testing

### Test Behavioral Patterns
```javascript
// Test typing pattern
const testTimestamps = [0.0, 0.5, 1.2, 1.8, 2.5, 3.1, 3.8, 4.4, 5.0];

// Test tap pattern  
const testTapTimestamps = [0.0, 0.8, 1.5, 2.3, 3.0, 3.8, 4.5];

// Test navigation pattern
const testNavigationLogs = [
    {screen: "login", timestamp: "2024-01-01T10:00:00", time_spent: 2.5, transition_type: "forward", navigation_depth: 0, gesture_type: "tap"},
    {screen: "dashboard", timestamp: "2024-01-01T10:00:02", time_spent: 5.0, transition_type: "forward", navigation_depth: 1, gesture_type: "tap"}
];
```

## Support

For integration support:
1. Check the API documentation at `/docs`
2. Review the health endpoint at `/api/health`
3. Test with the provided examples
4. Monitor authentication logs

## License

This behavioral authentication system is provided for banking security applications. 