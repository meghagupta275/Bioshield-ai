# 🔒 Security System Testing Guide

This guide shows you how to test all security features of your banking application.

## 🚀 Quick Start

### 1. **Automated Testing**
Run the comprehensive security test script:
```bash
cd "typing speed"
python security_test.py
```

### 2. **Manual Testing**
Follow the steps below to test each security feature manually.

---

## 📋 Security Features to Test

### **1. Session Management** ⏰

**Test Session Timeout:**
1. Login to your banking app
2. Watch the session timer (top-right corner)
3. Wait for 10 minutes or refresh the page
4. **Expected Result:** Session expires, redirects to login

**Test Session Status:**
1. Go to: `http://localhost:8000/api/v2/session/status`
2. **Expected Result:** Shows remaining time and session status

---

### **2. Transaction Anomaly Detection** 💰

**Test Large Transaction Blocking:**
1. Try to transfer ₹20,00,000 (2 crore)
2. **Expected Result:** Transaction blocked with error message

**Test Repeated Transaction Blocking:**
1. Transfer ₹10,000 to same account 4 times
2. **Expected Result:** 4th transaction blocked

**Test Normal Transaction:**
1. Transfer ₹5,000
2. **Expected Result:** Transaction allowed

---

### **3. Behavioral Analysis** 🧠

**Test Typing Pattern Analysis:**
1. Go to: `http://localhost:8000/api/v2/tap-speed/analyze`
2. Send typing timestamps
3. **Expected Result:** Returns anomaly score and recommendation

**Test Behavioral Matching:**
1. Go to: `http://localhost:8000/api/v2/behavioral/match`
2. Send behavioral data
3. **Expected Result:** Returns match score and mismatch status

---

### **4. GPS Anomaly Detection** 📍

**Test Normal GPS:**
1. Update GPS to your current location
2. **Expected Result:** No anomaly detected

**Test Anomalous GPS:**
1. Update GPS to a very far location (e.g., New York)
2. **Expected Result:** Anomaly detected

---

### **5. Transaction Limits** 🔒

**Check Current Limits:**
1. Go to: `http://localhost:8000/api/v2/banking/limits`
2. **Expected Result:** Shows all transaction limits

**Test Limit Enforcement:**
1. Try to transfer more than daily limit
2. **Expected Result:** Transaction blocked

---

### **6. Risk Profile** 📊

**Check User Risk:**
1. Go to: `http://localhost:8000/api/v2/user/risk-profile`
2. **Expected Result:** Shows confidence score, anomaly status

---

### **7. Security Logging** 📝

**View Security Logs:**
1. Go to: `http://localhost:8000/api/v2/admin/logs`
2. **Expected Result:** Shows all security events

---

## 🧪 Manual Testing Steps

### **Step 1: Start Your Server**
```bash
cd "typing speed"
uvicorn banking_auth_app:app --reload
```

### **Step 2: Open Banking Dashboard**
1. Go to: `http://localhost:8000`
2. Login with your credentials
3. Navigate to the banking dashboard

### **Step 3: Test Security Features**

#### **A. Test Transaction Limits**
1. Click "Transaction Limits" in Quick Actions
2. Check your current limits
3. Try exceeding limits with transfers

#### **B. Test Anomaly Detection**
1. Try large transfers (₹20,00,000+)
2. Try repeated transfers
3. Check if transactions are blocked

#### **C. Test Session Management**
1. Watch the session timer
2. Wait for session to expire
3. Try accessing protected pages

#### **D. Test Behavioral Security**
1. Use different typing patterns
2. Check if behavioral analysis flags anomalies
3. Monitor security badges

---

## 🔍 Security Indicators to Monitor

### **Visual Indicators:**
- ✅ **Green Security Badge:** Secure session
- ⚠️ **Yellow Security Badge:** Session expiring
- ❌ **Red Security Badge:** Security issue detected

### **Transaction Responses:**
- ✅ **200 OK:** Transaction successful
- ❌ **403 Forbidden:** Transaction blocked (anomaly)
- ❌ **400 Bad Request:** Invalid transaction

### **Session Responses:**
- ✅ **Valid Session:** Can access protected pages
- ❌ **Expired Session:** Redirected to login

---

## 📊 Expected Test Results

### **Normal User (Low Risk):**
- Daily Limit: ₹50,000
- Transaction Limit: ₹1,00,000
- Session Timeout: 10 minutes
- Behavioral Score: High confidence

### **Flagged User (High Risk):**
- Daily Limit: ₹5,000
- Transaction Limit: ₹10,000
- Session Timeout: 10 minutes
- Behavioral Score: Low confidence

### **Anomaly Detection:**
- Large transactions (>₹10,00,000): **BLOCKED**
- Repeated transactions (3+ times): **BLOCKED**
- GPS anomalies: **FLAGGED**
- Behavioral mismatches: **FLAGGED**

---

## 🛠️ Troubleshooting

### **Common Issues:**

**1. Server Not Starting:**
```bash
# Check if port 8000 is free
netstat -an | grep 8000
# Kill process if needed
kill -9 <PID>
```

**2. Database Issues:**
```bash
# Check database file
ls -la banking_auth.db
# Reset if needed
rm banking_auth.db
```

**3. Import Errors:**
```bash
# Install missing packages
pip install requests fastapi uvicorn
```

**4. Authentication Issues:**
- Check if user exists in database
- Verify password hash
- Check behavioral profile

---

## 📈 Security Metrics

### **Key Performance Indicators:**
- **False Positive Rate:** < 5%
- **False Negative Rate:** < 1%
- **Response Time:** < 2 seconds
- **Session Security:** 100% timeout compliance

### **Monitoring Points:**
- Transaction success/failure rates
- Anomaly detection accuracy
- Session management effectiveness
- Behavioral analysis precision

---

## 🎯 Advanced Testing

### **Load Testing:**
```bash
# Test multiple concurrent users
ab -n 100 -c 10 http://localhost:8000/
```

### **Stress Testing:**
```bash
# Test with high transaction volume
python stress_test.py
```

### **Penetration Testing:**
- Test SQL injection
- Test XSS attacks
- Test CSRF protection
- Test authentication bypass

---

## 📞 Support

If you encounter issues:
1. Check the server logs
2. Verify database connectivity
3. Test individual API endpoints
4. Review security configuration

**Need Help?** Check the security logs at `/api/v2/admin/logs`

---

## ✅ Security Checklist

- [ ] Session timeout working (10 minutes)
- [ ] Large transactions blocked (>₹10,00,000)
- [ ] Repeated transactions blocked (3+ times)
- [ ] GPS anomaly detection active
- [ ] Behavioral analysis working
- [ ] Transaction limits enforced
- [ ] Security logging active
- [ ] Risk profiling accurate
- [ ] Authentication secure
- [ ] API endpoints protected

**All items checked?** Your security system is working correctly! 🎉 