#!/usr/bin/env python3
"""
Test Typing Speed Anomaly in Web Interface
Demonstrates how to test typing speed anomaly detection
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_typing_anomaly_web():
    """Test typing speed anomaly detection in web interface"""
    
    print("⌨️ Testing Typing Speed Anomaly Detection...")
    print("=" * 60)
    
    # Step 1: Login with normal typing pattern
    print("🔐 Step 1: Login with Normal Typing Pattern...")
    login_data = {
        "username": "testuser",
        "password": "testpass123",
        "auth_type": "tap",
        "behavioral_data": {
            "timestamps": [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800]
        }
    }
    
    response = requests.post(f"{BASE_URL}/api/v2/authenticate", json=login_data)
    if response.status_code != 200:
        print(f"❌ Login failed: {response.status_code}")
        print(f"Response: {response.text}")
        return
    
    login_result = response.json()
    token = login_result.get("access_token")
    print("✅ Login successful with normal pattern")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Step 2: Test normal typing speed (should pass)
    print("\n⌨️ Step 2: Test Normal Typing Speed...")
    normal_timestamps = [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800]
    
    response = requests.post(
        f"{BASE_URL}/api/v2/tap-speed/analyze",
        json=normal_timestamps,
        headers=headers
    )
    
    print(f"Normal Typing Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print("✅ Normal typing pattern accepted")
        print(f"Anomaly Score: {result.get('tap_analysis', {}).get('anomaly_score', 'N/A')}")
    else:
        print(f"❌ Normal typing failed: {response.text}")
    
    # Step 3: Test suspicious typing speed (should trigger anomaly)
    print("\n⌨️ Step 3: Test Suspicious Typing Speed...")
    suspicious_timestamps = [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009]  # Too fast
    
    response = requests.post(
        f"{BASE_URL}/api/v2/tap-speed/analyze",
        json=suspicious_timestamps,
        headers=headers
    )
    
    print(f"Suspicious Typing Status: {response.status_code}")
    if response.status_code == 403:
        result = response.json()
        print("✅ Suspicious typing pattern detected!")
        print(f"Status: {result.get('status')}")
        print(f"Message: {result.get('message')}")
        print(f"Action: {result.get('action')}")
        print(f"Reason: {result.get('reason')}")
    else:
        print(f"⚠️ Unexpected response: {response.text}")
    
    # Step 4: Test machine-like typing pattern
    print("\n🤖 Step 4: Test Machine-like Typing Pattern...")
    machine_timestamps = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]  # Perfect timing
    
    response = requests.post(
        f"{BASE_URL}/api/v2/tap-speed/analyze",
        json=machine_timestamps,
        headers=headers
    )
    
    print(f"Machine-like Typing Status: {response.status_code}")
    if response.status_code == 403:
        result = response.json()
        print("✅ Machine-like pattern detected!")
        print(f"Status: {result.get('status')}")
        print(f"Message: {result.get('message')}")
        print(f"Action: {result.get('action')}")
        print(f"Reason: {result.get('reason')}")
    else:
        print(f"⚠️ Unexpected response: {response.text}")
    
    print("\n" + "=" * 60)
    print("⌨️ Typing Speed Anomaly Test Complete!")

def show_web_testing_guide():
    """Show how to test typing speed anomaly in web browser"""
    
    print("\n🌐 WEB INTERFACE TESTING GUIDE")
    print("=" * 60)
    print("To test typing speed anomaly detection in the web interface:")
    print()
    print("1. 🌐 Open your browser and go to: http://127.0.0.1:8000")
    print("2. 🔐 Login to the banking dashboard")
    print("3. ⌨️ Look for typing/tap speed analysis features")
    print("4. 🧪 Test different typing patterns:")
    print()
    print("   NORMAL PATTERN (should pass):")
    print("   - Type naturally with 200ms intervals")
    print("   - Example: [1000, 1200, 1400, 1600, 1800]")
    print()
    print("   SUSPICIOUS PATTERN (should trigger anomaly):")
    print("   - Type too fast with 1ms intervals")
    print("   - Example: [1000, 1001, 1002, 1003, 1004]")
    print()
    print("   MACHINE-LIKE PATTERN (should trigger severe anomaly):")
    print("   - Type with perfect timing")
    print("   - Example: [1000, 1000, 1000, 1000, 1000]")
    print()
    print("5. 🔍 Watch for security alerts and auto-logout")
    print("6. 📝 Check browser console for debug information")
    print()
    print("Expected Results:")
    print("- Normal typing: ✅ Accepted")
    print("- Suspicious typing: ⚠️ Warning or flag")
    print("- Machine-like typing: 🚫 Auto-logout")

if __name__ == "__main__":
    test_typing_anomaly_web()
    show_web_testing_guide() 