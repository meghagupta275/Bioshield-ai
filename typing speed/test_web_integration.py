#!/usr/bin/env python3
"""
Test Web Interface Integration
Tests that the web interface properly handles security violations
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_web_transfer_integration():
    """Test that web interface properly handles security violations"""
    
    print("🌐 Testing Web Interface Integration...")
    print("=" * 50)
    
    # Step 1: Login
    print("🔐 Step 1: Login...")
    login_data = {
        "username": "testuser",
        "password": "testpass123",
        "auth_type": "tap",
        "behavioral_data": {
            "timestamps": [1000, 1200, 1400, 1600, 1800]
        }
    }
    
    response = requests.post(f"{BASE_URL}/api/v2/authenticate", json=login_data)
    if response.status_code != 200:
        print(f"❌ Login failed: {response.status_code}")
        return
    
    login_result = response.json()
    token = login_result.get("access_token")
    print("✅ Login successful")
    
    # Step 2: Test large transaction (should trigger security violation)
    print("\n💰 Step 2: Test Large Transaction...")
    transfer_data = {
        "amount": 2000000.0,  # 20 lakhs (must be float)
        "to_account": "1234567890",
        "description": "Test large transfer"
    }
    
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(f"{BASE_URL}/api/v2/banking/transfer", json=transfer_data, headers=headers)
    
    print(f"Response Status: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    
    if response.status_code == 403:
        result = response.json()
        print("✅ Security violation detected!")
        print(f"Status: {result.get('status')}")
        print(f"Message: {result.get('message')}")
        print(f"Action: {result.get('action')}")
        print(f"Reason: {result.get('reason')}")
        
        # Check if Set-Cookie header is present (for logout)
        if 'set-cookie' in response.headers:
            print("✅ Auto-logout cookie set")
        else:
            print("⚠️  No logout cookie found")
            
    else:
        print(f"❌ Expected 403, got {response.status_code}")
        print(f"Response: {response.text}")
    
    # Step 3: Test session after security violation
    print("\n🔍 Step 3: Test Session After Security Violation...")
    response = requests.get(f"{BASE_URL}/api/v2/session/status", headers=headers)
    print(f"Session Status: {response.status_code}")
    
    if response.status_code == 401:
        print("✅ Session properly invalidated")
    else:
        print("⚠️  Session still active")
    
    print("\n" + "=" * 50)
    print("🌐 Web Interface Integration Test Complete!")

if __name__ == "__main__":
    test_web_transfer_integration() 