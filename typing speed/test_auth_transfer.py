#!/usr/bin/env python3
"""
Test Authenticated Transfer
Tests the transfer endpoint with authentication to show security violations
"""

import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_authenticated_transfer():
    """Test transfer with authentication"""
    
    print("🔐 Testing Authenticated Transfer...")
    print("=" * 50)
    
    # Step 1: Register a test user
    print("📝 Step 1: Registering test user...")
    register_data = {
        "name": "Test User",
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpass123",
        "auth_type": "tap",
        "behavioral_data": {
            "timestamps": [1000, 1200, 1400, 1600, 1800]
        },
        "baseline_behavior": {
            "typing_speed": 0.3,
            "tap_pattern": [0.2, 0.3, 0.2, 0.3]
        }
    }
    
    response = requests.post(f"{BASE_URL}/api/v2/register", json=register_data)
    print(f"Registration Status: {response.status_code}")
    
    # Step 2: Login
    print("\n🔐 Step 2: Logging in...")
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
        print(f"Response: {response.text}")
        return
    
    login_result = response.json()
    token = login_result.get("access_token")
    print("✅ Login successful")
    
    # Step 3: Test large transfer (should trigger security violation)
    print("\n💰 Step 3: Testing Large Transfer...")
    transfer_data = {
        "amount": 2000000.0,  # 20 lakhs (must be float)
        "to_account": "1234567890",
        "description": "Test large transfer"
    }
    
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(f"{BASE_URL}/api/v2/banking/transfer", json=transfer_data, headers=headers)
    
    print(f"Transfer Status: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    print(f"Response: {response.text}")
    
    try:
        json_response = response.json()
        print(f"JSON Response: {json.dumps(json_response, indent=2)}")
        
        if response.status_code == 403 and json_response.get("status") == "security_violation":
            print("✅ Security violation detected!")
            print(f"Message: {json_response.get('message')}")
            print(f"Action: {json_response.get('action')}")
            print(f"Reason: {json_response.get('reason')}")
        else:
            print("⚠️  Unexpected response")
            
    except Exception as e:
        print(f"❌ Error parsing response: {e}")
    
    # Step 4: Test small transfer (should succeed)
    print("\n💰 Step 4: Testing Small Transfer...")
    transfer_data = {
        "amount": 1000,  # 1 thousand
        "to_account": "1234567890",
        "description": "Test small transfer"
    }
    
    response = requests.post(f"{BASE_URL}/api/v2/banking/transfer", json=transfer_data, headers=headers)
    
    print(f"Transfer Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    try:
        json_response = response.json()
        print(f"JSON Response: {json.dumps(json_response, indent=2)}")
        
        if response.status_code == 200:
            print("✅ Small transfer successful!")
        else:
            print("⚠️  Small transfer failed")
            
    except Exception as e:
        print(f"❌ Error parsing response: {e}")
    
    print("\n" + "=" * 50)
    print("🔐 Authenticated Transfer Test Complete!")

if __name__ == "__main__":
    test_authenticated_transfer() 