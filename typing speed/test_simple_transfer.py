#!/usr/bin/env python3
"""
Simple Transfer Test
Tests the transfer endpoint with simple authentication
"""

import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_simple_transfer():
    """Test transfer with simple authentication"""
    
    print("💰 Testing Simple Transfer...")
    print("=" * 50)
    
    # Step 1: Try to login with simple credentials
    print("🔐 Step 1: Logging in...")
    login_data = {
        "username": "admin",
        "password": "admin123",
        "auth_type": "tap",
        "behavioral_data": {
            "timestamps": [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800]
        }
    }
    
    response = requests.post(f"{BASE_URL}/api/v2/authenticate", json=login_data)
    print(f"Login Status: {response.status_code}")
    print(f"Login Response: {response.text}")
    
    if response.status_code != 200:
        print("❌ Login failed, trying alternative...")
        
        # Try with different behavioral data
        login_data["behavioral_data"]["timestamps"] = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        response = requests.post(f"{BASE_URL}/api/v2/authenticate", json=login_data)
        print(f"Alternative Login Status: {response.status_code}")
        print(f"Alternative Login Response: {response.text}")
        
        if response.status_code != 200:
            print("❌ All login attempts failed")
            return
    
    login_result = response.json()
    token = login_result.get("access_token")
    print("✅ Login successful")
    
    # Step 2: Test large transfer
    print("\n💰 Step 2: Testing Large Transfer (₹20,00,000)...")
    transfer_data = {
        "amount": 2000000.0,  # Must be float
        "to_account": "1234567890",
        "description": "Test large transfer"
    }
    
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(f"{BASE_URL}/api/v2/banking/transfer", json=transfer_data, headers=headers)
    
    print(f"Large Transfer Status: {response.status_code}")
    print(f"Large Transfer Response: {response.text}")
    
    try:
        json_response = response.json()
        print(f"Large Transfer JSON: {json.dumps(json_response, indent=2)}")
        
        if response.status_code == 403 and json_response.get("status") == "security_violation":
            print("✅ Security violation correctly detected!")
            print(f"Message: {json_response.get('message')}")
            print(f"Action: {json_response.get('action')}")
            print(f"Reason: {json_response.get('reason')}")
        else:
            print("⚠️  Unexpected response for large transfer")
            
    except Exception as e:
        print(f"❌ Error parsing large transfer response: {e}")
    
    # Step 3: Test small transfer
    print("\n💰 Step 3: Testing Small Transfer (₹1,000)...")
    transfer_data = {
        "amount": 1000.0,  # Must be float
        "to_account": "1234567890",
        "description": "Test small transfer"
    }
    
    response = requests.post(f"{BASE_URL}/api/v2/banking/transfer", json=transfer_data, headers=headers)
    
    print(f"Small Transfer Status: {response.status_code}")
    print(f"Small Transfer Response: {response.text}")
    
    try:
        json_response = response.json()
        print(f"Small Transfer JSON: {json.dumps(json_response, indent=2)}")
        
        if response.status_code == 200:
            print("✅ Small transfer successful!")
        else:
            print("⚠️  Small transfer failed")
            
    except Exception as e:
        print(f"❌ Error parsing small transfer response: {e}")
    
    print("\n" + "=" * 50)
    print("💰 Simple Transfer Test Complete!")

if __name__ == "__main__":
    test_simple_transfer() 