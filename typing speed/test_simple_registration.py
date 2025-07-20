#!/usr/bin/env python3
"""
Simple Registration Test
Tests registration with correct data structure
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_simple_registration():
    """Test registration with minimal valid data"""
    
    print("👤 Testing Simple Registration...")
    print("=" * 50)
    
    # Generate unique username and email
    timestamp = int(time.time())
    username = f"testuser{timestamp}"
    email = f"testuser{timestamp}@example.com"
    
    # Minimal valid registration data
    registration_data = {
        "name": "Test User",
        "username": username,
        "email": email,
        "password": "testpass123",
        "auth_type": "tap",
        "behavioral_data": {
            "timestamps": [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8]
        },
        "baseline_behavior": {
            "typing_speed": 0.8,
            "tap_speed": 0.7,
            "swipe_pattern": [0.5, 0.6, 0.4, 0.7],
            "GPS": {"lat": 12.9716, "long": 77.5946},
            "nav_pattern": [0.6, 0.5, 0.7, 0.4]
        }
    }
    
    print(f"Username: {username}")
    print(f"Email: {email}")
    print(f"Sending registration data...")
    
    response = requests.post(f"{BASE_URL}/api/v2/register", json=registration_data)
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        print("✅ Registration successful!")
        return username
    else:
        print("❌ Registration failed!")
        return None

def test_login_after_registration(username):
    """Test login after successful registration"""
    
    if not username:
        print("❌ Cannot test login - registration failed")
        return None
    
    print(f"\n🔐 Testing Login for {username}...")
    print("=" * 50)
    
    login_data = {
        "username": username,
        "password": "testpass123",
        "auth_type": "tap",
        "behavioral_data": {
            "timestamps": [1.0, 1.25, 1.52, 1.78, 2.05, 2.32, 2.58, 2.85, 3.12, 3.38]
        }
    }
    
    response = requests.post(f"{BASE_URL}/api/v2/authenticate", json=login_data)
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Login successful!")
        return result.get("access_token")
    else:
        print("❌ Login failed!")
        return None

def test_transfer_after_login(token):
    """Test transfer after successful login"""
    
    if not token:
        print("❌ Cannot test transfer - login failed")
        return
    
    print(f"\n💰 Testing Transfer after Login...")
    print("=" * 50)
    
    headers = {"Authorization": f"Bearer {token}"}
    
    transfer_data = {
        "amount": 1000.0,
        "to_account": "1234567890",
        "description": "Test transfer after registration"
    }
    
    response = requests.post(f"{BASE_URL}/api/v2/banking/transfer", json=transfer_data, headers=headers)
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        print("✅ Transfer successful!")
    else:
        print("❌ Transfer failed!")

def main():
    """Main test function"""
    
    print("🚀 Complete Registration and Transfer Test")
    print("=" * 60)
    
    # Step 1: Register new user
    username = test_simple_registration()
    
    # Step 2: Login with new user
    token = test_login_after_registration(username)
    
    # Step 3: Test transfer
    test_transfer_after_login(token)
    
    print("\n" + "=" * 60)
    print("🎉 Test Complete!")

if __name__ == "__main__":
    main() 