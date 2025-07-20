#!/usr/bin/env python3
"""
Create Test User Script
Creates a test user for security testing
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:8000"

def create_test_user():
    """Create a test user for security testing"""
    print("👤 Creating Test User for Security Testing")
    print("=" * 50)
    
    # Test user data
    test_user = {
        "name": "Test User",  # Changed from full_name to name
        "username": "testuser",
        "password": "testpass123",
        "email": "test@example.com",
        "auth_type": "typing",
        "behavioral_data": {
            "timestamps": [time.time() + i * 0.5 for i in range(10)]  # Add timestamps for typing pattern
        },
        "baseline_behavior": {  # Add baseline_behavior field
            "typing_speed": 0.5,
            "confidence": 0.8,
            "tap_speed": 0.3,
            "typing_pattern": [0.5, 0.6, 0.4, 0.7, 0.5, 0.6, 0.4, 0.7, 0.5, 0.6]
        }
    }
    
    try:
        # Register the test user
        print("📝 Registering test user...")
        response = requests.post(f"{BASE_URL}/api/v2/register", json=test_user)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Test user created successfully!")
            print(f"   Username: {test_user['username']}")
            print(f"   Password: {test_user['password']}")
            print(f"   User ID: {data.get('user_id', 'N/A')}")
            print(f"   Message: {data.get('message', 'N/A')}")
            return True
        elif response.status_code == 400:
            data = response.json()
            if "already exists" in data.get('detail', '').lower():
                print("ℹ️ Test user already exists!")
                print(f"   Username: {test_user['username']}")
                print(f"   Password: {test_user['password']}")
                return True
            else:
                print(f"❌ Registration failed: {data.get('detail', 'Unknown error')}")
                return False
        else:
            print(f"❌ Registration failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error creating test user: {e}")
        return False

def test_login():
    """Test login with the created user"""
    print("\n🔐 Testing Login...")
    
    login_data = {
        "username": "testuser",
        "password": "testpass123",
        "auth_type": "typing",
        "behavioral_data": {
            "typing_speed": 0.5,
            "confidence": 0.8
        }
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/v2/authenticate", json=login_data)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Login successful!")
            print(f"   User ID: {data.get('user_id', 'N/A')}")
            print(f"   Token: {data.get('access_token', 'N/A')[:20]}...")
            return True
        else:
            print(f"❌ Login failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Login error: {e}")
        return False

def main():
    """Main function"""
    print("🔒 Test User Setup for Security Testing")
    print("=" * 50)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code != 200:
            print("❌ Server not responding. Please start your banking server first.")
            print("   Command: uvicorn banking_auth_app:app --reload")
            return
    except:
        print("❌ Cannot connect to server. Please start your banking server first.")
        print("   Command: uvicorn banking_auth_app:app --reload")
        return
    
    # Create test user
    if create_test_user():
        # Test login
        if test_login():
            print("\n" + "=" * 50)
            print("✅ Test User Setup Complete!")
            print("\n📋 Test User Credentials:")
            print("   Username: testuser")
            print("   Password: testpass123")
            print("   Auth Type: typing")
            print("\n🚀 You can now run the security test:")
            print("   python security_test.py")
        else:
            print("\n❌ Login test failed. Please check the server logs.")
    else:
        print("\n❌ Test user creation failed. Please check the server logs.")

if __name__ == "__main__":
    main() 