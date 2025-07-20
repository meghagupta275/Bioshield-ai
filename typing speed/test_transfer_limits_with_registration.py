#!/usr/bin/env python3
"""
Test Transfer Limits with User Registration
Creates a new user with realistic behavioral data and tests transfer limits
"""

import requests
import json
import time
import random

BASE_URL = "http://127.0.0.1:8000"

def generate_realistic_tap_timestamps():
    """Generate realistic tap timestamps with natural variation"""
    timestamps = []
    current_time = 1000  # Start at 1 second
    
    for i in range(10):
        # Add natural variation: 200-300ms between taps
        variation = random.uniform(200, 300)
        current_time += variation
        timestamps.append(current_time)
    
    return timestamps

def register_new_user():
    """Register a new user with realistic behavioral data"""
    
    print("👤 Registering New User...")
    
    # Generate realistic behavioral data
    tap_timestamps = generate_realistic_tap_timestamps()
    
    registration_data = {
        "name": "Transfer Test User",
        "username": f"transfertest{int(time.time())}",
        "email": f"transfertest{int(time.time())}@example.com",
        "password": "testpass123",
        "auth_type": "tap",
        "behavioral_data": {
            "timestamps": tap_timestamps
        },
        "baseline_behavior": {
            "typing_speed": 0.8,
            "tap_speed": 0.7,
            "swipe_pattern": [0.5, 0.6, 0.4, 0.7],
            "GPS": {"lat": 12.9716, "long": 77.5946},
            "nav_pattern": [0.6, 0.5, 0.7, 0.4]
        }
    }
    
    response = requests.post(f"{BASE_URL}/api/v2/register", json=registration_data)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ User registered successfully!")
        print(f"Username: {registration_data['username']}")
        return registration_data['username']
    else:
        print(f"❌ Registration failed: {response.status_code}")
        print(f"Response: {response.text}")
        return None

def login_user(username):
    """Login with the registered user"""
    
    print(f"🔐 Logging in as {username}...")
    
    # Generate similar but slightly different timestamps for login
    tap_timestamps = generate_realistic_tap_timestamps()
    
    login_data = {
        "username": username,
        "password": "testpass123",
        "auth_type": "tap",
        "behavioral_data": {
            "timestamps": tap_timestamps
        }
    }
    
    response = requests.post(f"{BASE_URL}/api/v2/authenticate", json=login_data)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Login successful!")
        return result.get("access_token")
    else:
        print(f"❌ Login failed: {response.status_code}")
        print(f"Response: {response.text}")
        return None

def test_transfer_limits(token):
    """Test transfer frequency and cumulative amount limits"""
    
    print("\n💰 Testing Transfer Limits...")
    print("=" * 60)
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Step 1: Test transfer frequency limit
    print("📊 Step 1: Testing Transfer Frequency Limit...")
    print("Making 6 transfers (limit is 5 per hour)...")
    
    for i in range(6):
        transfer_data = {
            "amount": 1000.0,
            "to_account": f"123456789{i}",
            "description": f"Test transfer {i+1}"
        }
        
        response = requests.post(f"{BASE_URL}/api/v2/banking/transfer", json=transfer_data, headers=headers)
        
        print(f"Transfer {i+1} Status: {response.status_code}")
        
        if response.status_code == 403:
            result = response.json()
            if result.get("reason") == "transfer_frequency_limit":
                print("✅ Transfer frequency limit correctly triggered!")
                print(f"Message: {result.get('message')}")
                print(f"Action: {result.get('action')}")
                print(f"Reason: {result.get('reason')}")
                return
        elif response.status_code == 200:
            print(f"✅ Transfer {i+1} successful")
        else:
            print(f"⚠️ Unexpected response: {response.text}")
    
    # Step 2: Test cumulative amount limit (if frequency limit wasn't triggered)
    print("\n💰 Step 2: Testing Cumulative Amount Limit...")
    
    # Login again for fresh session
    username = f"transfertest{int(time.time())}"
    token = login_user(username)
    if not token:
        return
    
    headers = {"Authorization": f"Bearer {token}"}
    
    print("Making transfers to reach ₹50,000 limit...")
    
    cumulative_amount = 0
    transfer_count = 0
    
    while cumulative_amount < 50000 and transfer_count < 10:
        transfer_amount = min(10000, 50000 - cumulative_amount)
        transfer_data = {
            "amount": float(transfer_amount),
            "to_account": f"123456789{transfer_count}",
            "description": f"Cumulative test transfer {transfer_count+1}"
        }
        
        response = requests.post(f"{BASE_URL}/api/v2/banking/transfer", json=transfer_data, headers=headers)
        
        if response.status_code == 200:
            cumulative_amount += transfer_amount
            transfer_count += 1
            print(f"✅ Transfer {transfer_count}: ₹{transfer_amount:,} (Total: ₹{cumulative_amount:,})")
        else:
            print(f"❌ Transfer failed: {response.text}")
            break
    
    # Try one more transfer that should exceed the limit
    print(f"\n💰 Attempting transfer that exceeds ₹50,000 limit...")
    transfer_data = {
        "amount": 1000.0,
        "to_account": "9999999999",
        "description": "Transfer exceeding cumulative limit"
    }
    
    response = requests.post(f"{BASE_URL}/api/v2/banking/transfer", json=transfer_data, headers=headers)
    
    print(f"Exceeding Limit Transfer Status: {response.status_code}")
    if response.status_code == 403:
        result = response.json()
        if result.get("reason") == "transfer_cumulative_limit":
            print("✅ Cumulative amount limit correctly triggered!")
            print(f"Message: {result.get('message')}")
            print(f"Action: {result.get('action')}")
            print(f"Reason: {result.get('reason')}")
        else:
            print(f"⚠️ Different limit triggered: {result.get('reason')}")
    else:
        print(f"⚠️ Unexpected response: {response.text}")

def test_limits_api(token):
    """Test the limits API endpoint"""
    
    print("\n📊 Testing Limits API...")
    print("=" * 60)
    
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}/api/v2/banking/limits", headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Limits API Response:")
        print(json.dumps(result, indent=2))
    else:
        print(f"❌ Limits API failed: {response.status_code}")
        print(f"Response: {response.text}")

def main():
    """Main test function"""
    
    print("💰 Testing Transfer Frequency and Cumulative Amount Limits...")
    print("=" * 70)
    
    # Step 1: Register new user
    username = register_new_user()
    if not username:
        print("❌ Cannot proceed without user registration")
        return
    
    # Step 2: Login
    token = login_user(username)
    if not token:
        print("❌ Cannot proceed without login")
        return
    
    # Step 3: Test limits API
    test_limits_api(token)
    
    # Step 4: Test transfer limits
    test_transfer_limits(token)
    
    print("\n" + "=" * 70)
    print("💰 Transfer Limits Test Complete!")

if __name__ == "__main__":
    main() 