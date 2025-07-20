#!/usr/bin/env python3
"""
Simple Behavioral Analysis Test
Tests the behavioral analysis endpoints individually
"""

import requests
import time
import json

# Configuration
BASE_URL = "http://localhost:8000"

def test_tap_speed_analysis():
    """Test tap speed analysis endpoint"""
    print("🧠 Testing Tap Speed Analysis...")
    
    # Login first
    login_data = {
        "username": "testuser",
        "password": "testpass123",
        "auth_type": "typing",
        "behavioral_data": {
            "timestamps": [time.time() + i * 0.5 for i in range(10)]
        }
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/v2/authenticate", json=login_data)
        if response.status_code != 200:
            print("❌ Login failed")
            return
        
        token = response.json().get("access_token")
        headers = {"Authorization": f"Bearer {token}"}
        
        # Test tap speed analysis
        timestamps = [time.time() + i * 0.5 for i in range(10)]
        
        response = requests.post(f"{BASE_URL}/api/v2/tap-speed/analyze", 
                               json=timestamps, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Tap speed analysis successful")
            print(f"   Anomaly detected: {data.get('user_flagged', 'N/A')}")
            print(f"   Recommended action: {data.get('recommended_action', 'N/A')}")
            return True
        else:
            print(f"❌ Tap speed analysis failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Tap speed analysis error: {e}")
        return False

def test_behavioral_matching():
    """Test behavioral matching endpoint"""
    print("\n🧠 Testing Behavioral Matching...")
    
    # Login first
    login_data = {
        "username": "testuser",
        "password": "testpass123",
        "auth_type": "typing",
        "behavioral_data": {
            "timestamps": [time.time() + i * 0.5 for i in range(10)]
        }
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/v2/authenticate", json=login_data)
        if response.status_code != 200:
            print("❌ Login failed")
            return
        
        token = response.json().get("access_token")
        headers = {"Authorization": f"Bearer {token}"}
        
        # Test behavioral matching
        behavioral_data = {
            "timestamps": [time.time() + i * 0.3 for i in range(5)]
        }
        
        response = requests.post(f"{BASE_URL}/api/v2/behavioral/match", 
                               json=behavioral_data, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Behavioral matching successful")
            print(f"   Match score: {data.get('match_score', 'N/A')}")
            print(f"   Behavioral mismatch: {data.get('behavioral_mismatch', 'N/A')}")
            print(f"   Recommended action: {data.get('recommended_action', 'N/A')}")
            return True
        else:
            print(f"❌ Behavioral matching failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Behavioral matching error: {e}")
        return False

def main():
    """Main test function"""
    print("🧠 Behavioral Analysis Test")
    print("=" * 40)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code != 200:
            print("❌ Server not responding. Please start your banking server first.")
            return
    except:
        print("❌ Cannot connect to server. Please start your banking server first.")
        return
    
    # Run tests
    tap_success = test_tap_speed_analysis()
    match_success = test_behavioral_matching()
    
    print("\n" + "=" * 40)
    print("📋 Test Results:")
    print(f"   Tap Speed Analysis: {'✅ Working' if tap_success else '❌ Failed'}")
    print(f"   Behavioral Matching: {'✅ Working' if match_success else '❌ Failed'}")
    
    if tap_success and match_success:
        print("\n🎉 All behavioral analysis tests passed!")
    else:
        print("\n⚠️ Some behavioral analysis tests failed.")

if __name__ == "__main__":
    main() 