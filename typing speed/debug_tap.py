#!/usr/bin/env python3
"""
Debug Tap Speed Analysis
Simple script to debug the 500 error in tap speed analysis
"""

import requests
import time
import json

# Configuration
BASE_URL = "http://localhost:8000"

def debug_tap_speed():
    """Debug tap speed analysis endpoint"""
    print("🔍 Debugging Tap Speed Analysis...")
    
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
        
        # Test with simple timestamps
        print("   Testing with simple timestamps...")
        timestamps = [time.time() + i * 0.5 for i in range(5)]
        print(f"   Timestamps: {timestamps}")
        
        response = requests.post(f"{BASE_URL}/api/v2/tap-speed/analyze", 
                               json=timestamps, headers=headers)
        
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Tap speed analysis successful")
            print(f"   Data: {json.dumps(data, indent=2)}")
        else:
            print(f"❌ Tap speed analysis failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_simple_endpoint():
    """Test a simple endpoint to make sure server is working"""
    print("\n🔍 Testing Simple Endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {response.text[:100]}...")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Main debug function"""
    print("🔍 Tap Speed Analysis Debug")
    print("=" * 40)
    
    test_simple_endpoint()
    debug_tap_speed()

if __name__ == "__main__":
    main() 