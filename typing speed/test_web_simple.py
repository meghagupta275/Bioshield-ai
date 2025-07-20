#!/usr/bin/env python3
"""
Simple Web Interface Test
Tests the web interface with a large transfer to show security violation
"""

import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_web_interface():
    """Test web interface with large transfer"""
    
    print("🌐 Testing Web Interface with Large Transfer...")
    print("=" * 60)
    
    # Step 1: Try to access the dashboard
    print("🔍 Step 1: Accessing Dashboard...")
    response = requests.get(f"{BASE_URL}/dashboard")
    print(f"Dashboard Status: {response.status_code}")
    
    if response.status_code == 200:
        print("✅ Dashboard accessible")
    else:
        print(f"❌ Dashboard not accessible: {response.status_code}")
    
    # Step 2: Try to access login page
    print("\n🔐 Step 2: Accessing Login Page...")
    response = requests.get(f"{BASE_URL}/login")
    print(f"Login Page Status: {response.status_code}")
    
    if response.status_code == 200:
        print("✅ Login page accessible")
    else:
        print(f"❌ Login page not accessible: {response.status_code}")
    
    # Step 3: Test transfer endpoint directly
    print("\n💰 Step 3: Testing Transfer Endpoint...")
    transfer_data = {
        "amount": 2000000.0,  # 20 lakhs (must be float)
        "to_account": "1234567890",
        "description": "Test large transfer from web"
    }
    
    # Try without authentication first
    response = requests.post(f"{BASE_URL}/api/v2/banking/transfer", json=transfer_data)
    print(f"Transfer without auth: {response.status_code}")
    
    if response.status_code == 401:
        print("✅ Properly requires authentication")
    else:
        print(f"⚠️  Unexpected response: {response.status_code}")
    
    print("\n" + "=" * 60)
    print("🌐 Web Interface Test Complete!")
    print("\n📝 Instructions for Testing:")
    print("1. Open browser and go to: http://127.0.0.1:8000")
    print("2. Login with any credentials")
    print("3. Try to make a transfer of ₹20,00,000")
    print("4. You should see a security violation message")
    print("5. You should be automatically logged out")

if __name__ == "__main__":
    test_web_interface() 