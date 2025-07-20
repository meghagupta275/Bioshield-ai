#!/usr/bin/env python3
"""
Test Transfer Limits via Web Interface
Tests the transfer limits through the web dashboard
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_web_interface():
    """Test transfer limits through web interface"""
    
    print("🌐 Testing Transfer Limits via Web Interface...")
    print("=" * 60)
    
    # Step 1: Check if web interface is accessible
    print("📱 Step 1: Checking Web Interface...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Home page status: {response.status_code}")
    
    response = requests.get(f"{BASE_URL}/login")
    print(f"Login page status: {response.status_code}")
    
    response = requests.get(f"{BASE_URL}/dashboard")
    print(f"Dashboard page status: {response.status_code}")
    
    # Step 2: Test transfer endpoint directly with session
    print("\n💰 Step 2: Testing Transfer Endpoint...")
    
    # Create a session to maintain cookies
    session = requests.Session()
    
    # Test transfer without authentication
    transfer_data = {
        "amount": 1000.0,
        "to_account": "1234567890",
        "description": "Web interface test"
    }
    
    response = session.post(f"{BASE_URL}/api/v2/banking/transfer", json=transfer_data)
    print(f"Transfer without auth status: {response.status_code}")
    print(f"Response: {response.text}")
    
    # Step 3: Test with different amounts
    print("\n💰 Step 3: Testing Different Amounts...")
    
    test_amounts = [
        {"amount": 1000.0, "description": "Small amount"},
        {"amount": 50000.0, "description": "Medium amount"},
        {"amount": 100000.0, "description": "Large amount"},
        {"amount": 2000000.0, "description": "Very large amount"}
    ]
    
    for test in test_amounts:
        transfer_data = {
            "amount": test["amount"],
            "to_account": "1234567890",
            "description": test["description"]
        }
        
        response = session.post(f"{BASE_URL}/api/v2/banking/transfer", json=transfer_data)
        print(f"Amount ₹{test['amount']:,} - Status: {response.status_code}")
        
        if response.status_code == 403:
            try:
                result = response.json()
                if "reason" in result:
                    print(f"  Reason: {result['reason']}")
                if "message" in result:
                    print(f"  Message: {result['message']}")
            except:
                pass

def show_web_testing_guide():
    """Show guide for testing via web interface"""
    
    print("\n📋 WEB INTERFACE TESTING GUIDE")
    print("=" * 60)
    print("🌐 Open your browser and navigate to:")
    print("   http://127.0.0.1:8000")
    print()
    print("🔐 Login Steps:")
    print("1. Click 'Login' or go to http://127.0.0.1:8000/login")
    print("2. Use any existing credentials")
    print("3. Complete behavioral authentication")
    print()
    print("💰 Test Transfer Limits:")
    print("1. Go to the banking dashboard")
    print("2. Try making multiple transfers quickly")
    print("3. Try making transfers with large amounts")
    print("4. Watch for auto-logout messages")
    print()
    print("📊 Expected Behaviors:")
    print("- After 5 transfers in 1 hour: Auto-logout")
    print("- After ₹50,000 cumulative: Auto-logout")
    print("- Clear error messages displayed")
    print("- Session automatically terminated")
    print()
    print("🎯 Test Scenarios:")
    print("1. Make 6 small transfers quickly")
    print("2. Make transfers totaling ₹50,000+")
    print("3. Try large single transfers (₹100,000+)")
    print("4. Check security logs after violations")

def test_limits_endpoint():
    """Test the limits endpoint"""
    
    print("\n📊 Testing Limits Endpoint...")
    print("=" * 60)
    
    # Test without authentication
    response = requests.get(f"{BASE_URL}/api/v2/banking/limits")
    print(f"Limits endpoint status: {response.status_code}")
    print(f"Response: {response.text}")
    
    # Test with invalid data
    print("\n🔍 Testing with invalid data...")
    
    invalid_data = [
        {"amount": 0, "description": "Zero amount"},
        {"amount": -100, "description": "Negative amount"},
        {"description": "Missing amount"},
        {"amount": "1000", "description": "String amount"}
    ]
    
    for data in invalid_data:
        response = requests.post(f"{BASE_URL}/api/v2/banking/transfer", json=data)
        print(f"{data['description']} - Status: {response.status_code}")

if __name__ == "__main__":
    test_web_interface()
    test_limits_endpoint()
    show_web_testing_guide() 