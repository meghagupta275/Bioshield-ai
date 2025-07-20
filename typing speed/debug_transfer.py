#!/usr/bin/env python3
"""
Debug Transfer Endpoint
Tests the transfer endpoint and shows exactly what response it returns
"""

import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def debug_transfer():
    """Debug the transfer endpoint"""
    
    print("🔍 Debugging Transfer Endpoint...")
    print("=" * 50)
    
    # Test 1: Large transfer without authentication
    print("💰 Test 1: Large Transfer (No Auth)...")
    transfer_data = {
        "amount": 2000000.0,  # 20 lakhs (must be float)
        "to_account": "1234567890",
        "description": "Debug test transfer"
    }
    
    response = requests.post(f"{BASE_URL}/api/v2/banking/transfer", json=transfer_data)
    print(f"Status Code: {response.status_code}")
    print(f"Headers: {dict(response.headers)}")
    print(f"Response: {response.text}")
    
    try:
        json_response = response.json()
        print(f"JSON Response: {json.dumps(json_response, indent=2)}")
    except:
        print("Not JSON response")
    
    print("\n" + "-" * 50)
    
    # Test 2: Small transfer without authentication
    print("💰 Test 2: Small Transfer (No Auth)...")
    transfer_data = {
        "amount": 1000.0,  # 1 thousand (must be float)
        "to_account": "1234567890",
        "description": "Debug small transfer"
    }
    
    response = requests.post(f"{BASE_URL}/api/v2/banking/transfer", json=transfer_data)
    print(f"Status Code: {response.status_code}")
    print(f"Headers: {dict(response.headers)}")
    print(f"Response: {response.text}")
    
    try:
        json_response = response.json()
        print(f"JSON Response: {json.dumps(json_response, indent=2)}")
    except:
        print("Not JSON response")
    
    print("\n" + "=" * 50)
    print("🔍 Transfer Endpoint Debug Complete!")

if __name__ == "__main__":
    debug_transfer() 