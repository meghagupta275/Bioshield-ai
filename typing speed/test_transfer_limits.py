#!/usr/bin/env python3
"""
Test Transfer Frequency and Cumulative Amount Limits
Demonstrates the new security features that log out users for excessive transfers
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_transfer_limits():
    """Test transfer frequency and cumulative amount limits"""
    
    print("💰 Testing Transfer Frequency and Cumulative Amount Limits...")
    print("=" * 70)
    
    # Step 1: Login
    print("🔐 Step 1: Login...")
    login_data = {
        "username": "testuser",
        "password": "testpass123",
        "auth_type": "tap",
        "behavioral_data": {
            "timestamps": [1000, 1250, 1520, 1780, 2050, 2320, 2580, 2850, 3120, 3380]
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
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Step 2: Test multiple small transfers (should trigger frequency limit)
    print("\n💰 Step 2: Testing Transfer Frequency Limit...")
    print("Making 6 transfers (limit is 5 per hour)...")
    
    for i in range(6):
        transfer_data = {
            "amount": 1000.0,  # ₹1,000 each (must be float)
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
                break
        elif response.status_code == 200:
            print(f"✅ Transfer {i+1} successful")
        else:
            print(f"⚠️ Unexpected response: {response.text}")
    
    # Step 3: Test cumulative amount limit
    print("\n💰 Step 3: Testing Cumulative Amount Limit...")
    print("Making transfers to reach ₹50,000 limit...")
    
    # Login again for fresh session
    login_data_fresh = {
        "username": "testuser",
        "password": "testpass123",
        "auth_type": "tap",
        "behavioral_data": {
            "timestamps": [1000, 1280, 1560, 1840, 2120, 2400, 2680, 2960, 3240, 3520]
        }
    }
    response = requests.post(f"{BASE_URL}/api/v2/authenticate", json=login_data_fresh)
    if response.status_code == 200:
        login_result = response.json()
        token = login_result.get("access_token")
        headers = {"Authorization": f"Bearer {token}"}
        print("✅ Re-login successful")
    
    cumulative_amount = 0
    transfer_count = 0
    
    while cumulative_amount < 50000 and transfer_count < 10:
        transfer_amount = min(10000, 50000 - cumulative_amount)  # Don't exceed limit
        transfer_data = {
            "amount": float(transfer_amount),  # Must be float
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
    
    # Now try one more transfer that should exceed the limit
    print(f"\n💰 Attempting transfer that exceeds ₹50,000 limit...")
    transfer_data = {
        "amount": 1000.0,  # Must be float
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
    
    print("\n" + "=" * 70)
    print("💰 Transfer Limits Test Complete!")

def show_limits_summary():
    """Show a summary of the transfer limits"""
    
    print("\n📋 TRANSFER LIMITS SUMMARY")
    print("=" * 50)
    print("🕐 Time Window: 1 hour (rolling)")
    print()
    print("📊 Frequency Limits:")
    print("- Maximum transfers per hour: 5")
    print("- Exceeding this triggers auto-logout")
    print()
    print("💰 Amount Limits:")
    print("- Maximum cumulative amount per hour: ₹50,000")
    print("- Exceeding this triggers auto-logout")
    print()
    print("🔒 Security Actions:")
    print("- User automatically logged out")
    print("- Security event logged")
    print("- Clear error message displayed")
    print("- Session token invalidated")
    print()
    print("🎯 Use Cases:")
    print("- Prevents rapid-fire transfers")
    print("- Stops large cumulative withdrawals")
    print("- Protects against automated attacks")
    print("- Maintains transaction velocity limits")

if __name__ == "__main__":
    test_transfer_limits()
    show_limits_summary() 