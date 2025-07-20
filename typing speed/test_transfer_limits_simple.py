#!/usr/bin/env python3
"""
Simple Test Transfer Limits (No Login Required)
Tests the transfer endpoint validation and limits
"""

import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_transfer_validation():
    """Test transfer endpoint validation without authentication"""
    
    print("💰 Testing Transfer Validation (No Auth)...")
    print("=" * 60)
    
    # Test 1: Valid transfer data format
    print("✅ Test 1: Valid Transfer Data Format...")
    valid_data = {
        "amount": 1000.0,
        "to_account": "1234567890",
        "description": "Test transfer"
    }
    
    response = requests.post(f"{BASE_URL}/api/v2/banking/transfer", json=valid_data)
    print(f"Status: {response.status_code}")
    print(f"Expected: 403 (Not authenticated) - Data format is correct")
    print(f"Response: {response.text}")
    
    # Test 2: Invalid amount (0)
    print("\n❌ Test 2: Invalid Amount (0)...")
    invalid_amount = {
        "amount": 0,
        "to_account": "1234567890",
        "description": "Test transfer"
    }
    
    response = requests.post(f"{BASE_URL}/api/v2/banking/transfer", json=invalid_amount)
    print(f"Status: {response.status_code}")
    print(f"Expected: 422 (Validation error) - Amount must be > 0")
    print(f"Response: {response.text}")
    
    # Test 3: Invalid amount (negative)
    print("\n❌ Test 3: Invalid Amount (negative)...")
    negative_amount = {
        "amount": -100.0,
        "to_account": "1234567890",
        "description": "Test transfer"
    }
    
    response = requests.post(f"{BASE_URL}/api/v2/banking/transfer", json=negative_amount)
    print(f"Status: {response.status_code}")
    print(f"Expected: 422 (Validation error) - Amount must be > 0")
    print(f"Response: {response.text}")
    
    # Test 4: Missing amount
    print("\n❌ Test 4: Missing Amount...")
    missing_amount = {
        "to_account": "1234567890",
        "description": "Test transfer"
    }
    
    response = requests.post(f"{BASE_URL}/api/v2/banking/transfer", json=missing_amount)
    print(f"Status: {response.status_code}")
    print(f"Expected: 422 (Validation error) - Amount is required")
    print(f"Response: {response.text}")
    
    # Test 5: Wrong data type for amount
    print("\n❌ Test 5: Wrong Data Type for Amount...")
    wrong_type = {
        "amount": "1000",  # String instead of float
        "to_account": "1234567890",
        "description": "Test transfer"
    }
    
    response = requests.post(f"{BASE_URL}/api/v2/banking/transfer", json=wrong_type)
    print(f"Status: {response.status_code}")
    print(f"Expected: 422 (Validation error) - Amount must be float")
    print(f"Response: {response.text}")
    
    print("\n" + "=" * 60)
    print("✅ Transfer Validation Test Complete!")

def show_transfer_limits_info():
    """Show information about the transfer limits"""
    
    print("\n📋 TRANSFER LIMITS INFORMATION")
    print("=" * 60)
    print("🕐 Time Window: 1 hour (rolling)")
    print()
    print("📊 Frequency Limits:")
    print("- Maximum transfers per hour: 5")
    print("- Exceeding this triggers auto-logout")
    print("- Reason: transfer_frequency_limit")
    print()
    print("💰 Amount Limits:")
    print("- Maximum cumulative amount per hour: ₹50,000")
    print("- Exceeding this triggers auto-logout")
    print("- Reason: transfer_cumulative_limit")
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
    print()
    print("📝 API Response Format:")
    print("When limits are exceeded:")
    print(json.dumps({
        "status": "security_violation",
        "message": "Transfer frequency limit exceeded. You have made 5 transfers in the last hour (limit: 5)",
        "action": "user_logged_out",
        "reason": "transfer_frequency_limit",
        "details": "Excessive transfer activity detected: 5 transfers in 1 hour"
    }, indent=2))

def test_limits_api():
    """Test the limits API endpoint"""
    
    print("\n📊 Testing Limits API...")
    print("=" * 60)
    
    # Test without authentication
    response = requests.get(f"{BASE_URL}/api/v2/banking/limits")
    print(f"Status: {response.status_code}")
    print(f"Expected: 403 (Not authenticated)")
    print(f"Response: {response.text}")
    
    print("\n✅ Limits API Test Complete!")

if __name__ == "__main__":
    test_transfer_validation()
    test_limits_api()
    show_transfer_limits_info() 