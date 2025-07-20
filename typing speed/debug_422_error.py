#!/usr/bin/env python3
"""
Debug 422 Error
Tests the transfer endpoint to identify the validation error
"""

import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def debug_422_error():
    """Debug the 422 Unprocessable Entity error"""
    
    print("🔍 Debugging 422 Error...")
    print("=" * 50)
    
    # Test 1: Valid transfer data
    print("💰 Test 1: Valid Transfer Data...")
    valid_transfer_data = {
        "amount": 1000.0,  # Must be float and > 0
        "to_account": "1234567890",
        "description": "Test transfer"
    }
    
    print(f"Sending data: {json.dumps(valid_transfer_data, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/api/v2/banking/transfer", json=valid_transfer_data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    # Test 2: Invalid amount (0)
    print("\n💰 Test 2: Invalid Amount (0)...")
    invalid_amount_data = {
        "amount": 0,
        "to_account": "1234567890",
        "description": "Test transfer"
    }
    
    print(f"Sending data: {json.dumps(invalid_amount_data, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/api/v2/banking/transfer", json=invalid_amount_data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    # Test 3: Invalid amount (negative)
    print("\n💰 Test 3: Invalid Amount (negative)...")
    negative_amount_data = {
        "amount": -100,
        "to_account": "1234567890",
        "description": "Test transfer"
    }
    
    print(f"Sending data: {json.dumps(negative_amount_data, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/api/v2/banking/transfer", json=negative_amount_data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    # Test 4: Missing amount
    print("\n💰 Test 4: Missing Amount...")
    missing_amount_data = {
        "to_account": "1234567890",
        "description": "Test transfer"
    }
    
    print(f"Sending data: {json.dumps(missing_amount_data, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/api/v2/banking/transfer", json=missing_amount_data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    # Test 5: Wrong data type for amount
    print("\n💰 Test 5: Wrong Data Type for Amount...")
    wrong_type_data = {
        "amount": "1000",  # String instead of float
        "to_account": "1234567890",
        "description": "Test transfer"
    }
    
    print(f"Sending data: {json.dumps(wrong_type_data, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/api/v2/banking/transfer", json=wrong_type_data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    print("\n" + "=" * 50)
    print("🔍 422 Error Debug Complete!")

def show_expected_format():
    """Show the expected request format"""
    
    print("\n📋 EXPECTED REQUEST FORMAT")
    print("=" * 50)
    print("The TransactionRequest model expects:")
    print()
    print("Required Fields:")
    print("- amount: float (must be > 0)")
    print()
    print("Optional Fields:")
    print("- to_account: str (optional)")
    print("- upi_id: str (optional)")
    print("- description: str (optional)")
    print()
    print("Example Valid Request:")
    print(json.dumps({
        "amount": 1000.0,
        "to_account": "1234567890",
        "description": "Test transfer"
    }, indent=2))
    print()
    print("Common 422 Errors:")
    print("- amount is 0 or negative")
    print("- amount is missing")
    print("- amount is not a number")
    print("- amount is a string instead of float")

if __name__ == "__main__":
    debug_422_error()
    show_expected_format() 