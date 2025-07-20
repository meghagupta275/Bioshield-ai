#!/usr/bin/env python3
"""
Debug Registration 422 Error
Tests the registration endpoint to identify the validation error
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def debug_registration():
    """Debug the 422 Unprocessable Entity error in registration"""
    
    print("🔍 Debugging Registration 422 Error...")
    print("=" * 60)
    
    # Test 1: Valid registration data
    print("✅ Test 1: Valid Registration Data...")
    valid_registration = {
        "name": "Test User",
        "username": f"testuser{int(time.time())}",
        "email": f"testuser{int(time.time())}@example.com",
        "password": "testpass123",
        "auth_type": "tap",
        "behavioral_data": {
            "timestamps": [1000, 1250, 1520, 1780, 2050, 2320, 2580, 2850, 3120, 3380]
        },
        "baseline_behavior": {
            "typing_speed": 0.8,
            "tap_speed": 0.7,
            "swipe_pattern": [0.5, 0.6, 0.4, 0.7],
            "GPS": {"lat": 12.9716, "long": 77.5946},
            "nav_pattern": [0.6, 0.5, 0.7, 0.4]
        }
    }
    
    print(f"Sending data: {json.dumps(valid_registration, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/api/v2/register", json=valid_registration)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    # Test 2: Missing required fields
    print("\n❌ Test 2: Missing Required Fields...")
    
    missing_fields = [
        {"username": "testuser", "email": "test@example.com", "password": "testpass123", "auth_type": "tap", "behavioral_data": {}, "baseline_behavior": {}},
        {"name": "Test User", "email": "test@example.com", "password": "testpass123", "auth_type": "tap", "behavioral_data": {}, "baseline_behavior": {}},
        {"name": "Test User", "username": "testuser", "password": "testpass123", "auth_type": "tap", "behavioral_data": {}, "baseline_behavior": {}},
        {"name": "Test User", "username": "testuser", "email": "test@example.com", "auth_type": "tap", "behavioral_data": {}, "baseline_behavior": {}},
        {"name": "Test User", "username": "testuser", "email": "test@example.com", "password": "testpass123", "behavioral_data": {}, "baseline_behavior": {}},
        {"name": "Test User", "username": "testuser", "email": "test@example.com", "password": "testpass123", "auth_type": "tap", "baseline_behavior": {}},
        {"name": "Test User", "username": "testuser", "email": "test@example.com", "password": "testpass123", "auth_type": "tap", "behavioral_data": {}}
    ]
    
    field_names = ["name", "username", "email", "password", "auth_type", "behavioral_data", "baseline_behavior"]
    
    for i, data in enumerate(missing_fields):
        missing_field = field_names[i]
        print(f"\nMissing {missing_field}:")
        print(f"Data: {json.dumps(data, indent=2)}")
        
        response = requests.post(f"{BASE_URL}/api/v2/register", json=data)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
    
    # Test 3: Invalid field values
    print("\n❌ Test 3: Invalid Field Values...")
    
    # Short name
    short_name = {
        "name": "A",  # Too short (min_length=2)
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpass123",
        "auth_type": "tap",
        "behavioral_data": {},
        "baseline_behavior": {}
    }
    
    print("Short name (min_length=2):")
    response = requests.post(f"{BASE_URL}/api/v2/register", json=short_name)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    # Short username
    short_username = {
        "name": "Test User",
        "username": "ab",  # Too short (min_length=3)
        "email": "test@example.com",
        "password": "testpass123",
        "auth_type": "tap",
        "behavioral_data": {},
        "baseline_behavior": {}
    }
    
    print("\nShort username (min_length=3):")
    response = requests.post(f"{BASE_URL}/api/v2/register", json=short_username)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    # Short password
    short_password = {
        "name": "Test User",
        "username": "testuser",
        "email": "test@example.com",
        "password": "123",  # Too short (min_length=8)
        "auth_type": "tap",
        "behavioral_data": {},
        "baseline_behavior": {}
    }
    
    print("\nShort password (min_length=8):")
    response = requests.post(f"{BASE_URL}/api/v2/register", json=short_password)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    # Invalid email
    invalid_email = {
        "name": "Test User",
        "username": "testuser",
        "email": "invalid-email",  # Invalid email format
        "password": "testpass123",
        "auth_type": "tap",
        "behavioral_data": {},
        "baseline_behavior": {}
    }
    
    print("\nInvalid email format:")
    response = requests.post(f"{BASE_URL}/api/v2/register", json=invalid_email)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")

def show_registration_requirements():
    """Show the registration requirements"""
    
    print("\n📋 REGISTRATION REQUIREMENTS")
    print("=" * 60)
    print("Required Fields:")
    print("- name: str (min_length=2, max_length=50)")
    print("- username: str (min_length=3, max_length=50)")
    print("- email: str (valid email format)")
    print("- password: str (min_length=8)")
    print("- auth_type: str")
    print("- behavioral_data: Dict")
    print("- baseline_behavior: Dict")
    print()
    print("Example Valid Registration:")
    print(json.dumps({
        "name": "Test User",
        "username": "testuser123",
        "email": "test@example.com",
        "password": "testpass123",
        "auth_type": "tap",
        "behavioral_data": {
            "timestamps": [1000, 1250, 1520, 1780, 2050, 2320, 2580, 2850, 3120, 3380]
        },
        "baseline_behavior": {
            "typing_speed": 0.8,
            "tap_speed": 0.7,
            "swipe_pattern": [0.5, 0.6, 0.4, 0.7],
            "GPS": {"lat": 12.9716, "long": 77.5946},
            "nav_pattern": [0.6, 0.5, 0.7, 0.4]
        }
    }, indent=2))
    print()
    print("Common 422 Errors:")
    print("- Missing required fields")
    print("- Field values too short/long")
    print("- Invalid email format")
    print("- Invalid data types")

if __name__ == "__main__":
    debug_registration()
    show_registration_requirements() 