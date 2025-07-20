import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_auto_logout():
    """Test if users are automatically logged out on security violations"""
    print("🔒 Testing Auto-Logout Security Feature")
    print("=" * 50)
    
    # Login first
    login_data = {
        "username": "testuser",
        "password": "testpass123",
        "auth_type": "typing",
        "behavioral_data": {
            "timestamps": [time.time() + i * 0.3 for i in range(5)]
        }
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/v2/authenticate", json=login_data)
        if response.status_code != 200:
            print("❌ Login failed")
            return
        
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        print("✅ Login successful")
        print(f"Token: {token[:20]}...")
        
        # Test 1: Large transaction that should trigger auto-logout
        print("\n💰 Test 1: Large Transaction Auto-Logout")
        print("-" * 40)
        
        transaction_data = {
            "amount": 2000000.0,  # 20 lakhs (must be float)
            "to_account": "1234567890",
            "description": "Test large transfer"
        }
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/v2/banking/transfer", 
                json=transaction_data, 
                headers=headers
            )
            
            print(f"Response Status: {response.status_code}")
            print(f"Response: {response.json()}")
            
            if response.status_code == 403:
                response_data = response.json()
                if response_data.get("action") == "user_logged_out":
                    print("✅ Auto-logout triggered successfully!")
                    print(f"Reason: {response_data.get('reason')}")
                    print(f"Message: {response_data.get('message')}")
                else:
                    print("❌ Auto-logout not triggered")
            else:
                print("❌ Unexpected response")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test 2: Verify user is actually logged out by trying another request
        print("\n🔍 Test 2: Verifying User is Logged Out")
        print("-" * 40)
        
        try:
            # Try to access a protected endpoint
            response = requests.get(f"{BASE_URL}/api/v2/user/risk-profile", headers=headers)
            
            if response.status_code == 401:
                print("✅ User successfully logged out - 401 Unauthorized")
            else:
                print(f"❌ User still has access - Status: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test 3: Try to login again after auto-logout
        print("\n🔐 Test 3: Re-login After Auto-Logout")
        print("-" * 40)
        
        try:
            response = requests.post(f"{BASE_URL}/api/v2/authenticate", json=login_data)
            if response.status_code == 200:
                new_token = response.json()["access_token"]
                print("✅ Re-login successful")
                print(f"New Token: {new_token[:20]}...")
                
                # Test if new session works
                new_headers = {"Authorization": f"Bearer {new_token}"}
                response = requests.get(f"{BASE_URL}/api/v2/user/risk-profile", headers=new_headers)
                if response.status_code == 200:
                    print("✅ New session working properly")
                else:
                    print("❌ New session not working")
            else:
                print("❌ Re-login failed")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "=" * 50)
        print("🔒 Auto-Logout Security Test Complete!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_auto_logout() 