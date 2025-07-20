import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_large_transaction():
    """Test if large transactions are properly blocked"""
    print("🔒 Testing Large Transaction Blocking")
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
        
        # Test different transaction amounts
        test_amounts = [
            (1000, "Small transaction (₹1,000)"),
            (5000, "Medium transaction (₹5,000)"),
            (10000, "Large transaction (₹10,000)"),
            (50000, "Very large transaction (₹50,000)"),
            (100000, "Huge transaction (₹1,00,000)"),
            (2000000, "Massive transaction (₹20,00,000) - This should be BLOCKED!")
        ]
        
        for amount, description in test_amounts:
            print(f"\n💰 Testing {description}...")
            
            transaction_data = {
                "amount": amount,
                "to_account": "1234567890",
                "description": f"Test transfer of ₹{amount:,}"
            }
            
            try:
                response = requests.post(
                    f"{BASE_URL}/api/v2/banking/transfer", 
                    json=transaction_data, 
                    headers=headers
                )
                
                if response.status_code == 200:
                    print(f"   ❌ TRANSACTION WENT THROUGH! This is a SECURITY FAILURE!")
                    print(f"   Amount: ₹{amount:,}")
                    print(f"   Response: {response.json()}")
                elif response.status_code == 403:
                    print(f"   ✅ Transaction BLOCKED (403) - Good!")
                    print(f"   Reason: {response.json().get('detail', 'Unknown')}")
                elif response.status_code == 400:
                    print(f"   ✅ Transaction BLOCKED (400) - Good!")
                    print(f"   Reason: {response.json().get('detail', 'Unknown')}")
                else:
                    print(f"   ⚠️ Unexpected response: {response.status_code}")
                    print(f"   Response: {response.text}")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        print("\n" + "=" * 50)
        print("🔒 Large Transaction Security Test Complete!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_large_transaction() 