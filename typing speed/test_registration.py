import requests
import json

def test_registration():
    url = "http://127.0.0.1:8000/api/v2/register"
    
    payload = {
        "name": "Test User 2",
        "username": "testuser456",
        "email": "test456@example.com",
        "password": "password123",
        "auth_type": "typing",
        "behavioral_data": {
            "timestamps": [0.0, 0.5, 1.0, 1.5, 2.0]
        },
        "baseline_behavior": {
            "timestamps": [0.0, 0.5, 1.0, 1.5, 2.0]
        }
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        try:
            result = response.json()
            print(f"Response JSON: {json.dumps(result, indent=2)}")
        except json.JSONDecodeError:
            print(f"Response Text: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_registration() 