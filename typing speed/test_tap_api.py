import requests
import json

def test_tap_speed_analysis():
    url = "http://127.0.0.1:8000/api/v2/tap-speed/analyze"
    
    # Test data - human-like tap pattern
    timestamps = [0.0, 0.8, 1.6, 2.5, 3.3, 4.2, 5.1]
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer test_token"
    }
    
    try:
        response = requests.post(url, json=timestamps, headers=headers)
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
    test_tap_speed_analysis() 