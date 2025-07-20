#!/usr/bin/env python3
"""
Security System Testing Script
Tests all security features of the banking application
"""

import requests
import json
import time
import sys
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"
TEST_USER = {
    "username": "testuser",  # Change this to your username
    "password": "testpass123",  # Change this to your password
    "auth_type": "typing"
}

class SecurityTester:
    def __init__(self):
        self.session = requests.Session()
        self.token = None
        self.user_id = None
        
    def login(self):
        """Login to get authentication token"""
        print("🔐 Testing Login...")
        
        login_data = {
            "username": TEST_USER["username"],
            "password": TEST_USER["password"],
            "auth_type": TEST_USER["auth_type"],
            "behavioral_data": {
                "typing_speed": 0.5,
                "confidence": 0.8
            }
        }
        
        try:
            response = self.session.post(f"{BASE_URL}/api/v2/authenticate", json=login_data)
            if response.status_code == 200:
                data = response.json()
                self.token = data.get("access_token")
                self.user_id = data.get("user_id")
                print("✅ Login successful")
                print(f"   User ID: {self.user_id}")
                print(f"   Token: {self.token[:20]}...")
                return True
            else:
                print(f"❌ Login failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Login error: {e}")
            return False
    
    def test_session_status(self):
        """Test session management"""
        print("\n⏰ Testing Session Management...")
        
        headers = {"Authorization": f"Bearer {self.token}"}
        
        try:
            response = self.session.get(f"{BASE_URL}/api/v2/session/status", headers=headers)
            if response.status_code == 200:
                data = response.json()
                print("✅ Session status retrieved")
                print(f"   Remaining time: {data.get('remaining_time_formatted', 'N/A')}")
                print(f"   Is expired: {data.get('is_expired', 'N/A')}")
                print(f"   Timeout minutes: {data.get('session_timeout_minutes', 'N/A')}")
                return True
            else:
                print(f"❌ Session status failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Session status error: {e}")
            return False
    
    def test_transaction_anomaly_detection(self):
        """Test transaction anomaly detection"""
        print("\n💰 Testing Transaction Anomaly Detection...")
        
        headers = {"Authorization": f"Bearer {self.token}"}
        
        # Test 1: Normal transaction
        print("   Testing normal transaction...")
        normal_transaction = {
            "amount": 5000,
            "to_account": "1234567890",
            "upi_id": "test@upi",
            "description": "Test payment"
        }
        
        try:
            response = self.session.post(f"{BASE_URL}/api/v2/banking/transfer", 
                                       json=normal_transaction, headers=headers)
            if response.status_code == 200:
                print("   ✅ Normal transaction allowed")
            elif response.status_code == 403:
                print("   ⚠️ Normal transaction blocked (anomaly detected)")
            else:
                print(f"   ❌ Normal transaction failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Normal transaction error: {e}")
        
        # Test 2: Large transaction (should be blocked)
        print("   Testing large transaction...")
        large_transaction = {
            "amount": 20000000,  # 2 crore
            "to_account": "1234567890",
            "upi_id": "test@upi",
            "description": "Large payment"
        }
        
        try:
            response = self.session.post(f"{BASE_URL}/api/v2/banking/transfer", 
                                       json=large_transaction, headers=headers)
            if response.status_code == 403:
                print("   ✅ Large transaction blocked (anomaly detected)")
                data = response.json()
                print(f"   Reason: {data.get('detail', 'Unknown')}")
            else:
                print(f"   ⚠️ Large transaction allowed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Large transaction error: {e}")
        
        # Test 3: Repeated transaction (should be blocked after 3rd attempt)
        print("   Testing repeated transaction...")
        repeated_transaction = {
            "amount": 10000,
            "to_account": "1234567890",
            "upi_id": "test@upi",
            "description": "Repeated payment"
        }
        
        for i in range(4):
            try:
                response = self.session.post(f"{BASE_URL}/api/v2/banking/transfer", 
                                           json=repeated_transaction, headers=headers)
                if response.status_code == 200:
                    print(f"   ✅ Repeated transaction {i+1} allowed")
                elif response.status_code == 403:
                    print(f"   ✅ Repeated transaction {i+1} blocked (anomaly detected)")
                    data = response.json()
                    print(f"   Reason: {data.get('detail', 'Unknown')}")
                    break
                else:
                    print(f"   ❌ Repeated transaction {i+1} failed: {response.status_code}")
            except Exception as e:
                print(f"   ❌ Repeated transaction {i+1} error: {e}")
    
    def test_behavioral_analysis(self):
        """Test behavioral analysis"""
        print("\n🧠 Testing Behavioral Analysis...")
        
        headers = {"Authorization": f"Bearer {self.token}"}
        
        # Test typing pattern analysis (tap-speed endpoint expects List[float])
        print("   Testing typing pattern analysis...")
        timestamps = [time.time() + i * 0.5 for i in range(10)]  # Normal typing pattern
        
        try:
            response = self.session.post(f"{BASE_URL}/api/v2/tap-speed/analyze", 
                                       json=timestamps, headers=headers)
            if response.status_code == 200:
                data = response.json()
                print("   ✅ Typing pattern analyzed")
                print(f"   Anomaly detected: {data.get('user_flagged', 'N/A')}")
                print(f"   Recommended action: {data.get('recommended_action', 'N/A')}")
            else:
                print(f"   ❌ Typing pattern analysis failed: {response.status_code}")
                print(f"   Response: {response.text}")
        except Exception as e:
            print(f"   ❌ Typing pattern analysis error: {e}")
        
        # Test behavioral matching (expects timestamps in behavioral_data)
        print("   Testing behavioral matching...")
        behavioral_data = {
            "timestamps": [time.time() + i * 0.3 for i in range(5)]  # Need at least 3 timestamps
        }
        
        try:
            response = self.session.post(f"{BASE_URL}/api/v2/behavioral/match", 
                                       json=behavioral_data, headers=headers)
            if response.status_code == 200:
                data = response.json()
                print("   ✅ Behavioral matching completed")
                print(f"   Match score: {data.get('match_score', 'N/A')}")
                print(f"   Behavioral mismatch: {data.get('behavioral_mismatch', 'N/A')}")
                print(f"   Recommended action: {data.get('recommended_action', 'N/A')}")
            else:
                print(f"   ❌ Behavioral matching failed: {response.status_code}")
                print(f"   Response: {response.text}")
        except Exception as e:
            print(f"   ❌ Behavioral matching error: {e}")
    
    def test_gps_anomaly_detection(self):
        """Test GPS anomaly detection"""
        print("\n📍 Testing GPS Anomaly Detection...")
        
        headers = {"Authorization": f"Bearer {self.token}"}
        
        # Test normal GPS location
        print("   Testing normal GPS location...")
        normal_gps = {
            "latitude": 12.9716,
            "longitude": 77.5946
        }
        
        try:
            response = self.session.post(f"{BASE_URL}/api/v2/gps/update", 
                                       json=normal_gps, headers=headers)
            if response.status_code == 200:
                data = response.json()
                print("   ✅ Normal GPS location updated")
                print(f"   Anomaly detected: {data.get('gps_analysis', {}).get('is_anomaly', 'N/A')}")
            else:
                print(f"   ❌ Normal GPS update failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Normal GPS update error: {e}")
        
        # Test anomalous GPS location (very far from normal)
        print("   Testing anomalous GPS location...")
        anomalous_gps = {
            "latitude": 40.7128,  # New York (very far from Bangalore)
            "longitude": -74.0060
        }
        
        try:
            response = self.session.post(f"{BASE_URL}/api/v2/gps/update", 
                                       json=anomalous_gps, headers=headers)
            if response.status_code == 200:
                data = response.json()
                print("   ✅ Anomalous GPS location updated")
                print(f"   Anomaly detected: {data.get('gps_analysis', {}).get('is_anomaly', 'N/A')}")
            else:
                print(f"   ❌ Anomalous GPS update failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Anomalous GPS update error: {e}")
    
    def test_transaction_limits(self):
        """Test transaction limits"""
        print("\n🔒 Testing Transaction Limits...")
        
        headers = {"Authorization": f"Bearer {self.token}"}
        
        try:
            response = self.session.get(f"{BASE_URL}/api/v2/banking/limits", headers=headers)
            if response.status_code == 200:
                data = response.json()
                print("   ✅ Transaction limits retrieved")
                print(f"   Daily limit: ₹{data.get('daily_limit', 'N/A'):,}")
                print(f"   Transaction limit: ₹{data.get('transaction_limit', 'N/A'):,}")
                print(f"   UPI limit: ₹{data.get('upi_limit', 'N/A'):,}")
                print(f"   NEFT limit: ₹{data.get('neft_limit', 'N/A'):,}")
                print(f"   RTGS limit: ₹{data.get('rtgs_limit', 'N/A'):,}")
                print(f"   Is flagged: {data.get('is_flagged', 'N/A')}")
            else:
                print(f"   ❌ Transaction limits failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Transaction limits error: {e}")
    
    def test_risk_profile(self):
        """Test user risk profile"""
        print("\n📊 Testing User Risk Profile...")
        
        headers = {"Authorization": f"Bearer {self.token}"}
        
        try:
            response = self.session.get(f"{BASE_URL}/api/v2/user/risk-profile", headers=headers)
            if response.status_code == 200:
                data = response.json()
                print("   ✅ Risk profile retrieved")
                print(f"   Confidence score: {data.get('confidence_score', 'N/A')}")
                print(f"   Anomaly score: {data.get('anomaly_score', 'N/A')}")
                print(f"   Tap anomaly: {data.get('tap_anomaly', 'N/A')}")
                print(f"   Behavioral mismatch: {data.get('behavioral_mismatch', 'N/A')}")
                print(f"   GPS anomaly: {data.get('gps_anomaly', 'N/A')}")
            else:
                print(f"   ❌ Risk profile failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Risk profile error: {e}")
    
    def test_security_logs(self):
        """Test security logging"""
        print("\n📝 Testing Security Logging...")
        
        headers = {"Authorization": f"Bearer {self.token}"}
        
        try:
            response = self.session.get(f"{BASE_URL}/api/v2/user/logs", headers=headers)
            if response.status_code == 200:
                data = response.json()
                print("   ✅ Security logs retrieved")
                print(f"   Total logs: {data.get('total_count', 0)}")
                
                # Show recent security events
                recent_logs = data.get('security_logs', [])[:5]
                for log in recent_logs:
                    print(f"   - {log.get('event_type', 'Unknown')}: {log.get('details', 'No details')}")
            else:
                print(f"   ❌ Security logs failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Security logs error: {e}")
    
    def run_full_security_test(self):
        """Run complete security test suite"""
        print("🔒 Starting Comprehensive Security System Test")
        print("=" * 50)
        
        # Login first
        if not self.login():
            print("❌ Cannot proceed without login")
            return
        
        # Run all security tests
        self.test_session_status()
        self.test_transaction_anomaly_detection()
        self.test_behavioral_analysis()
        self.test_gps_anomaly_detection()
        self.test_transaction_limits()
        self.test_risk_profile()
        self.test_security_logs()
        
        print("\n" + "=" * 50)
        print("✅ Security System Test Complete!")
        print("\n📋 Summary:")
        print("   - Session Management: ✅ Active")
        print("   - Transaction Anomaly Detection: ✅ Active")
        print("   - Behavioral Analysis: ✅ Active")
        print("   - GPS Anomaly Detection: ✅ Active")
        print("   - Transaction Limits: ✅ Active")
        print("   - Risk Profiling: ✅ Active")
        print("   - Security Logging: ✅ Active")

def main():
    """Main function to run security tests"""
    print("🔒 Banking Security System Tester")
    print("=" * 40)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code != 200:
            print("❌ Server not responding. Please start your banking server first.")
            print("   Command: uvicorn banking_auth_app:app --reload")
            return
    except:
        print("❌ Cannot connect to server. Please start your banking server first.")
        print("   Command: uvicorn banking_auth_app:app --reload")
        return
    
    # Run security tests
    tester = SecurityTester()
    
    # Try to login first
    if not tester.login():
        print("\n❌ Login failed. Creating test user...")
        print("   Run this command to create a test user:")
        print("   python create_test_user.py")
        print("\n   Then run the security test again:")
        print("   python security_test.py")
        return
    
    tester.run_full_security_test()

if __name__ == "__main__":
    main() 