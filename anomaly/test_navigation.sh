#!/bin/bash

BASE_URL="http://127.0.0.1:8000"
USER_ID="alice"

# Simulate normal navigation
echo "Simulating normal navigation..."
curl -s -H "X-User-Id: $USER_ID" $BASE_URL/account
sleep 2
curl -s -H "X-User-Id: $USER_ID" $BASE_URL/transfer
sleep 2
curl -s -H "X-User-Id: $USER_ID" $BASE_URL/account
sleep 2

# Simulate rapid/suspicious navigation
echo "\nSimulating rapid navigation (should trigger anomaly)..."
for i in {1..10}; do
  curl -s -H "X-User-Id: $USER_ID" $BASE_URL/account > /dev/null
done

# Simulate another user
echo "\nSimulating another user (bob)..."
curl -s -H "X-User-Id: bob" $BASE_URL/account

# Print navigation logs
echo "\nNavigation logs:"
curl -s $BASE_URL/logs | jq 