from fastapi import FastAPI, Request, Depends, HTTPException
from datetime import datetime
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

app = FastAPI()

# In-memory navigation logs (replace with DB in production)
navigation_logs = []

# Load or train ML model
MODEL_PATH = "navigation_anomaly_model.joblib"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    # Dummy training data: [number_of_accesses, avg_time_between_accesses]
    X_train = np.array([
        [5, 60],   # 5 accesses, 60s avg between
        [10, 45],
        [7, 70],
        [8, 55]
    ])
    model = IsolationForest(contamination='auto', random_state=42)
    model.fit(X_train)
    joblib.dump(model, MODEL_PATH)

def get_current_user(request: Request):
    # In real app, extract user from session/token
    return request.headers.get("X-User-Id", "anonymous")

def extract_features(user_id):
    # Get last N logs for this user
    user_logs = [log for log in navigation_logs if log["user_id"] == user_id]
    if len(user_logs) < 2:
        return np.array([[1, 999]])  # Not enough data, treat as normal
    times = [datetime.fromisoformat(log["timestamp"]) for log in user_logs]
    times.sort()
    intervals = [(t2 - t1).total_seconds() for t1, t2 in zip(times, times[1:])]
    avg_interval = sum(intervals) / len(intervals) if intervals else 999
    return np.array([[len(user_logs), avg_interval]])

@app.middleware("http")
async def log_and_check_navigation(request: Request, call_next):
    user_id = get_current_user(request)
    path = request.url.path
    timestamp = datetime.now().isoformat()
    navigation_logs.append({
        "user_id": user_id,
        "path": path,
        "timestamp": timestamp
    })

    # Extract features and check for anomaly
    features = extract_features(user_id)
    prediction = model.predict(features)[0]
    if prediction == -1:
        # Anomaly detected!
        print(f"Anomaly detected for user {user_id} on path {path}")
        # Optionally, block or alert
        # return JSONResponse(status_code=403, content={"detail": "Suspicious activity detected"})
    response = await call_next(request)
    return response

@app.get("/account")
def view_account(user_id: str = Depends(get_current_user)):
    return {"message": f"Account details for {user_id}"}

@app.get("/transfer")
def transfer_money(user_id: str = Depends(get_current_user)):
    return {"message": f"Transfer page for {user_id}"}

@app.get("/logs")
def get_logs():
    return navigation_logs