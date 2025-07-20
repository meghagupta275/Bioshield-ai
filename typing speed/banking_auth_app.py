import os
import secrets
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any

from fastapi import FastAPI, Request, Form, HTTPException, Depends, status
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, validator
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import bcrypt
import jwt
import redis
import numpy as np
import json
from cryptography.fernet import Fernet
import time
import joblib
import sys
sys.path.append('../anoamly')
from transaction_anomaly import fetch_user_transactions, is_transaction_anomalous, add_transaction, init_db
import geocoder
from location_history import create_location_table, add_user_location, fetch_user_locations

# --- Config ---
# DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/banking_auth")
DATABASE_URL = "sqlite:///./banking_auth.db"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_MINUTES = 10  # 10-minute session timeout

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Database ---
Base = declarative_base()
# engine = create_engine(DATABASE_URL)
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# --- Redis ---
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# --- Models ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False)  # New field
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    auth_type = Column(String(20), nullable=False)
    behavioral_profile = Column(Text, nullable=False)
    baseline_behavior = Column(Text, nullable=True)  # Encrypted JSON
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    failed_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime, nullable=True)
    is_admin = Column(Boolean, default=False)  # New field

class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    transaction_id = Column(String(50), unique=True, nullable=False)
    transaction_type = Column(String(20), nullable=False)
    amount = Column(Float, nullable=False)
    confidence_score = Column(Float, nullable=False)
    is_flagged = Column(Boolean, default=False)
    status = Column(String(20), default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)

class SecurityLog(Base):
    __tablename__ = "security_logs"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=True)
    event_type = Column(String(50), nullable=False)
    ip_address = Column(String(45), nullable=False)
    user_agent = Column(String(500), nullable=True)
    details = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

class BehaviorLog(Base):
    __tablename__ = "behavior_logs"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    typing_speed = Column(Float)
    nav_pattern = Column(Text)  # JSON-encoded list
    swipe_pattern = Column(Text)  # JSON-encoded list
    gps = Column(Text)  # JSON-encoded dict
    tap_speed = Column(Float)
    anomaly_score = Column(Float)

Base.metadata.create_all(bind=engine)

# --- Pydantic Models ---
class AuthType(str):
    TYPING = "typing"
    TAP = "tap"
    NAVIGATION = "navigation"

class UserRegistration(BaseModel):
    name: str = Field(..., min_length=2, max_length=50)
    username: str = Field(..., min_length=3, max_length=50)
    email: str
    password: str = Field(..., min_length=8)
    auth_type: str
    behavioral_data: Dict
    baseline_behavior: Dict  # New field for baseline

class UserAuthentication(BaseModel):
    username: str
    password: str
    auth_type: str
    behavioral_data: Dict

class TransactionRequest(BaseModel):
    amount: float = Field(..., gt=0)
    to_account: Optional[str] = None
    upi_id: Optional[str] = None
    description: Optional[str] = None

class FDRequest(BaseModel):
    amount: float
    tenure_months: int
    interest_rate: float

class FixedDeposit(Base):
    __tablename__ = "fixed_deposits"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    amount = Column(Float, nullable=False)
    tenure_months = Column(Integer, nullable=False)
    interest_rate = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# --- FastAPI App ---
app = FastAPI(
    title="Banking Behavioral Authentication System",
    description="Multi-factor behavioral biometric authentication for banking security",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
security = HTTPBearer()

# --- Utility Functions ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRATION_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=JWT_ALGORITHM)

def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def log_security_event(event_type: str, user_id: int = None, details: str = None, ip_address: str = "unknown", user_agent: str = None, db: Session = None):
    log = SecurityLog(
        user_id=user_id,
        event_type=event_type,
        ip_address=ip_address,
        user_agent=user_agent,
        details=details
    )
    db.add(log)
    db.commit()

# Load anomaly model (global)
anomaly_model = None
ANOMALY_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../anoamly/anomaly_model.joblib'))
if os.path.exists(ANOMALY_MODEL_PATH):
    anomaly_model = joblib.load(ANOMALY_MODEL_PATH)

# GPS Anomaly Detection (from anoamly/gps.py)
gps_anomaly_model = None
GPS_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../anoamly/gps_anomaly_model.joblib'))
if os.path.exists(GPS_MODEL_PATH):
    gps_anomaly_model = joblib.load(GPS_MODEL_PATH)

# Tap Speed Anomaly Detection (from anoamly/tap.py)
tap_anomaly_model = None
TAP_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'tap_anomaly_model.joblib'))
if os.path.exists(TAP_MODEL_PATH):
    try:
        tap_anomaly_model = joblib.load(TAP_MODEL_PATH)
        logger.info("Tap anomaly model loaded successfully")
        print(f"✅ Tap anomaly model loaded from: {TAP_MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load tap anomaly model: {e}")
        print(f"❌ Failed to load tap anomaly model: {e}")
else:
    logger.warning(f"Tap anomaly model file not found: {TAP_MODEL_PATH}")
    print(f"⚠️ Tap anomaly model file not found: {TAP_MODEL_PATH}")

# In-memory GPS location history (in production, use database)
user_gps_history = {}

# Navigation anomaly detection (from anoamly/navigation.py)
navigation_anomaly_model = None
NAVIGATION_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../anoamly/navigation_anomaly_model.joblib'))
if os.path.exists(NAVIGATION_MODEL_PATH):
    navigation_anomaly_model = joblib.load(NAVIGATION_MODEL_PATH)

# In-memory navigation logs (in production, use database)
user_navigation_logs = {}

# Feature extraction for anomaly model (similar to anoamly/tap.py)
def extract_tap_features(timestamps):
    """Extract features from tap timestamps for ML model (from anoamly/tap.py)."""
    if len(timestamps) < 3:
        return None
    intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
    avg_interval = sum(intervals) / len(intervals)
    min_interval = min(intervals)
    max_interval = max(intervals)
    variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
    tap_speed = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0]) if timestamps[-1] > timestamps[0] else 0
    return [avg_interval, min_interval, max_interval, variance, tap_speed]

# --- Enhanced Anomaly Detection Functions ---
def analyze_tap_speed_anomaly(timestamps: List[float]) -> dict:
    """Analyze tap speed for anomalies and flag suspicious patterns."""
    if len(timestamps) < 3:
        return {"is_anomaly": True, "confidence": 0.0, "flags": ["Insufficient tap data"], "tap_speed": 0.0}
    
    # Calculate intervals between taps
    intervals = []
    for i in range(1, len(timestamps)):
        interval = timestamps[i] - timestamps[i-1]
        intervals.append(interval)
    
    # Calculate metrics
    avg_interval = sum(intervals) / len(intervals)
    min_interval = min(intervals)
    max_interval = max(intervals)
    variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
    tap_speed = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0]) if timestamps[-1] > timestamps[0] else 0
    
    # Anomaly detection flags
    flags = []
    anomaly_score = 0.0
    
    # 1. Too Perfect Timing (BOT)
    if variance < 0.01:
        flags.append("Too perfect timing (no natural variation)")
        anomaly_score += 0.3
    
    # 2. Too Fast Taps (BOT)
    if min_interval < 0.05:
        flags.append("Unnaturally fast taps (< 50ms)")
        anomaly_score += 0.4
    
    # 3. Too Consistent (BOT)
    if max_interval - min_interval < 0.1:
        flags.append("Too consistent rhythm (suspicious)")
        anomaly_score += 0.2
    
    # 4. Unrealistic Speed (BOT)
    if tap_speed > 8.0:
        flags.append("Unrealistic tap speed (> 8 taps/sec)")
        anomaly_score += 0.3
    
    # 5. Perfect Intervals (BOT)
    if all(abs(interval - avg_interval) < 0.01 for interval in intervals):
        flags.append("Perfect intervals (machine-like)")
        anomaly_score += 0.4
    
    # 6. Too Slow (Suspicious)
    if tap_speed < 0.5:
        flags.append("Unusually slow tapping (< 0.5 taps/sec)")
        anomaly_score += 0.2
    
    # 7. Irregular Pattern (Suspicious)
    if variance > 2.0:
        flags.append("Highly irregular pattern (possible automation)")
        anomaly_score += 0.2
    
    # ML-based detection (from anomaly folder)
    ml_anomaly = False
    ml_confidence = 1.0
    if tap_anomaly_model is not None and len(timestamps) >= 3:
        features = extract_tap_features(timestamps)
        if features:
            try:
                X = np.array(features).reshape(1, -1)
                prediction = tap_anomaly_model.predict(X)
                ml_anomaly = prediction[0] == -1  # -1 for anomaly, 1 for normal
                if ml_anomaly:
                    flags.append("ML model: Tap pattern anomaly detected")
                    anomaly_score += 0.5
                    ml_confidence = 0.3
                else:
                    ml_confidence = 0.9
                logger.info(f"ML prediction successful: {prediction[0]}, features: {features}")
            except Exception as e:
                logger.error(f"ML tap analysis error: {e}")
                print(f"❌ ML tap analysis error: {e}")
                # Fallback to rule-based only if ML fails
                ml_confidence = 1.0
        else:
            logger.warning("Failed to extract features for ML model")
            print("⚠️ Failed to extract features for ML model")
    else:
        # No ML model available, use rule-based only
        if tap_anomaly_model is None:
            logger.info("Tap anomaly model not available, using rule-based detection only")
            print("ℹ️ Tap anomaly model not available, using rule-based detection only")
        else:
            logger.info("Insufficient timestamps for ML analysis")
            print("ℹ️ Insufficient timestamps for ML analysis")
    
    is_anomaly = anomaly_score > 0.3 or len(flags) > 0
    confidence = max(0.0, min(1.0, 1.0 - anomaly_score))
    
    # Combine rule-based and ML confidence
    combined_confidence = (confidence + ml_confidence) / 2
    
    return {
        "is_anomaly": bool(is_anomaly),
        "confidence": float(combined_confidence),
        "flags": list(flags),
        "tap_speed": float(tap_speed),
        "avg_interval": float(avg_interval),
        "variance": float(variance),
        "anomaly_score": float(anomaly_score),
        "total_taps": int(len(timestamps)),
        "ml_anomaly": bool(ml_anomaly),
        "ml_confidence": float(ml_confidence),
        "rule_confidence": float(confidence)
    }

def analyze_transaction_anomaly(transaction_data: dict, user_behavior: dict) -> dict:
    """Analyze transaction for anomalies based on amount, timing, and user behavior."""
    flags = []
    anomaly_score = 0.0
    
    amount = transaction_data.get("amount", 0)
    transaction_type = transaction_data.get("transaction_type", "transfer")
    user_confidence = user_behavior.get("confidence", 1.0)
    tap_anomaly = user_behavior.get("tap_anomaly", False)
    
    # 1. High Amount Flag
    if amount > 100000:  # ₹1 lakh
        flags.append("High transaction amount (> ₹1,00,000)")
        anomaly_score += 0.3
    
    # 2. Unusual Amount Pattern
    if amount % 1000 == 0 and amount > 10000:
        flags.append("Suspicious round amount pattern")
        anomaly_score += 0.2
    
    # 3. Low Behavioral Confidence
    if user_confidence < 0.5:
        flags.append("Low behavioral confidence score")
        anomaly_score += 0.4
    
    # 4. Tap Speed Anomaly
    if tap_anomaly:
        flags.append("Tap speed anomaly detected")
        anomaly_score += 0.5
    
    # 5. Multiple Rapid Transactions (would need transaction history)
    # This could be enhanced with actual transaction history
    
    # 6. Unusual Transaction Time
    current_hour = datetime.now().hour
    if current_hour < 6 or current_hour > 23:
        flags.append("Transaction during unusual hours")
        anomaly_score += 0.2
    
    # 7. Amount Threshold Based on User Profile
    if amount > 50000 and user_confidence < 0.7:
        flags.append("High amount with low confidence")
        anomaly_score += 0.3
    
    is_anomaly = anomaly_score > 0.3 or len(flags) > 0
    confidence = max(0.0, min(1.0, 1.0 - anomaly_score))
    
    return {
        "is_anomaly": is_anomaly,
        "confidence": confidence,
        "flags": flags,
        "anomaly_score": anomaly_score,
        "recommended_action": "block" if anomaly_score > 0.6 else "flag" if anomaly_score > 0.3 else "allow"
    }

def flag_suspicious_activity(user_id: int, activity_type: str, details: dict, db: Session):
    """Flag suspicious activity and log it for review."""
    log = SecurityLog(
        user_id=user_id,
        event_type=f"suspicious_{activity_type}",
        ip_address="unknown",
        details=json.dumps(details)
    )
    db.add(log)
    db.commit()
    
    # Update user's risk profile
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        current_profile = json.loads(user.behavioral_profile)
        current_profile["risk_score"] = current_profile.get("risk_score", 0) + 1
        current_profile["last_flagged"] = datetime.utcnow().isoformat()
        user.behavioral_profile = json.dumps(current_profile)
        db.commit()

# --- GPS Anomaly Detection Functions (from anoamly/gps.py) ---
def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two GPS coordinates in kilometers."""
    from geopy.distance import geodesic
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers

def analyze_gps_anomaly(user_id: int, current_lat: float, current_lon: float, db: Session) -> dict:
    """Analyze GPS location for anomalies using ML and distance-based rules."""
    if user_id not in user_gps_history:
        user_gps_history[user_id] = []
    
    current_location = (current_lat, current_lon)
    user_history = user_gps_history[user_id]
    
    # Add current location to history
    user_history.append(current_location)
    
    # Keep only last 10 locations
    if len(user_history) > 10:
        user_history = user_history[-10:]
        user_gps_history[user_id] = user_history
    
    flags = []
    anomaly_score = 0.0
    is_anomaly = False
    
    # Distance-based anomaly detection (50km rule)
    if len(user_history) >= 2:
        for old_location in user_history[:-1]:  # Check against all previous locations
            distance = calculate_distance(current_lat, current_lon, old_location[0], old_location[1])
            if distance > 50:  # 50km threshold
                flags.append(f"Location anomaly: {distance:.1f}km from previous location")
                anomaly_score += 0.4
                is_anomaly = True
    
    # ML-based anomaly detection (if model exists)
    if gps_anomaly_model is not None and len(user_history) >= 3:
        try:
            # Prepare features for ML model
            X = np.array(user_history)
            prediction = gps_anomaly_model.predict(X)
            if prediction[-1] == -1:  # Latest location is anomaly
                flags.append("ML model: GPS location anomaly detected")
                anomaly_score += 0.6
                is_anomaly = True
        except Exception as e:
            logger.error(f"GPS ML model error: {e}")
    
    # Rapid location changes (within short time)
    if len(user_history) >= 3:
        recent_locations = user_history[-3:]
        total_distance = 0
        for i in range(1, len(recent_locations)):
            dist = calculate_distance(
                recent_locations[i-1][0], recent_locations[i-1][1],
                recent_locations[i][0], recent_locations[i][1]
            )
            total_distance += dist
        
        if total_distance > 100:  # 100km in 3 locations
            flags.append(f"Rapid location changes: {total_distance:.1f}km in recent activity")
            anomaly_score += 0.3
            is_anomaly = True
    
    # Impossible travel speeds (would need timestamps for accurate calculation)
    # This could be enhanced with actual timestamps
    
    confidence = max(0.0, min(1.0, 1.0 - anomaly_score))
    
    # Log GPS anomaly if detected
    if is_anomaly:
        # Note: log_security_event is async, but this function is not
        # We'll handle logging in the calling async function
        pass
        
        # Update user's risk profile
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            current_profile = json.loads(user.behavioral_profile)
            current_profile["gps_anomaly"] = True
            current_profile["gps_anomaly_score"] = anomaly_score
            current_profile["gps_flags"] = flags
            current_profile["last_gps_check"] = datetime.utcnow().isoformat()
            user.behavioral_profile = json.dumps(current_profile)
            db.commit()
    
    return {
        "is_anomaly": is_anomaly,
        "anomaly_score": anomaly_score,
        "confidence": confidence,
        "flags": flags,
        "current_location": current_location,
        "distance_from_previous": calculate_distance(
            user_history[-2][0], user_history[-2][1], current_lat, current_lon
        ) if len(user_history) >= 2 else 0
    }

def update_user_gps_location(user_id: int, lat: float, lon: float, db: Session) -> dict:
    """Update user's GPS location and check for anomalies."""
    return analyze_gps_anomaly(user_id, lat, lon, db)

# --- Navigation Anomaly Detection Functions (from anoamly/navigation.py) ---
def extract_navigation_features(user_id: int) -> np.ndarray:
    """Extract navigation features for ML model (from anoamly/navigation.py)."""
    if user_id not in user_navigation_logs:
        return np.array([[1, 999]])  # Not enough data, treat as normal
    
    user_logs = user_navigation_logs[user_id]
    if len(user_logs) < 2:
        return np.array([[1, 999]])
    
    # Calculate time intervals between navigation events
    times = [datetime.fromisoformat(log["timestamp"]) for log in user_logs]
    times.sort()
    intervals = [(t2 - t1).total_seconds() for t1, t2 in zip(times, times[1:])]
    avg_interval = sum(intervals) / len(intervals) if intervals else 999
    
    return np.array([[len(user_logs), avg_interval]])

def analyze_navigation_anomaly(user_id: int, navigation_data: dict, db: Session) -> dict:
    """Analyze navigation patterns for anomalies using ML and pattern matching."""
    if user_id not in user_navigation_logs:
        user_navigation_logs[user_id] = []
    
    # Add navigation data to user's history
    user_navigation_logs[user_id].append(navigation_data)
    
    # Keep only last 20 navigation events
    if len(user_navigation_logs[user_id]) > 20:
        user_navigation_logs[user_id] = user_navigation_logs[user_id][-20:]
    
    flags = []
    anomaly_score = 0.0
    is_anomaly = False
    
    # ML-based anomaly detection (from anoamly/navigation.py)
    if navigation_anomaly_model is not None and len(user_navigation_logs[user_id]) >= 2:
        try:
            features = extract_navigation_features(user_id)
            prediction = navigation_anomaly_model.predict(features)[0]
            if prediction == -1:
                flags.append("ML model: Navigation pattern anomaly detected")
                anomaly_score += 0.6
                is_anomaly = True
        except Exception as e:
            logger.error(f"Navigation ML model error: {e}")
    
    # Pattern-based anomaly detection (from typing speed/navigation.py)
    user_logs = user_navigation_logs[user_id]
    
    # Check for rapid navigation (too many events in short time)
    if len(user_logs) >= 3:
        recent_logs = user_logs[-3:]
        total_time = sum(log.get("time_spent", 1) for log in recent_logs)
        if total_time < 5:  # Less than 5 seconds for 3 navigation events
            flags.append("Rapid navigation detected (suspicious automation)")
            anomaly_score += 0.4
            is_anomaly = True
    
    # Check for unusual navigation depth
    navigation_depth = navigation_data.get("navigation_depth", 0)
    if navigation_depth > 5:  # Very deep navigation
        flags.append("Unusually deep navigation path")
        anomaly_score += 0.3
        is_anomaly = True
    
    # Check for repetitive patterns (bot-like behavior)
    if len(user_logs) >= 5:
        recent_screens = [log.get("screen", "") for log in user_logs[-5:]]
        if len(set(recent_screens)) == 1:  # Same screen repeated
            flags.append("Repetitive navigation pattern detected")
            anomaly_score += 0.5
            is_anomaly = True
    
    # Check for impossible transitions
    transition_type = navigation_data.get("transition_type", "")
    if transition_type not in ["forward", "back", "home", "random"]:
        flags.append("Invalid navigation transition type")
        anomaly_score += 0.3
        is_anomaly = True
    
    confidence = max(0.0, min(1.0, 1.0 - anomaly_score))
    
    # Log navigation anomaly if detected
    if is_anomaly:
        # Note: log_security_event is async, but this function is not
        # We'll handle logging in the calling async function
        pass
        
        # Update user's risk profile
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            current_profile = json.loads(user.behavioral_profile)
            current_profile["navigation_anomaly"] = True
            current_profile["navigation_anomaly_score"] = anomaly_score
            current_profile["navigation_flags"] = flags
            current_profile["last_navigation_check"] = datetime.utcnow().isoformat()
            user.behavioral_profile = json.dumps(current_profile)
            db.commit()
    
    return {
        "is_anomaly": is_anomaly,
        "anomaly_score": anomaly_score,
        "confidence": confidence,
        "flags": flags,
        "navigation_count": len(user_navigation_logs[user_id]),
        "recent_screens": [log.get("screen", "") for log in user_logs[-5:]] if user_logs else []
    }

def update_user_navigation(user_id: int, navigation_data: dict, db: Session) -> dict:
    """Update user's navigation data and check for anomalies."""
    return analyze_navigation_anomaly(user_id, navigation_data, db)

# Fernet encryption utilities for baseline_behavior
FERNET_KEY = os.getenv("FERNET_KEY", Fernet.generate_key().decode())
fernet = Fernet(FERNET_KEY.encode())

def encrypt_baseline(data: dict) -> str:
    json_data = json.dumps(data).encode()
    return fernet.encrypt(json_data).decode()

def decrypt_baseline(token: str) -> dict:
    decrypted = fernet.decrypt(token.encode()).decode()
    return json.loads(decrypted)

# --- Rate Limiting (simple in-memory for demo) ---
RATE_LIMIT_SECONDS = 5
user_last_log_time = {}

# --- Session Management ---
def get_session_remaining_time(token: str) -> int:
    """Get remaining session time in seconds."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
        exp_timestamp = payload.get("exp")
        if exp_timestamp:
            current_time = datetime.utcnow().timestamp()
            remaining = int(exp_timestamp - current_time)
            return max(0, remaining)
    except:
        pass
    return 0

def is_session_expired(token: str) -> bool:
    """Check if session has expired."""
    return get_session_remaining_time(token) <= 0

def refresh_session_token(token: str) -> str:
    """Refresh session token with new expiration."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
        # Remove expiration from payload
        payload.pop("exp", None)
        # Create new token with fresh expiration
        return create_access_token(payload)
    except:
        return None

# --- Dependencies ---
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    token = credentials.credentials
    payload = decode_access_token(token)
    user = db.query(User).filter(User.id == payload["user_id"]).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# --- Admin Auth Dependency ---
def get_current_admin(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    token = credentials.credentials
    payload = decode_access_token(token)
    user = db.query(User).filter(User.id == payload["user_id"]).first()
    if not user or not user.is_admin:
        raise HTTPException(status_code=401, detail="Admin privileges required")
    return user

# --- Enhanced Behavioral Analysis with Matching ---
def analyze_typing_pattern(timestamps: List[float], user_id: int = None, db: Session = None) -> dict:
    """Enhanced behavioral analysis with tap speed anomaly detection and baseline matching."""
    if len(timestamps) < 2:
        return {"confidence": 0.0, "anomaly_score": None, "tap_anomaly": True, "behavioral_mismatch": True}
    
    # Basic typing pattern analysis
    intervals = np.diff(timestamps)
    avg_interval = float(np.mean(intervals))
    std_interval = float(np.std(intervals))
    confidence = max(0.0, min(1.0, 1.0 - std_interval))
    
    # Calculate typing speed
    typing_speed = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0]) if timestamps[-1] > timestamps[0] else 0
    
    # Tap speed anomaly detection
    tap_analysis = analyze_tap_speed_anomaly(timestamps)
    tap_anomaly = tap_analysis["is_anomaly"]
    
    # ML model prediction - Using tap anomaly model for typing patterns
    anomaly_score = None
    if tap_anomaly_model is not None and len(timestamps) >= 3:
        features = extract_tap_features(timestamps)
        if features:
            try:
                X = np.array(features).reshape(1, -1)
                prediction = tap_anomaly_model.predict(X)
                anomaly_score = float(prediction[0])  # -1 for anomaly, 1 for normal
            except Exception as e:
                logger.error(f"Tap anomaly model prediction error: {e}")
                anomaly_score = None
    else:
        # No ML model available, use rule-based only
        logger.info("Tap anomaly model not available for typing pattern analysis")
    
    # Prepare current behavior data
    current_behavior = {
        "timestamps": timestamps,
        "typing_speed": typing_speed,
        "confidence": confidence,
        "anomaly_score": anomaly_score,
        "tap_anomaly": tap_anomaly,
        "tap_analysis": tap_analysis,
        "auth_type": "typing"
    }
    
    # Behavioral matching if user_id and db are provided
    behavioral_mismatch = False
    match_score = 1.0
    mismatch_flags = []
    
    if user_id and db:
        try:
            mismatch_analysis = analyze_behavioral_mismatch(user_id, current_behavior, db)
            behavioral_mismatch = mismatch_analysis["is_mismatch"]
            match_score = mismatch_analysis["match_score"]
            mismatch_flags = mismatch_analysis["flags"]
            
            # Flag behavioral mismatch
            if behavioral_mismatch:
                flag_behavioral_mismatch(user_id, mismatch_analysis, db)
        except Exception as e:
            logger.error(f"Behavioral matching error: {e}")
    
    # GPS anomaly integration
    gps_anomaly = False
    gps_flags = []
    gps_anomaly_score = 0.0
    
    if user_id and db:
        # Check for GPS anomalies in user profile
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            profile = json.loads(user.behavioral_profile)
            gps_anomaly = profile.get("gps_anomaly", False)
            gps_flags = profile.get("gps_flags", [])
            gps_anomaly_score = profile.get("gps_anomaly_score", 0.0)
            
            # Add GPS anomaly to overall anomaly score
            if gps_anomaly and anomaly_score is not None:
                anomaly_score = max(-1, anomaly_score + gps_anomaly_score)
    
    return {
        "confidence": confidence,
        "anomaly_score": anomaly_score,
        "tap_anomaly": tap_anomaly,
        "tap_analysis": tap_analysis,
        "gps_anomaly": gps_anomaly,
        "gps_flags": gps_flags,
        "gps_anomaly_score": gps_anomaly_score,
        "behavioral_mismatch": behavioral_mismatch,
        "match_score": match_score,
        "mismatch_flags": mismatch_flags,
        "typing_speed": typing_speed
    }

# --- New API Endpoints for Enhanced Anomaly Detection ---
@app.post("/api/v2/tap-speed/analyze")
async def analyze_tap_speed(
    timestamps: List[float],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    request: Request = None
):
    """Analyze tap speed in real-time and flag anomalies."""
    try:
        if len(timestamps) < 3:
            raise HTTPException(status_code=400, detail="Need at least 3 tap timestamps")
        
        # Analyze tap speed
        tap_analysis = analyze_tap_speed_anomaly(timestamps)
        
        # Update user's behavioral profile with tap analysis
        profile = json.loads(current_user.behavioral_profile)
        profile["tap_anomaly"] = tap_analysis["is_anomaly"]
        profile["tap_analysis"] = tap_analysis
        profile["last_tap_check"] = datetime.utcnow().isoformat()
        
        current_user.behavioral_profile = json.dumps(profile)
        db.commit()
        
        # Log suspicious tap behavior
        if tap_analysis["is_anomaly"]:
            flag_suspicious_activity(
                current_user.id,
                "tap_speed",
                {
                    "flags": tap_analysis["flags"],
                    "tap_speed": tap_analysis["tap_speed"],
                    "anomaly_score": tap_analysis["anomaly_score"]
                },
                db
            )
            
            await log_security_event(
                "tap_speed_anomaly",
                user_id=current_user.id,
                details=f"Tap speed anomaly detected: {', '.join(tap_analysis['flags'])}",
                ip_address=request.client.host,
                db=db
            )
            
            # Auto-logout for severe tap anomalies
            if tap_analysis["anomaly_score"] > 0.7:
                await log_security_event(
                    "user_auto_logout_security",
                    user_id=current_user.id,
                    details=f"User automatically logged out due to severe tap anomaly: {', '.join(tap_analysis['flags'])}",
                    ip_address=request.client.host,
                    db=db
                )
                
                response_data = {
                    "status": "security_violation",
                    "message": f"Severe tap speed anomaly detected: {', '.join(tap_analysis['flags'])}",
                    "action": "user_logged_out",
                    "reason": "severe_tap_anomaly",
                    "details": f"Tap pattern indicates potential security threat"
                }
                
                response = JSONResponse(content=response_data, status_code=403)
                response.delete_cookie(key="access_token")
                
                return response
        
        return {
            "tap_analysis": tap_analysis,
            "user_flagged": tap_analysis["is_anomaly"],
            "recommended_action": "block" if tap_analysis["anomaly_score"] > 0.6 else "flag" if tap_analysis["anomaly_score"] > 0.3 else "allow"
        }
    except Exception as e:
        logger.error(f"Tap speed analysis error: {e}")
        print(f"❌ Tap speed analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Tap speed analysis failed: {str(e)}")

@app.post("/api/v2/transaction/analyze")
async def analyze_transaction_risk(
    transaction_data: dict,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    request: Request = None
):
    """Analyze transaction risk before processing."""
    profile = json.loads(current_user.behavioral_profile)
    
    transaction_analysis = analyze_transaction_anomaly(
        transaction_data,
        {
            "confidence": profile.get("confidence", 1.0),
            "tap_anomaly": profile.get("tap_anomaly", False)
        }
    )
    
    return {
        "transaction_analysis": transaction_analysis,
        "risk_level": "high" if transaction_analysis["anomaly_score"] > 0.6 else "medium" if transaction_analysis["anomaly_score"] > 0.3 else "low",
        "recommended_action": transaction_analysis["recommended_action"]
    }

@app.get("/api/v2/user/risk-profile")
async def get_user_risk_profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's current risk profile and anomaly status."""
    profile = json.loads(current_user.behavioral_profile)
    
    # Get recent security events
    recent_events = db.query(SecurityLog).filter(
        SecurityLog.user_id == current_user.id,
        SecurityLog.event_type.like("suspicious_%")
    ).order_by(SecurityLog.timestamp.desc()).limit(5).all()
    
    risk_score = profile.get("risk_score", 0)
    risk_level = "high" if risk_score > 5 else "medium" if risk_score > 2 else "low"
    
    return {
        "user_id": current_user.id,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "confidence_score": profile.get("confidence", 1.0),
        "tap_anomaly": profile.get("tap_anomaly", False),
        "last_flagged": profile.get("last_flagged"),
        "recent_suspicious_events": [
            {
                "event_type": event.event_type,
                "timestamp": event.timestamp.isoformat(),
                "details": event.details
            }
            for event in recent_events
        ]
    }

@app.post("/api/v2/behavioral/update")
async def update_behavioral_data(
    behavioral_data: dict,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    request: Request = None
):
    """Update user's behavioral data and check for anomalies."""
    timestamps = behavioral_data.get("timestamps", [])
    
    if not timestamps:
        raise HTTPException(status_code=400, detail="No behavioral data provided")
    
    # Analyze typing/tap patterns
    analysis = analyze_typing_pattern(timestamps)
    
    # Update user profile
    profile = json.loads(current_user.behavioral_profile)
    profile.update(analysis)
    profile["last_updated"] = datetime.utcnow().isoformat()
    
    current_user.behavioral_profile = json.dumps(profile)
    db.commit()
    
    # Log behavioral update
    await log_security_event(
        "behavioral_update",
        user_id=current_user.id,
        details=f"Behavioral data updated. Confidence: {analysis['confidence']}, Tap Anomaly: {analysis['tap_anomaly']}",
        ip_address=request.client.host,
        db=db
    )
    
    return {
        "status": "updated",
        "analysis": analysis,
        "user_flagged": analysis["tap_anomaly"] or analysis["confidence"] < 0.5
    }

# --- Enhanced Behavioral Matching and Flagging ---
def compare_behavioral_patterns(current_behavior: dict, baseline_behavior: dict) -> dict:
    """Compare current behavior against baseline and flag mismatches."""
    if not baseline_behavior:
        return {"match_score": 0.0, "is_mismatch": True, "flags": ["No baseline behavior available"]}
    
    flags = []
    mismatch_score = 0.0
    
    # 1. Typing Speed Comparison
    current_typing_speed = current_behavior.get("typing_speed", 0)
    baseline_typing_speed = baseline_behavior.get("typing_speed", 0)
    
    if baseline_typing_speed > 0:
        speed_diff = abs(current_typing_speed - baseline_typing_speed) / baseline_typing_speed
        if speed_diff > 0.3:  # 30% difference
            flags.append(f"Typing speed mismatch: {speed_diff:.1%} difference")
            mismatch_score += 0.3
    
    # 2. Tap Pattern Comparison
    current_tap_analysis = current_behavior.get("tap_analysis", {})
    baseline_tap_analysis = baseline_behavior.get("tap_analysis", {})
    
    if baseline_tap_analysis and current_tap_analysis:
        # Compare tap intervals
        current_avg_interval = current_tap_analysis.get("avg_interval", 0)
        baseline_avg_interval = baseline_tap_analysis.get("avg_interval", 0)
        
        if baseline_avg_interval > 0:
            interval_diff = abs(current_avg_interval - baseline_avg_interval) / baseline_avg_interval
            if interval_diff > 0.4:  # 40% difference
                flags.append(f"Tap interval mismatch: {interval_diff:.1%} difference")
                mismatch_score += 0.25
        
        # Compare tap variance
        current_variance = current_tap_analysis.get("variance", 0)
        baseline_variance = baseline_tap_analysis.get("variance", 0)
        
        if baseline_variance > 0:
            variance_diff = abs(current_variance - baseline_variance) / baseline_variance
            if variance_diff > 0.5:  # 50% difference
                flags.append(f"Tap pattern variance mismatch: {variance_diff:.1%} difference")
                mismatch_score += 0.2
    
    # 3. Typing Pattern Consistency
    current_timestamps = current_behavior.get("timestamps", [])
    baseline_timestamps = baseline_behavior.get("timestamps", [])
    
    if len(current_timestamps) >= 3 and len(baseline_timestamps) >= 3:
        # Compare typing rhythm
        current_intervals = np.diff(current_timestamps)
        baseline_intervals = np.diff(baseline_timestamps)
        
        if len(baseline_intervals) > 0:
            current_std = np.std(current_intervals)
            baseline_std = np.std(baseline_intervals)
            
            if baseline_std > 0:
                std_diff = abs(current_std - baseline_std) / baseline_std
                if std_diff > 0.6:  # 60% difference in rhythm
                    flags.append(f"Typing rhythm mismatch: {std_diff:.1%} difference")
                    mismatch_score += 0.25
    
    # 4. Confidence Score Drop
    current_confidence = current_behavior.get("confidence", 1.0)
    baseline_confidence = baseline_behavior.get("confidence", 1.0)
    
    if baseline_confidence > 0:
        confidence_drop = baseline_confidence - current_confidence
        if confidence_drop > 0.3:  # 30% drop in confidence
            flags.append(f"Confidence score dropped: {confidence_drop:.1%}")
            mismatch_score += 0.2
    
    # 5. Anomaly Score Comparison
    current_anomaly = current_behavior.get("anomaly_score", 1)
    baseline_anomaly = baseline_behavior.get("anomaly_score", 1)
    
    if current_anomaly == -1 and baseline_anomaly != -1:
        flags.append("Anomaly detected where none existed in baseline")
        mismatch_score += 0.4
    
    # 6. Behavioral Type Mismatch
    current_auth_type = current_behavior.get("auth_type", "typing")
    baseline_auth_type = baseline_behavior.get("auth_type", "typing")
    
    if current_auth_type != baseline_auth_type:
        flags.append(f"Authentication type changed: {baseline_auth_type} -> {current_auth_type}")
        mismatch_score += 0.3
    
    # Calculate match score
    match_score = max(0.0, 1.0 - mismatch_score)
    is_mismatch = mismatch_score > 0.3 or len(flags) > 0
    
    return {
        "match_score": match_score,
        "is_mismatch": is_mismatch,
        "mismatch_score": mismatch_score,
        "flags": flags,
        "confidence_drop": baseline_confidence - current_confidence if baseline_confidence > 0 else 0,
        "recommended_action": "block" if mismatch_score > 0.7 else "flag" if mismatch_score > 0.4 else "allow"
    }

def analyze_behavioral_mismatch(user_id: int, current_behavior: dict, db: Session) -> dict:
    """Analyze behavioral mismatch for a specific user."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return {"error": "User not found"}
    
    try:
        # Decrypt baseline behavior
        baseline_behavior = decrypt_baseline(user.baseline_behavior)
    except Exception:
        baseline_behavior = {}
    
    # Compare behaviors
    comparison = compare_behavioral_patterns(current_behavior, baseline_behavior)
    
    # Update user's risk profile
    profile = json.loads(user.behavioral_profile)
    profile["behavioral_mismatch"] = comparison["is_mismatch"]
    profile["match_score"] = comparison["match_score"]
    profile["last_behavioral_check"] = datetime.utcnow().isoformat()
    
    if comparison["is_mismatch"]:
        profile["mismatch_count"] = profile.get("mismatch_count", 0) + 1
        profile["last_mismatch"] = datetime.utcnow().isoformat()
        profile["mismatch_flags"] = comparison["flags"]
    
    user.behavioral_profile = json.dumps(profile)
    db.commit()
    
    return comparison

def flag_behavioral_mismatch(user_id: int, mismatch_details: dict, db: Session):
    """Flag behavioral mismatch and log for review."""
    log = SecurityLog(
        user_id=user_id,
        event_type="behavioral_mismatch",
        ip_address="unknown",
        details=json.dumps(mismatch_details)
    )
    db.add(log)
    db.commit()
    
    # Update user's risk score
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        profile = json.loads(user.behavioral_profile)
        profile["risk_score"] = profile.get("risk_score", 0) + 2  # Higher penalty for behavioral mismatch
        profile["last_flagged"] = datetime.utcnow().isoformat()
        user.behavioral_profile = json.dumps(profile)
        db.commit()

# --- New API Endpoints for Behavioral Matching ---
@app.post("/api/v2/behavioral/match")
async def check_behavioral_match(
    behavioral_data: dict,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    request: Request = None
):
    """Check if current behavior matches user's baseline."""
    timestamps = behavioral_data.get("timestamps", [])
    
    if len(timestamps) < 3:
        raise HTTPException(status_code=400, detail="Need at least 3 timestamps for behavioral matching")
    
    # Analyze behavior with matching
    analysis = analyze_typing_pattern(timestamps, current_user.id, db)
    
    # Log behavioral matching attempt
    await log_security_event(
        "behavioral_matching_check",
        user_id=current_user.id,
        details=f"Match score: {analysis['match_score']:.2f}, Mismatch: {analysis['behavioral_mismatch']}",
        ip_address=request.client.host,
        db=db
    )
    
    return {
        "match_score": analysis["match_score"],
        "behavioral_mismatch": analysis["behavioral_mismatch"],
        "mismatch_flags": analysis["mismatch_flags"],
        "confidence": analysis["confidence"],
        "tap_anomaly": analysis["tap_anomaly"],
        "recommended_action": "block" if analysis["match_score"] < 0.3 else "flag" if analysis["match_score"] < 0.6 else "allow"
    }

@app.post("/api/v2/behavioral/update-baseline")
async def update_baseline_behavior(
    behavioral_data: dict,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    request: Request = None
):
    """Update user's baseline behavior (for legitimate changes)."""
    timestamps = behavioral_data.get("timestamps", [])
    
    if len(timestamps) < 5:
        raise HTTPException(status_code=400, detail="Need at least 5 timestamps for baseline update")
    
    # Analyze new behavior
    analysis = analyze_typing_pattern(timestamps)
    
    # Create new baseline
    new_baseline = {
        "timestamps": timestamps,
        "typing_speed": analysis["typing_speed"],
        "confidence": analysis["confidence"],
        "anomaly_score": analysis["anomaly_score"],
        "tap_analysis": analysis["tap_analysis"],
        "auth_type": "typing",
        "updated_at": datetime.utcnow().isoformat()
    }
    
    # Encrypt and save new baseline
    encrypted_baseline = encrypt_baseline(new_baseline)
    current_user.baseline_behavior = encrypted_baseline
    
    # Reset mismatch counters
    profile = json.loads(current_user.behavioral_profile)
    profile["mismatch_count"] = 0
    profile["behavioral_mismatch"] = False
    profile["match_score"] = 1.0
    profile["baseline_updated"] = datetime.utcnow().isoformat()
    
    current_user.behavioral_profile = json.dumps(profile)
    db.commit()
    
    await log_security_event(
        "baseline_updated",
        user_id=current_user.id,
        details="User baseline behavior updated",
        ip_address=request.client.host,
        db=db
    )
    
    return {
        "status": "baseline_updated",
        "new_confidence": analysis["confidence"],
        "new_typing_speed": analysis["typing_speed"]
    }

@app.get("/api/v2/behavioral/baseline-info")
async def get_baseline_info(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get information about user's baseline behavior."""
    try:
        baseline = decrypt_baseline(current_user.baseline_behavior)
        profile = json.loads(current_user.behavioral_profile)
        
        return {
            "has_baseline": True,
            "baseline_typing_speed": baseline.get("typing_speed", 0),
            "baseline_confidence": baseline.get("confidence", 0),
            "baseline_updated": baseline.get("updated_at"),
            "current_match_score": profile.get("match_score", 1.0),
            "mismatch_count": profile.get("mismatch_count", 0),
            "last_mismatch": profile.get("last_mismatch"),
            "behavioral_mismatch": profile.get("behavioral_mismatch", False)
        }
    except Exception:
        return {
            "has_baseline": False,
            "message": "No baseline behavior available"
        }

@app.post("/api/v2/behavioral/force-match")
async def force_behavioral_match(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    request: Request = None
):
    """Force a behavioral match check with current session data."""
    # Get current session behavioral data
    profile = json.loads(current_user.behavioral_profile)
    
    # This would typically get current session timestamps
    # For demo, we'll use a placeholder
    current_timestamps = [time.time() - 10, time.time() - 8, time.time() - 5, time.time() - 2, time.time()]
    
    # Perform matching
    analysis = analyze_typing_pattern(current_timestamps, current_user.id, db)
    
    await log_security_event(
        "forced_behavioral_match",
        user_id=current_user.id,
        details=f"Match score: {analysis['match_score']:.2f}, Flags: {analysis['mismatch_flags']}",
        ip_address=request.client.host,
        db=db
    )
    
    return {
        "match_score": analysis["match_score"],
        "behavioral_mismatch": analysis["behavioral_mismatch"],
        "mismatch_flags": analysis["mismatch_flags"],
        "recommended_action": "block" if analysis["match_score"] < 0.3 else "flag" if analysis["match_score"] < 0.6 else "allow"
    }

# --- Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/api/v2/register")
async def register_user(user_data: UserRegistration, request: Request, db: Session = Depends(get_db)):
    try:
        # Check if user already exists
        if db.query(User).filter((User.username == user_data.username) | (User.email == user_data.email)).first():
            raise HTTPException(status_code=400, detail="Username or email already exists")
        
        # Analyze behavioral data
        analysis = analyze_typing_pattern(user_data.behavioral_data.get("timestamps", []))
        confidence = analysis["confidence"]
        anomaly_score = analysis["anomaly_score"]
        
        # Hash password and encrypt baseline
        password_hash = hash_password(user_data.password)
        encrypted_baseline = encrypt_baseline(user_data.baseline_behavior)
        
        # Create user
        user = User(
            name=user_data.name,
            username=user_data.username,
            email=user_data.email,
            password_hash=password_hash,
            auth_type=user_data.auth_type,
            behavioral_profile=json.dumps({
                "confidence": confidence, 
                "anomaly_score": anomaly_score,
                "risk_score": 0,
                "match_score": 1.0,
                "behavioral_mismatch": False,
                "mismatch_count": 0,
                "created_at": datetime.utcnow().isoformat()
            }),
            baseline_behavior=encrypted_baseline,
            created_at=datetime.utcnow()
        )
        
        db.add(user)
        db.commit()
        db.refresh(user)
        
        # Log registration event
        await log_security_event(
            "user_registered", 
            user_id=user.id, 
            details=f"User registered with confidence: {confidence:.2f}", 
            ip_address=request.client.host, 
            db=db
        )
        
        # Return JSON response for API calls
        return {
            "status": "success",
            "message": "User registered successfully",
            "user_id": user.id,
            "username": user.username,
            "confidence": confidence,
            "anomaly_score": anomaly_score
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the error and return a proper error response
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.get("/register_success", response_class=HTMLResponse)
async def register_success(request: Request):
    return templates.TemplateResponse("registration_success.html", {"request": request})

@app.post("/api/v2/authenticate")
async def authenticate_user(auth_data: UserAuthentication, request: Request, db: Session = Depends(get_db)):
    try:
        # Check user credentials
        user = db.query(User).filter(User.username == auth_data.username).first()
        if not user or not verify_password(auth_data.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Analyze behavioral data (typing/tap)
        analysis = analyze_typing_pattern(auth_data.behavioral_data.get("timestamps", []))
        confidence = analysis["confidence"]
        anomaly_score = analysis["anomaly_score"]
        tap_anomaly = analysis.get("tap_anomaly", False)
        tap_flags = analysis.get("tap_analysis", {}).get("flags", [])
        
        # --- Less Strict Tap Anomaly Check ---
        print(f"[DEBUG] Tap anomaly: {tap_anomaly}, anomaly_score: {anomaly_score}, flags: {tap_flags}")
        # Only block if anomaly_score is not None and > 0.5, or if there are actual flags
        if (anomaly_score is not None and anomaly_score > 0.5) or (tap_flags and len(tap_flags) > 0):
            await log_security_event(
                "tap_anomaly_login_blocked",
                user_id=user.id,
                details=f"Tap anomaly detected during login: {', '.join(tap_flags)}",
                ip_address=request.client.host,
                db=db
            )
            raise HTTPException(status_code=403, detail=f"Login blocked due to tap anomaly: {', '.join(tap_flags)}")
        
        # --- Less Strict Navigation Anomaly Check ---
        nav_data = auth_data.behavioral_data.get("navigation", {})
        nav_result = None
        if nav_data:
            nav_result = analyze_navigation_anomaly(user.id, nav_data, db)
            nav_flags = nav_result.get("flags", [])
            nav_score = nav_result.get("anomaly_score", None)
            print(f"[DEBUG] Navigation anomaly_score: {nav_score}, flags: {nav_flags}")
            if (nav_score is not None and nav_score > 0.5) or (nav_flags and len(nav_flags) > 0):
                await log_security_event(
                    "navigation_anomaly_login_blocked",
                    user_id=user.id,
                    details=f"Navigation anomaly detected during login: {', '.join(nav_flags)}",
                    ip_address=request.client.host,
                    db=db
                )
                raise HTTPException(status_code=403, detail=f"Login blocked due to navigation anomaly: {', '.join(nav_flags)}")
        
        # Check for behavioral anomalies
        if anomaly_score == -1:
            await log_security_event(
                "anomaly_detected_login",
                user_id=user.id,
                details="Anomaly detected during login attempt.",
                ip_address=request.client.host,
                db=db
            )
            raise HTTPException(status_code=403, detail="Login blocked due to behavioral anomaly.")
        
        # Create access token
        token = create_access_token({
            "user_id": user.id, 
            "username": user.username, 
            "confidence": confidence, 
            "anomaly_score": anomaly_score
        })
        
        # Log successful login
        await log_security_event(
            "login_successful", 
            user_id=user.id, 
            details=f"User logged in with confidence: {confidence:.2f}", 
            ip_address=request.client.host, 
            db=db
        )
        
        # After successful login, store user location
        g = geocoder.ip('me')
        print("[DEBUG] Geocoder result:", g.ok, g.latlng)
        if g.ok:
            lat, lon = g.latlng
            add_user_location(user.id, lat, lon)
            user_history = fetch_user_locations(user.id, limit=10)
            print("[DEBUG] User location history:", user_history)
            print("[DEBUG] Current location:", (lat, lon))
            def is_gps_anomaly(user_history, new_location, threshold_km=50):
                from geopy.distance import geodesic
                for old_location in user_history:
                    distance = geodesic(old_location, new_location).km
                    print(f"[DEBUG] Distance from {old_location} to {new_location}: {distance} km")
                    if distance <= threshold_km:
                        return False  # Not an anomaly
                return True  # Anomaly
            is_anomaly = is_gps_anomaly(user_history, (lat, lon))
            print(f"[DEBUG] [GPS] Anomaly detected on login: {is_anomaly}")
            if is_anomaly:
                await log_security_event(
                    "gps_anomaly_login_blocked",
                    user_id=user.id,
                    details=f"GPS anomaly detected during login. Location: ({lat}, {lon})",
                    ip_address=request.client.host,
                    db=db
                )
                raise HTTPException(status_code=403, detail="Login blocked due to GPS anomaly (suspicious location change detected).")
        
        # Return JSON response with token
        return {
            "status": "success",
            "message": "Authentication successful",
            "user_id": user.id,
            "username": user.username,
            "confidence": confidence,
            "anomaly_score": anomaly_score,
            "access_token": token,
            "redirect_url": "/dashboard"
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the error and return a proper error response
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(get_db)):
    # Extract user from cookie token
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse(url="/login")
    
    # Check if session has expired
    if is_session_expired(token):
        response = RedirectResponse(url="/login")
        response.delete_cookie(key="access_token")
        return response
    
    try:
        payload = decode_access_token(token)
        user = db.query(User).filter(User.id == payload["user_id"]).first()
        if not user:
            return RedirectResponse(url="/login")
    except Exception:
        return RedirectResponse(url="/login")
    
    return templates.TemplateResponse("banking_dashboard.html", {"request": request, "user": user})

@app.post("/api/v2/banking/transfer")
async def process_transfer(
    transaction_data: TransactionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    request: Request = None
):
    # --- Less Strict Tap Anomaly Check ---
    profile = json.loads(current_user.behavioral_profile)
    tap_timestamps = profile.get("tap_timestamps", [])
    tap_analysis = analyze_tap_speed_anomaly(tap_timestamps) if tap_timestamps else {"is_anomaly": False, "flags": []}
    if tap_analysis["is_anomaly"] and tap_analysis.get("anomaly_score", 0) > 0.5:
        await log_security_event(
            "tap_anomaly_transfer_blocked",
            user_id=current_user.id,
            details=f"Tap anomaly detected during transfer: {', '.join(tap_analysis['flags'])}",
            ip_address=request.client.host,
            db=db
        )
        response_data = {
            "status": "security_violation",
            "message": f"Transfer blocked due to tap anomaly: {', '.join(tap_analysis['flags'])}",
            "action": "user_logged_out",
            "reason": "tap_anomaly_detected",
            "details": f"Tap pattern indicates potential security threat"
        }
        response = JSONResponse(content=response_data, status_code=403)
        response.delete_cookie(key="access_token")
        return response
    # --- Stricter Navigation Anomaly Check ---
    nav_data = profile.get("navigation", {})
    nav_result = None
    if nav_data:
        nav_result = analyze_navigation_anomaly(current_user.id, nav_data, db)
        if nav_result["is_anomaly"] and nav_result["anomaly_score"] > 0.5:
            await log_security_event(
                "navigation_anomaly_transfer_blocked",
                user_id=current_user.id,
                details=f"Navigation anomaly detected during transfer: {', '.join(nav_result['flags'])}",
                ip_address=request.client.host,
                db=db
            )
            response_data = {
                "status": "security_violation",
                "message": f"Transfer blocked due to navigation anomaly: {', '.join(nav_result['flags'])}",
                "action": "user_logged_out",
                "reason": "navigation_anomaly_detected",
                "details": f"Navigation pattern indicates potential security threat"
            }
            response = JSONResponse(content=response_data, status_code=403)
            response.delete_cookie(key="access_token")
            return response
    
    # --- Transaction Anomaly Detection Integration ---
    init_db()  # Ensure DB is initialized (safe to call multiple times)
    user_id = str(current_user.id)
    new_transaction = {"amount": transaction_data.amount, "type": "transfer"}
    user_transactions = fetch_user_transactions(user_id)
    
    # Debug logging
    print(f"🔍 DEBUG: Processing transaction of ₹{transaction_data.amount:,}")
    print(f"🔍 DEBUG: User ID: {user_id}")
    print(f"🔍 DEBUG: Previous transactions: {len(user_transactions)}")
    
    is_anomaly, reason = is_transaction_anomalous(user_transactions, new_transaction)
    print(f"🔍 DEBUG: Anomaly detected: {is_anomaly}, Reason: {reason}")
    
    if is_anomaly:
        # Log the security event
        await log_security_event(
            "transaction_blocked_anomaly",
            user_id=current_user.id,
            details=f"Transaction blocked by rule-based anomaly detection. Amount: {transaction_data.amount}, Reason: {reason}",
            ip_address=request.client.host,
            db=db
        )
        
        # Log out the user for security
        await log_security_event(
            "user_auto_logout_security",
            user_id=current_user.id,
            details=f"User automatically logged out due to transaction anomaly: {reason}",
            ip_address=request.client.host,
            db=db
        )
        
        # Create response with logout message
        response_data = {
            "status": "security_violation",
            "message": f"Transaction blocked by anomaly detection: {reason}",
            "action": "user_logged_out",
            "reason": "security_anomaly_detected",
            "details": f"Large transaction amount ₹{transaction_data.amount:,} flagged as suspicious"
        }
        
        response = JSONResponse(content=response_data, status_code=403)
        response.delete_cookie(key="access_token")
        
        return response
    
    # Get current behavioral profile for additional security checks
    profile = json.loads(current_user.behavioral_profile)
    confidence_score = profile.get("confidence", 1.0)
    anomaly_score = profile.get("anomaly_score", 1.0)
    tap_anomaly = profile.get("tap_anomaly", False)
    
    # Enhanced transaction anomaly detection with behavioral matching
    transaction_analysis = analyze_transaction_anomaly(
        {
            "amount": transaction_data.amount,
            "transaction_type": "transfer"
        },
        {
            "confidence": confidence_score,
            "tap_anomaly": tap_anomaly
        }
    )
    
    # Check behavioral matching
    behavioral_mismatch = profile.get("behavioral_mismatch", False)
    match_score = profile.get("match_score", 1.0)
    mismatch_flags = profile.get("mismatch_flags", [])
    
    # Combine transaction and behavioral analysis
    combined_risk_score = transaction_analysis["anomaly_score"]
    if behavioral_mismatch:
        combined_risk_score += 0.4  # Add significant risk for behavioral mismatch
        transaction_analysis["flags"].extend(mismatch_flags)
    
    # Update recommended action based on combined risk
    if combined_risk_score > 0.7 or match_score < 0.3:
        transaction_analysis["recommended_action"] = "block"
    elif combined_risk_score > 0.4 or match_score < 0.6:
        transaction_analysis["recommended_action"] = "flag"
    
    # Determine if transaction should be flagged or blocked
    is_flagged = (
        confidence_score < 0.7 or 
        anomaly_score == -1 or 
        tap_anomaly or 
        transaction_analysis["is_anomaly"]
    )
    
    # Set transaction limits based on risk assessment
    if transaction_analysis["recommended_action"] == "block":
        # Log the security event
        await log_security_event(
            "transaction_blocked_anomaly",
            user_id=current_user.id,
            details=f"Transaction blocked due to anomaly. Amount: {transaction_data.amount}, Flags: {transaction_analysis['flags']}",
            ip_address=request.client.host,
            db=db
        )
        
        # Log out the user for security
        await log_security_event(
            "user_auto_logout_security",
            user_id=current_user.id,
            details=f"User automatically logged out due to behavioral anomaly: {', '.join(transaction_analysis['flags'])}",
            ip_address=request.client.host,
            db=db
        )
        
        # Create response with logout message
        response_data = {
            "status": "security_violation",
            "message": f"Transaction blocked due to suspicious activity: {', '.join(transaction_analysis['flags'])}",
            "action": "user_logged_out",
            "reason": "behavioral_anomaly_detected",
            "details": f"Behavioral analysis flagged transaction as suspicious"
        }
        
        response = JSONResponse(content=response_data, status_code=403)
        response.delete_cookie(key="access_token")
        
        return response
    
    # Flag suspicious activity for review
    if transaction_analysis["is_anomaly"]:
        flag_suspicious_activity(
            current_user.id, 
            "transaction", 
            {
                "amount": transaction_data.amount,
                "flags": transaction_analysis["flags"],
                "anomaly_score": transaction_analysis["anomaly_score"]
            },
            db
        )
    
    # Set amount limits based on risk (MUCH STRICTER)
    max_amount = 1000 if is_flagged else 10000  # Reduced from 5K/50K to 1K/10K
    if transaction_data.amount > max_amount:
        # Log the security event
        await log_security_event(
            "transaction_limit_exceeded", 
            user_id=current_user.id, 
            details=f"Amount: {transaction_data.amount}, Limit: {max_amount}", 
            ip_address=request.client.host, 
            db=db
        )
        
        # Log out the user for security
        await log_security_event(
            "user_auto_logout_security",
            user_id=current_user.id,
            details=f"User automatically logged out due to limit violation: ₹{transaction_data.amount:,} > ₹{max_amount:,}",
            ip_address=request.client.host,
            db=db
        )
        
        # Create response with logout message
        response_data = {
            "status": "security_violation",
            "message": f"Transaction amount exceeds limit of ₹{max_amount:,.2f} due to behavioral risk assessment",
            "action": "user_logged_out",
            "reason": "limit_violation",
            "details": f"Attempted transaction of ₹{transaction_data.amount:,} exceeds limit of ₹{max_amount:,}"
        }
        
        response = JSONResponse(content=response_data, status_code=400)
        response.delete_cookie(key="access_token")
        
        return response
    
    # Additional security: Block extremely large amounts regardless of risk
    if transaction_data.amount > 100000:  # Block amounts over 1 lakh
        # Log the security event
        await log_security_event(
            "transaction_blocked_large_amount", 
            user_id=current_user.id, 
            details=f"Large amount blocked: {transaction_data.amount}", 
            ip_address=request.client.host, 
            db=db
        )
        
        # Log out the user for security
        await log_security_event(
            "user_auto_logout_security",
            user_id=current_user.id,
            details=f"User automatically logged out due to large amount attempt: ₹{transaction_data.amount:,}",
            ip_address=request.client.host,
            db=db
        )
        
        # Create response with logout message
        response_data = {
            "status": "security_violation",
            "message": f"Large transaction amount ₹{transaction_data.amount:,.2f} requires manual approval",
            "action": "user_logged_out",
            "reason": "large_amount_attempt",
            "details": f"Attempted transaction of ₹{transaction_data.amount:,} exceeds security limits"
        }
        
        response = JSONResponse(content=response_data, status_code=403)
        response.delete_cookie(key="access_token")
        
        return response
    
    # Process transaction
    transaction_id = f"TXN_{secrets.token_hex(8).upper()}"
    txn = Transaction(
        user_id=current_user.id,
        transaction_id=transaction_id,
        transaction_type="transfer",
        amount=transaction_data.amount,
        confidence_score=confidence_score,
        is_flagged=is_flagged,
        status="completed"
    )
    db.add(txn)
    db.commit()
    
    # Log the transaction in the SQLite DB for cross-checking
    add_transaction(user_id, transaction_data.amount, "transfer")
    
    await log_security_event(
        "transaction_processed", 
        user_id=current_user.id, 
        details=f"Amount: {transaction_data.amount}, TxnID: {transaction_id}, Anomaly: {transaction_analysis['is_anomaly']}", 
        ip_address=request.client.host, 
        db=db
    )
    
    # After passing anomaly checks, store user location
    g = geocoder.ip('me')
    print("[DEBUG] Geocoder result:", g.ok, g.latlng)
    if g.ok:
        lat, lon = g.latlng
        add_user_location(current_user.id, lat, lon)
        user_history = fetch_user_locations(current_user.id, limit=10)
        print("[DEBUG] User location history:", user_history)
        print("[DEBUG] Current location:", (lat, lon))
        def is_gps_anomaly(user_history, new_location, threshold_km=50):
            from geopy.distance import geodesic
            for old_location in user_history:
                distance = geodesic(old_location, new_location).km
                print(f"[DEBUG] Distance from {old_location} to {new_location}: {distance} km")
                if distance <= threshold_km:
                    return False  # Not an anomaly
            return True  # Anomaly
        is_anomaly = is_gps_anomaly(user_history, (lat, lon))
        print(f"[DEBUG] [GPS] Anomaly detected on transfer: {is_anomaly}")
        if is_anomaly:
            await log_security_event(
                "gps_anomaly_transfer_blocked",
                user_id=current_user.id,
                details=f"GPS anomaly detected during transfer. Location: ({lat}, {lon})",
                ip_address=request.client.host,
                db=db
            )
            response_data = {
                "status": "security_violation",
                "message": "Transaction blocked due to GPS anomaly (suspicious location change detected).",
                "action": "user_logged_out",
                "reason": "gps_anomaly_detected",
                "details": f"GPS anomaly detected at location ({lat}, {lon})"
            }
            response = JSONResponse(content=response_data, status_code=403)
            response.delete_cookie(key="access_token")
            return response
    
    return {
        "status": "success",
        "transaction_id": transaction_id,
        "amount": transaction_data.amount,
        "to_account": transaction_data.to_account,
        "upi_id": transaction_data.upi_id,
        "description": transaction_data.description,
        "confidence_score": confidence_score,
        "anomaly_score": anomaly_score,
        "tap_anomaly": tap_anomaly,
        "transaction_anomaly": transaction_analysis["is_anomaly"],
        "transaction_flags": transaction_analysis["flags"],
        "is_flagged": is_flagged,
        "timestamp": datetime.utcnow().isoformat()
    }

# --- Enhanced Banking Models ---
class BeneficiaryRequest(BaseModel):
    name: str
    account_number: str
    ifsc_code: str
    bank_name: str
    account_type: str = "savings"

class LoanRequest(BaseModel):
    loan_type: str
    amount: float
    tenure_months: int
    purpose: str

class CreditCardRequest(BaseModel):
    card_type: str
    credit_limit: float

class InsuranceRequest(BaseModel):
    insurance_type: str
    coverage_amount: float
    premium_amount: float

class MutualFundRequest(BaseModel):
    fund_name: str
    amount: float
    sip_frequency: str = "monthly"

# --- Database Models for Enhanced Features ---
class Beneficiary(Base):
    __tablename__ = "beneficiaries"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    name = Column(String(100), nullable=False)
    account_number = Column(String(50), nullable=False)
    ifsc_code = Column(String(20), nullable=False)
    bank_name = Column(String(100), nullable=False)
    account_type = Column(String(20), default="savings")
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Loan(Base):
    __tablename__ = "loans"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    loan_type = Column(String(50), nullable=False)
    amount = Column(Float, nullable=False)
    tenure_months = Column(Integer, nullable=False)
    interest_rate = Column(Float, nullable=False)
    emi_amount = Column(Float, nullable=False)
    status = Column(String(20), default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)

class CreditCard(Base):
    __tablename__ = "credit_cards"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    card_number = Column(String(20), nullable=False)
    card_type = Column(String(50), nullable=False)
    credit_limit = Column(Float, nullable=False)
    available_limit = Column(Float, nullable=False)
    status = Column(String(20), default="active")
    created_at = Column(DateTime, default=datetime.utcnow)

class Insurance(Base):
    __tablename__ = "insurance"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    insurance_type = Column(String(50), nullable=False)
    coverage_amount = Column(Float, nullable=False)
    premium_amount = Column(Float, nullable=False)
    status = Column(String(20), default="active")
    created_at = Column(DateTime, default=datetime.utcnow)

class MutualFund(Base):
    __tablename__ = "mutual_funds"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    fund_name = Column(String(100), nullable=False)
    amount = Column(Float, nullable=False)
    sip_frequency = Column(String(20), default="monthly")
    status = Column(String(20), default="active")
    created_at = Column(DateTime, default=datetime.utcnow)

@app.post("/api/v2/fd/create")
async def create_fd(fd: FDRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Calculate interest rate based on tenure
    interest_rates = {12: 6.5, 24: 7.0, 36: 7.5, 60: 8.0}
    interest_rate = interest_rates.get(fd.tenure_months, 6.0)
    
    fd_obj = FixedDeposit(
        user_id=current_user.id,
        amount=fd.amount,
        tenure_months=fd.tenure_months,
        interest_rate=interest_rate
    )
    db.add(fd_obj)
    db.commit()
    
    await log_security_event(
        "fd_created",
        user_id=current_user.id,
        details=f"FD created: Amount: {fd.amount}, Tenure: {fd.tenure_months} months",
        db=db
    )
    
    return {
        "status": "success", 
        "fd_id": fd_obj.id,
        "interest_rate": interest_rate,
        "maturity_amount": fd.amount * (1 + interest_rate/100 * fd.tenure_months/12)
    }

@app.get("/api/v2/fd/list")
async def list_fds(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    fds = db.query(FixedDeposit).filter_by(user_id=current_user.id).all()
    return {"fds": [
        {
            "id": fd.id,
            "amount": fd.amount,
            "tenure_months": fd.tenure_months,
            "interest_rate": fd.interest_rate,
            "maturity_amount": fd.amount * (1 + fd.interest_rate/100 * fd.tenure_months/12),
            "created_at": fd.created_at.isoformat()
        } for fd in fds
    ]}

# --- Enhanced Banking API Endpoints ---

@app.post("/api/v2/beneficiary/add")
async def add_beneficiary(
    beneficiary: BeneficiaryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Add a new beneficiary for transfers."""
    # Check if beneficiary already exists
    existing = db.query(Beneficiary).filter_by(
        user_id=current_user.id,
        account_number=beneficiary.account_number,
        ifsc_code=beneficiary.ifsc_code
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="Beneficiary already exists")
    
    new_beneficiary = Beneficiary(
        user_id=current_user.id,
        name=beneficiary.name,
        account_number=beneficiary.account_number,
        ifsc_code=beneficiary.ifsc_code,
        bank_name=beneficiary.bank_name,
        account_type=beneficiary.account_type
    )
    
    db.add(new_beneficiary)
    db.commit()
    
    await log_security_event(
        "beneficiary_added",
        user_id=current_user.id,
        details=f"Beneficiary added: {beneficiary.name}",
        db=db
    )
    
    return {"status": "success", "beneficiary_id": new_beneficiary.id}

@app.get("/api/v2/beneficiary/list")
async def list_beneficiaries(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get list of user's beneficiaries."""
    beneficiaries = db.query(Beneficiary).filter_by(user_id=current_user.id).all()
    return {"beneficiaries": [
        {
            "id": b.id,
            "name": b.name,
            "account_number": b.account_number,
            "ifsc_code": b.ifsc_code,
            "bank_name": b.bank_name,
            "account_type": b.account_type,
            "is_verified": b.is_verified
        } for b in beneficiaries
    ]}

@app.post("/api/v2/loan/apply")
async def apply_loan(
    loan: LoanRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Apply for a loan."""
    # Calculate interest rate based on loan type
    interest_rates = {
        "personal": 12.5,
        "home": 8.5,
        "car": 10.5,
        "education": 9.5,
        "business": 13.5
    }
    
    interest_rate = interest_rates.get(loan.loan_type, 12.0)
    emi_amount = (loan.amount * interest_rate/100/12 * (1 + interest_rate/100/12)**loan.tenure_months) / ((1 + interest_rate/100/12)**loan.tenure_months - 1)
    
    new_loan = Loan(
        user_id=current_user.id,
        loan_type=loan.loan_type,
        amount=loan.amount,
        tenure_months=loan.tenure_months,
        interest_rate=interest_rate,
        emi_amount=emi_amount
    )
    
    db.add(new_loan)
    db.commit()
    
    await log_security_event(
        "loan_applied",
        user_id=current_user.id,
        details=f"Loan applied: {loan.loan_type} - {loan.amount}",
        db=db
    )
    
    return {
        "status": "success",
        "loan_id": new_loan.id,
        "emi_amount": emi_amount,
        "interest_rate": interest_rate
    }

@app.get("/api/v2/loan/list")
async def list_loans(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get user's loan history."""
    loans = db.query(Loan).filter_by(user_id=current_user.id).all()
    return {"loans": [
        {
            "id": l.id,
            "loan_type": l.loan_type,
            "amount": l.amount,
            "tenure_months": l.tenure_months,
            "interest_rate": l.interest_rate,
            "emi_amount": l.emi_amount,
            "status": l.status,
            "created_at": l.created_at.isoformat()
        } for l in loans
    ]}

@app.post("/api/v2/credit-card/apply")
async def apply_credit_card(
    card: CreditCardRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Apply for a credit card."""
    # Generate card number (in real app, this would be from card provider)
    card_number = f"4{secrets.token_hex(7).upper()}"
    
    new_card = CreditCard(
        user_id=current_user.id,
        card_number=card_number,
        card_type=card.card_type,
        credit_limit=card.credit_limit,
        available_limit=card.credit_limit
    )
    
    db.add(new_card)
    db.commit()
    
    await log_security_event(
        "credit_card_applied",
        user_id=current_user.id,
        details=f"Credit card applied: {card.card_type}",
        db=db
    )
    
    return {
        "status": "success",
        "card_id": new_card.id,
        "card_number": card_number,
        "credit_limit": card.credit_limit
    }

@app.get("/api/v2/credit-card/list")
async def list_credit_cards(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get user's credit cards."""
    cards = db.query(CreditCard).filter_by(user_id=current_user.id).all()
    return {"credit_cards": [
        {
            "id": c.id,
            "card_number": f"**** **** **** {c.card_number[-4:]}",
            "card_type": c.card_type,
            "credit_limit": c.credit_limit,
            "available_limit": c.available_limit,
            "status": c.status
        } for c in cards
    ]}

@app.post("/api/v2/insurance/apply")
async def apply_insurance(
    insurance: InsuranceRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Apply for insurance."""
    new_insurance = Insurance(
        user_id=current_user.id,
        insurance_type=insurance.insurance_type,
        coverage_amount=insurance.coverage_amount,
        premium_amount=insurance.premium_amount
    )
    
    db.add(new_insurance)
    db.commit()
    
    await log_security_event(
        "insurance_applied",
        user_id=current_user.id,
        details=f"Insurance applied: {insurance.insurance_type}",
        db=db
    )
    
    return {"status": "success", "insurance_id": new_insurance.id}

@app.get("/api/v2/insurance/list")
async def list_insurance(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get user's insurance policies."""
    policies = db.query(Insurance).filter_by(user_id=current_user.id).all()
    return {"insurance": [
        {
            "id": i.id,
            "insurance_type": i.insurance_type,
            "coverage_amount": i.coverage_amount,
            "premium_amount": i.premium_amount,
            "status": i.status
        } for i in policies
    ]}

@app.post("/api/v2/mutual-fund/invest")
async def invest_mutual_fund(
    mf: MutualFundRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Invest in mutual funds."""
    new_investment = MutualFund(
        user_id=current_user.id,
        fund_name=mf.fund_name,
        amount=mf.amount,
        sip_frequency=mf.sip_frequency
    )
    
    db.add(new_investment)
    db.commit()
    
    await log_security_event(
        "mutual_fund_invested",
        user_id=current_user.id,
        details=f"MF invested: {mf.fund_name} - {mf.amount}",
        db=db
    )
    
    return {"status": "success", "investment_id": new_investment.id}

@app.get("/api/v2/mutual-fund/list")
async def list_mutual_funds(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get user's mutual fund investments."""
    investments = db.query(MutualFund).filter_by(user_id=current_user.id).all()
    return {"mutual_funds": [
        {
            "id": mf.id,
            "fund_name": mf.fund_name,
            "amount": mf.amount,
            "sip_frequency": mf.sip_frequency,
            "status": mf.status
        } for mf in investments
    ]}

@app.get("/api/v2/account/statement")
async def get_account_statement(
    account_type: str = "savings",
    from_date: str = None,
    to_date: str = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get account statement."""
    # In real app, this would fetch from actual transaction database
    transactions = db.query(Transaction).filter_by(user_id=current_user.id).limit(50).all()
    
    return {
        "account_type": account_type,
        "from_date": from_date,
        "to_date": to_date,
        "transactions": [
            {
                "id": t.id,
                "transaction_id": t.transaction_id,
                "type": t.transaction_type,
                "amount": t.amount,
                "status": t.status,
                "created_at": t.created_at.isoformat()
            } for t in transactions
        ]
    }

@app.get("/api/v2/banking/limits")
async def get_transaction_limits(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get user's transaction limits and current usage."""
    from datetime import datetime, timedelta
    
    profile = json.loads(current_user.behavioral_profile)
    is_flagged = profile.get("tap_anomaly", False) or profile.get("behavioral_mismatch", False)
    
    # Get recent transactions (last 1 hour)
    one_hour_ago = datetime.utcnow() - timedelta(hours=1)
    recent_transactions = db.query(Transaction).filter(
        Transaction.user_id == current_user.id,
        Transaction.created_at >= one_hour_ago,
        Transaction.status == "completed"
    ).all()
    
    # Calculate current usage
    transfer_count_last_hour = len(recent_transactions)
    cumulative_amount_last_hour = sum(t.amount for t in recent_transactions)
    
    # Limits
    MAX_TRANSFERS_PER_HOUR = 5
    MAX_CUMULATIVE_AMOUNT_PER_HOUR = 50000
    
    return {
        "limits": {
            "max_transfers_per_hour": MAX_TRANSFERS_PER_HOUR,
            "max_cumulative_amount_per_hour": MAX_CUMULATIVE_AMOUNT_PER_HOUR,
            "daily_limit": 50000 if not is_flagged else 5000,
            "transaction_limit": 100000 if not is_flagged else 10000,
            "upi_limit": 100000,
            "neft_limit": 1000000,
            "rtgs_limit": 2000000,
            "is_flagged": is_flagged
        },
        "current_usage": {
            "transfers_last_hour": transfer_count_last_hour,
            "cumulative_amount_last_hour": cumulative_amount_last_hour,
            "remaining_transfers": max(0, MAX_TRANSFERS_PER_HOUR - transfer_count_last_hour),
            "remaining_amount": max(0, MAX_CUMULATIVE_AMOUNT_PER_HOUR - cumulative_amount_last_hour)
        },
        "status": {
            "frequency_limit_exceeded": transfer_count_last_hour >= MAX_TRANSFERS_PER_HOUR,
            "amount_limit_exceeded": cumulative_amount_last_hour >= MAX_CUMULATIVE_AMOUNT_PER_HOUR
        }
    }

@app.post("/api/v2/log_behavior")
async def log_behavior(
    behavior: dict,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    request: Request = None
):
    now = time.time()
    user_id = current_user.id
    last_time = user_last_log_time.get(user_id, 0)
    if now - last_time < RATE_LIMIT_SECONDS:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please wait before sending more data.")
    user_last_log_time[user_id] = now
    # Store behavior log
    log = BehaviorLog(
        user_id=user_id,
        typing_speed=behavior.get("typing_speed"),
        nav_pattern=json.dumps(behavior.get("nav_pattern")),
        swipe_pattern=json.dumps(behavior.get("swipe_pattern")),
        gps=json.dumps(behavior.get("GPS")),
        tap_speed=behavior.get("tap_speed"),
        anomaly_score=behavior.get("anomaly_score")
    )
    db.add(log)
    db.commit()
    return {"status": "logged"}

@app.post("/api/v2/analyze")
async def analyze_behavior(
    behavior: dict,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Decrypt baseline
    baseline = decrypt_baseline(current_user.baseline_behavior)
    # Prepare vectors for comparison
    def to_vector(data):
        return [
            float(data.get("typing_speed", 0)),
            float(data.get("tap_speed", 0)),
            np.mean(data.get("swipe_pattern", []) or [0]),
            float(data.get("GPS", {}).get("lat", 0)),
            float(data.get("GPS", {}).get("long", 0)),
        ]
    v1 = to_vector(baseline)
    v2 = to_vector(behavior)
    # Cosine similarity
    def cosine_similarity(a, b):
        a = np.array(a)
        b = np.array(b)
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    confidence = cosine_similarity(v1, v2)
    return {"confidence": confidence}

@app.post("/api/v2/lock_session")
async def lock_session(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    profile = json.loads(current_user.behavioral_profile)
    anomaly_score = profile.get("anomaly_score", 1.0)
    if anomaly_score == -1:
        await log_security_event(
            "anomaly_detected_lock_session",
            user_id=current_user.id,
            details="Anomaly detected during session lock attempt.",
            db=db
        )
        # Optionally, alert admins here (e.g., send email/notification)
        raise HTTPException(status_code=403, detail="Session lock blocked due to behavioral anomaly.")
    current_user.is_active = False
    db.commit()
    await log_security_event("session_locked", user_id=current_user.id, details="Session locked due to anomaly.", db=db)
    return {"status": "locked"}

# Add more endpoints as needed (FD, loan, etc.)

@app.post("/api/v2/admin/login")
async def admin_login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password_hash) or not user.is_admin:
        raise HTTPException(status_code=401, detail="Invalid admin credentials")
    token = create_access_token({"user_id": user.id, "username": user.username, "is_admin": True})
    return {"access_token": token, "token_type": "bearer"}

@app.get("/api/v2/admin/logs")
async def get_anomaly_logs(current_admin: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    behavior_logs = db.query(BehaviorLog).order_by(BehaviorLog.timestamp.desc()).limit(100).all()
    security_logs = db.query(SecurityLog).order_by(SecurityLog.timestamp.desc()).limit(100).all()
    return {
        "behavior_logs": [
            {
                "user_id": log.user_id,
                "timestamp": log.timestamp.isoformat(),
                "typing_speed": log.typing_speed,
                "nav_pattern": log.nav_pattern,
                "swipe_pattern": log.swipe_pattern,
                "gps": log.gps,
                "tap_speed": log.tap_speed,
                "anomaly_score": log.anomaly_score
            } for log in behavior_logs
        ],
        "security_logs": [
            {
                "user_id": log.user_id,
                "event_type": log.event_type,
                "ip_address": log.ip_address,
                "user_agent": log.user_agent,
                "details": log.details,
                "timestamp": log.timestamp.isoformat()
            } for log in security_logs
        ]
    }

@app.get("/api/v2/user/logs")
async def get_user_logs(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get user's own security logs."""
    security_logs = db.query(SecurityLog).filter(
        SecurityLog.user_id == current_user.id
    ).order_by(SecurityLog.timestamp.desc()).limit(50).all()
    
    return {
        "security_logs": [
            {
                "id": log.id,
                "event_type": log.event_type,
                "ip_address": log.ip_address,
                "details": log.details,
                "timestamp": log.timestamp.isoformat()
            }
            for log in security_logs
        ],
        "total_count": len(security_logs),
        "user_id": current_user.id
    }

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/admin_login", response_class=HTMLResponse)
async def admin_login_page(request: Request):
    return templates.TemplateResponse("admin_login.html", {"request": request})

@app.get("/admin_dashboard", response_class=HTMLResponse)
async def admin_dashboard_page(request: Request):
    return templates.TemplateResponse("admin_dashboard.html", {"request": request})

# --- Session Management API Endpoints ---
@app.get("/api/v2/session/status")
async def get_session_status(
    current_user: User = Depends(get_current_user),
    request: Request = None
):
    """Get current session status and remaining time."""
    # Try to get token from cookie first, then from Authorization header
    token = request.cookies.get("access_token")
    if not token:
        # Try to extract from Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
    
    if not token:
        raise HTTPException(status_code=401, detail="No session token found")
    
    remaining_seconds = get_session_remaining_time(token)
    is_expired = is_session_expired(token)
    
    # Convert seconds to MM:SS format
    minutes = remaining_seconds // 60
    seconds = remaining_seconds % 60
    formatted_time = f"{minutes:02d}:{seconds:02d}"
    
    return {
        "user_id": current_user.id,
        "username": current_user.username,
        "remaining_time": remaining_seconds,
        "remaining_time_formatted": formatted_time,
        "is_expired": is_expired,
        "session_timeout_minutes": JWT_EXPIRATION_MINUTES,
        "last_activity": datetime.utcnow().isoformat()
    }

@app.post("/api/v2/session/refresh")
async def refresh_session(
    current_user: User = Depends(get_current_user),
    request: Request = None
):
    """Refresh session token."""
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="No session token found")
    
    if is_session_expired(token):
        raise HTTPException(status_code=401, detail="Session has expired")
    
    new_token = refresh_session_token(token)
    if not new_token:
        raise HTTPException(status_code=500, detail="Failed to refresh session")
    
    response = {"status": "refreshed", "remaining_time": get_session_remaining_time(new_token)}
    
    # Set new cookie
    response_obj = JSONResponse(content=response)
    response_obj.set_cookie(key="access_token", value=new_token, httponly=True, max_age=JWT_EXPIRATION_MINUTES * 60)
    
    return response_obj

@app.post("/api/v2/session/logout")
async def logout_session(
    current_user: User = Depends(get_current_user),
    request: Request = None,
    db: Session = Depends(get_db)
):
    """Logout user and clear session."""
    await log_security_event(
        "user_logout",
        user_id=current_user.id,
        details="User logged out",
        ip_address=request.client.host,
        db=db
    )
    
    response = {"status": "logged_out"}
    response_obj = JSONResponse(content=response)
    response_obj.delete_cookie(key="access_token")
    
    return response_obj

@app.get("/api/v2/session/check")
async def check_session_validity(
    request: Request = None
):
    """Check if current session is valid without requiring authentication."""
    token = request.cookies.get("access_token")
    if not token:
        return {"valid": False, "reason": "no_token"}
    
    if is_session_expired(token):
        return {"valid": False, "reason": "expired"}
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return {
            "valid": True,
            "user_id": payload.get("user_id"),
            "remaining_time": get_session_remaining_time(token)
        }
    except:
        return {"valid": False, "reason": "invalid_token"}

# --- GPS Anomaly Detection API Endpoints ---
@app.post("/api/v2/gps/update")
async def update_gps_location(
    gps_data: dict,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    request: Request = None
):
    """Update user's GPS location and check for anomalies."""
    lat = gps_data.get("latitude")
    lon = gps_data.get("longitude")
    
    if not lat or not lon:
        raise HTTPException(status_code=400, detail="Latitude and longitude required")
    
    # Analyze GPS for anomalies
    gps_analysis = update_user_gps_location(current_user.id, lat, lon, db)
    
    # Log GPS update
    await log_security_event(
        "gps_location_updated",
        user_id=current_user.id,
        details=f"GPS updated: ({lat}, {lon}), Anomaly: {gps_analysis['is_anomaly']}",
        ip_address=request.client.host,
        db=db
    )
    
    return {
        "status": "success",
        "gps_analysis": gps_analysis,
        "message": "GPS location updated and analyzed"
    }

@app.get("/api/v2/gps/status")
async def get_gps_status(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's GPS status and anomaly information."""
    profile = json.loads(current_user.behavioral_profile)
    
    return {
        "user_id": current_user.id,
        "gps_anomaly": profile.get("gps_anomaly", False),
        "gps_anomaly_score": profile.get("gps_anomaly_score", 0.0),
        "gps_flags": profile.get("gps_flags", []),
        "last_gps_check": profile.get("last_gps_check"),
        "location_history_count": len(user_gps_history.get(current_user.id, [])),
        "current_location": user_gps_history.get(current_user.id, [])[-1] if user_gps_history.get(current_user.id) else None
    }

@app.post("/api/v2/gps/clear-anomaly")
async def clear_gps_anomaly(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Clear GPS anomaly flags (for legitimate location changes)."""
    profile = json.loads(current_user.behavioral_profile)
    profile["gps_anomaly"] = False
    profile["gps_anomaly_score"] = 0.0
    profile["gps_flags"] = []
    profile["gps_anomaly_cleared"] = datetime.utcnow().isoformat()
    
    current_user.behavioral_profile = json.dumps(profile)
    db.commit()
    
    await log_security_event(
        "gps_anomaly_cleared",
        user_id=current_user.id,
        details="GPS anomaly flags cleared by user",
        ip_address="unknown",
        db=db
    )
    
    return {
        "status": "success",
        "message": "GPS anomaly flags cleared"
    }

# --- Navigation Anomaly Detection API Endpoints ---
@app.post("/api/v2/navigation/update")
async def update_navigation(
    navigation_data: dict,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    request: Request = None
):
    """Update user's navigation data and check for anomalies."""
    required_fields = ["screen", "transition_type", "navigation_depth"]
    for field in required_fields:
        if field not in navigation_data:
            raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
    
    # Add timestamp if not provided
    if "timestamp" not in navigation_data:
        navigation_data["timestamp"] = datetime.utcnow().isoformat()
    
    # Add time_spent if not provided
    if "time_spent" not in navigation_data:
        navigation_data["time_spent"] = 1.0
    
    # Analyze navigation for anomalies
    navigation_analysis = update_user_navigation(current_user.id, navigation_data, db)
    
    # Log navigation update
    await log_security_event(
        "navigation_updated",
        user_id=current_user.id,
        details=f"Navigation: {navigation_data['screen']}, Anomaly: {navigation_analysis['is_anomaly']}",
        ip_address=request.client.host,
        db=db
    )
    
    return {
        "status": "success",
        "navigation_analysis": navigation_analysis,
        "message": "Navigation data updated and analyzed"
    }

@app.get("/api/v2/navigation/status")
async def get_navigation_status(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's navigation status and anomaly information."""
    profile = json.loads(current_user.behavioral_profile)
    
    return {
        "user_id": current_user.id,
        "navigation_anomaly": profile.get("navigation_anomaly", False),
        "navigation_anomaly_score": profile.get("navigation_anomaly_score", 0.0),
        "navigation_flags": profile.get("navigation_flags", []),
        "last_navigation_check": profile.get("last_navigation_check"),
        "navigation_count": len(user_navigation_logs.get(current_user.id, [])),
        "recent_screens": [log.get("screen", "") for log in user_navigation_logs.get(current_user.id, [])[-5:]]
    }

@app.post("/api/v2/navigation/clear-anomaly")
async def clear_navigation_anomaly(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Clear navigation anomaly flags (for legitimate navigation patterns)."""
    profile = json.loads(current_user.behavioral_profile)
    profile["navigation_anomaly"] = False
    profile["navigation_anomaly_score"] = 0.0
    profile["navigation_flags"] = []
    profile["navigation_anomaly_cleared"] = datetime.utcnow().isoformat()
    
    current_user.behavioral_profile = json.dumps(profile)
    db.commit()
    
    await log_security_event(
        "navigation_anomaly_cleared",
        user_id=current_user.id,
        details="Navigation anomaly flags cleared by user",
        ip_address="unknown",
        db=db
    )
    
    return {
        "status": "success",
        "message": "Navigation anomaly flags cleared"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080) 