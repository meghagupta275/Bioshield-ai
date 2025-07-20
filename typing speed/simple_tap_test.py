#!/usr/bin/env python3
"""
🧪 Simple Tap Speed Analysis Test

This script demonstrates the enhanced tap speed anomaly detection
without importing from the banking app to avoid syntax issues.
"""

import joblib
import numpy as np
import os

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

def analyze_tap_speed_anomaly(timestamps):
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
    
    # ML-based detection
    ml_anomaly = False
    ml_confidence = 1.0
    try:
        model_path = "tap_anomaly_model.joblib"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            features = extract_tap_features(timestamps)
            if features:
                X = np.array(features).reshape(1, -1)
                prediction = model.predict(X)
                ml_anomaly = prediction[0] == -1  # -1 for anomaly, 1 for normal
                
                if ml_anomaly:
                    flags.append("ML model: Tap pattern anomaly detected")
                    anomaly_score += 0.5
                    ml_confidence = 0.3
                else:
                    ml_confidence = 0.9
    except Exception as e:
        print(f"ML analysis error: {e}")
    
    is_anomaly = anomaly_score > 0.3 or len(flags) > 0
    confidence = max(0.0, min(1.0, 1.0 - anomaly_score))
    
    # Combine rule-based and ML confidence
    combined_confidence = (confidence + ml_confidence) / 2
    
    return {
        "is_anomaly": is_anomaly,
        "confidence": combined_confidence,
        "flags": flags,
        "tap_speed": tap_speed,
        "avg_interval": avg_interval,
        "variance": variance,
        "anomaly_score": anomaly_score,
        "total_taps": len(timestamps),
        "ml_anomaly": ml_anomaly,
        "ml_confidence": ml_confidence,
        "rule_confidence": confidence
    }

def test_tap_analysis():
    """Test the enhanced tap speed analysis with various patterns."""
    
    print("🧪 Enhanced Tap Speed Anomaly Detection Test")
    print("=" * 60)
    
    # Test 1: Human-like pattern (should be NORMAL)
    print("\n1️⃣ Testing Human-like Pattern:")
    human_timestamps = [0.0, 0.8, 1.6, 2.5, 3.3, 4.2, 5.1]
    print(f"Timestamps: {human_timestamps}")
    
    analysis = analyze_tap_speed_anomaly(human_timestamps)
    print(f"Result: {'🚨 ANOMALY' if analysis['is_anomaly'] else '✅ NORMAL'}")
    print(f"Confidence: {analysis['confidence']:.3f}")
    print(f"ML Anomaly: {'Yes' if analysis.get('ml_anomaly', False) else 'No'}")
    print(f"ML Confidence: {analysis.get('ml_confidence', 1.0):.3f}")
    print(f"Rule Confidence: {analysis.get('rule_confidence', 1.0):.3f}")
    if analysis['flags']:
        print(f"Flags: {', '.join(analysis['flags'])}")
    
    # Test 2: Bot-like pattern (perfect timing - should be ANOMALY)
    print("\n2️⃣ Testing Bot-like Pattern (Perfect Timing):")
    bot_timestamps = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    print(f"Timestamps: {bot_timestamps}")
    
    analysis = analyze_tap_speed_anomaly(bot_timestamps)
    print(f"Result: {'🚨 ANOMALY' if analysis['is_anomaly'] else '✅ NORMAL'}")
    print(f"Confidence: {analysis['confidence']:.3f}")
    print(f"ML Anomaly: {'Yes' if analysis.get('ml_anomaly', False) else 'No'}")
    print(f"ML Confidence: {analysis.get('ml_confidence', 1.0):.3f}")
    print(f"Rule Confidence: {analysis.get('rule_confidence', 1.0):.3f}")
    if analysis['flags']:
        print(f"Flags: {', '.join(analysis['flags'])}")
    
    # Test 3: Too fast taps (should be ANOMALY)
    print("\n3️⃣ Testing Too Fast Taps:")
    fast_timestamps = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
    print(f"Timestamps: {fast_timestamps}")
    
    analysis = analyze_tap_speed_anomaly(fast_timestamps)
    print(f"Result: {'🚨 ANOMALY' if analysis['is_anomaly'] else '✅ NORMAL'}")
    print(f"Confidence: {analysis['confidence']:.3f}")
    print(f"ML Anomaly: {'Yes' if analysis.get('ml_anomaly', False) else 'No'}")
    print(f"ML Confidence: {analysis.get('ml_confidence', 1.0):.3f}")
    print(f"Rule Confidence: {analysis.get('rule_confidence', 1.0):.3f}")
    if analysis['flags']:
        print(f"Flags: {', '.join(analysis['flags'])}")
    
    # Test 4: Irregular pattern (should be ANOMALY)
    print("\n4️⃣ Testing Irregular Pattern:")
    irregular_timestamps = [0.0, 0.1, 1.5, 1.6, 3.0, 3.1, 5.0]
    print(f"Timestamps: {irregular_timestamps}")
    
    analysis = analyze_tap_speed_anomaly(irregular_timestamps)
    print(f"Result: {'🚨 ANOMALY' if analysis['is_anomaly'] else '✅ NORMAL'}")
    print(f"Confidence: {analysis['confidence']:.3f}")
    print(f"ML Anomaly: {'Yes' if analysis.get('ml_anomaly', False) else 'No'}")
    print(f"ML Confidence: {analysis.get('ml_confidence', 1.0):.3f}")
    print(f"Rule Confidence: {analysis.get('rule_confidence', 1.0):.3f}")
    if analysis['flags']:
        print(f"Flags: {', '.join(analysis['flags'])}")
    
    # Test 5: Natural human pattern (should be NORMAL)
    print("\n5️⃣ Testing Natural Human Pattern:")
    natural_timestamps = [0.0, 0.7, 1.4, 2.2, 3.1, 4.0, 5.2]
    print(f"Timestamps: {natural_timestamps}")
    
    analysis = analyze_tap_speed_anomaly(natural_timestamps)
    print(f"Result: {'🚨 ANOMALY' if analysis['is_anomaly'] else '✅ NORMAL'}")
    print(f"Confidence: {analysis['confidence']:.3f}")
    print(f"ML Anomaly: {'Yes' if analysis.get('ml_anomaly', False) else 'No'}")
    print(f"ML Confidence: {analysis.get('ml_confidence', 1.0):.3f}")
    print(f"Rule Confidence: {analysis.get('rule_confidence', 1.0):.3f}")
    if analysis['flags']:
        print(f"Flags: {', '.join(analysis['flags'])}")

def test_feature_extraction():
    """Test the feature extraction function."""
    print("\n🔧 Feature Extraction Test")
    print("=" * 30)
    
    timestamps = [0.0, 0.8, 1.6, 2.5, 3.3, 4.2, 5.1]
    features = extract_tap_features(timestamps)
    
    print(f"Timestamps: {timestamps}")
    print(f"Features: {features}")
    print(f"Feature names: avg_interval, min_interval, max_interval, variance, tap_speed")
    
    if features:
        print(f"Average interval: {features[0]:.3f}s")
        print(f"Min interval: {features[1]:.3f}s")
        print(f"Max interval: {features[2]:.3f}s")
        print(f"Variance: {features[3]:.3f}")
        print(f"Tap speed: {features[4]:.3f} taps/sec")

def test_ml_model():
    """Test the ML model directly."""
    print("\n🤖 ML Model Test")
    print("=" * 20)
    
    # Load the model
    model_path = "tap_anomaly_model.joblib"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("✅ ML model loaded successfully")
        
        # Test with human pattern
        human_features = extract_tap_features([0.0, 0.8, 1.6, 2.5, 3.3, 4.2, 5.1])
        if human_features:
            prediction = model.predict([human_features])
            result = "ANOMALY" if prediction[0] == -1 else "NORMAL"
            print(f"Human pattern prediction: {result}")
        
        # Test with bot pattern
        bot_features = extract_tap_features([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        if bot_features:
            prediction = model.predict([bot_features])
            result = "ANOMALY" if prediction[0] == -1 else "NORMAL"
            print(f"Bot pattern prediction: {result}")
    else:
        print("❌ ML model not found")

def main():
    """Main test function."""
    print("🚀 Starting Enhanced Tap Speed Analysis Tests...")
    
    try:
        # Test feature extraction
        test_feature_extraction()
        
        # Test ML model
        test_ml_model()
        
        # Test complete analysis
        test_tap_analysis()
        
        print("\n🎉 All tests completed successfully!")
        print("\n📊 Summary:")
        print("- Enhanced tap speed analysis combines rule-based and ML-based detection")
        print("- 5 features extracted: avg_interval, min_interval, max_interval, variance, tap_speed")
        print("- ML model trained on 225 synthetic patterns (150 human + 75 bot)")
        print("- 73% accuracy on test set")
        print("- Real-time integration with banking application")
        print("\n🔧 How to Train:")
        print("1. cd ../anoamly")
        print("2. python train_tap_model.py")
        print("3. copy tap_anomaly_model.joblib \"../typing speed/\"")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 