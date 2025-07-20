import time
import math
from datetime import datetime

# Store legitimate users
legitimate_users = {}
# Store bot detection logs
detection_logs = []

def get_tap_timestamps():
    """Get tap timestamps from user input."""
    timestamps = []
    print("\n️  HUMAN VERIFICATION TEST")
    print("Tap naturally (like a human would)...")
    print("Enter timestamps for each tap (e.g., 0.0, 0.8, 1.5, 2.3)")
    print("Type 'done' when finished\n")
    
    tap_count = 1
    while True:
        try:
            user_input = input(f"Tap {tap_count} timestamp: ")
            if user_input.lower() == 'done':
                break
            timestamp = float(user_input)
            timestamps.append(timestamp)
            tap_count += 1
        except ValueError:
            print("Invalid input. Please enter a number or 'done'.")
    
    return timestamps

def analyze_pattern(timestamps):
    """Analyze tap pattern to determine if human or bot."""
    if len(timestamps) < 3:
        return None, "Need at least 3 taps"
    
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
    
    return {
        'avg_interval': avg_interval,
        'min_interval': min_interval,
        'max_interval': max_interval,
        'variance': variance,
        'tap_speed': tap_speed,
        'intervals': intervals,
        'total_taps': len(timestamps)
    }

def is_bot(pattern):
    """Determine if pattern indicates a bot."""
    if not pattern:
        return True, "Invalid pattern"
    
    bot_reasons = []
    
    # 1. Too Perfect Timing (BOT)
    if pattern['variance'] < 0.01:
        bot_reasons.append("Too perfect timing (no natural variation)")
    
    # 2. Too Fast Taps (BOT)
    if pattern['min_interval'] < 0.05:
        bot_reasons.append("Unnaturally fast taps (< 50ms)")
    
    # 3. Too Consistent (BOT)
    if pattern['max_interval'] - pattern['min_interval'] < 0.1:
        bot_reasons.append("Too consistent rhythm (suspicious)")
    
    # 4. Unrealistic Speed (BOT)
    if pattern['tap_speed'] > 8.0:
        bot_reasons.append("Unrealistic tap speed (> 8 taps/sec)")
    
    # 5. Perfect Intervals (BOT)
    if all(abs(interval - pattern['avg_interval']) < 0.01 for interval in pattern['intervals']):
        bot_reasons.append("Perfect intervals (machine-like)")
    
    # If any bot indicators found
    if bot_reasons:
        return True, bot_reasons
    
    return False, ["Human-like behavior detected"]

def verify_user_or_bot():
    """Main verification function."""
    print("\n" + "="*50)
    print("    🤖 BOT DETECTION SYSTEM")
    print("="*50)
    
    timestamps = get_tap_timestamps()
    if len(timestamps) < 3:
        print("❌ Need at least 3 taps for verification.")
        return
    
    pattern = analyze_pattern(timestamps)
    if not pattern:
        print("❌ Failed to analyze pattern.")
        return
    
    is_bot_user, reasons = is_bot(pattern)
    
    print(f"\n=== VERIFICATION RESULTS ===")
    print(f"Average interval: {pattern['avg_interval']:.3f}s")
    print(f"Min interval: {pattern['min_interval']:.3f}s")
    print(f"Max interval: {pattern['max_interval']:.3f}s")
    print(f"Tap speed: {pattern['tap_speed']:.2f} taps/sec")
    print(f"Variance: {pattern['variance']:.3f}")
    
    if is_bot_user:
        print(f"\n🚨 BOT DETECTED!")
        print("Reasons:")
        for reason in reasons:
            print(f"  ❌ {reason}")
        print("\nAction: BLOCKED")
        
        # Log bot detection
        detection_logs.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'BOT',
            'pattern': pattern,
            'reasons': reasons
        })
        
        return False
    else:
        print(f"\n✅ HUMAN VERIFIED!")
        print("Reasons:")
        for reason in reasons:
            print(f"  ✅ {reason}")
        print("\nAction: ALLOWED")
        
        # Log human verification
        detection_logs.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'HUMAN',
            'pattern': pattern,
            'reasons': reasons
        })
        
        return True

def register_human_user():
    """Register a human user."""
    user_id = input("Enter user ID: ")
    print(f"\n🔐 Registering human user: {user_id}")
    
    timestamps = get_tap_timestamps()
    if len(timestamps) < 3:
        print("❌ Need at least 3 taps for registration.")
        return
    
    pattern = analyze_pattern(timestamps)
    if not pattern:
        print("❌ Failed to analyze pattern.")
        return
    
    is_bot_user, reasons = is_bot(pattern)
    if is_bot_user:
        print(f"❌ BOT DETECTED during registration!")
        print("Cannot register bot patterns.")
        return
    
    legitimate_users[user_id] = pattern
    print(f"\n✅ Human user {user_id} registered successfully!")
    print("Pattern saved for future verification.")

def verify_registered_user():
    """Verify against registered user pattern."""
    user_id = input("Enter user ID to verify: ")
    
    if user_id not in legitimate_users:
        print(f"❌ User {user_id} not found.")
        return
    
    print(f"\n🔍 Verifying user: {user_id}")
    
    timestamps = get_tap_timestamps()
    if len(timestamps) < 3:
        print("❌ Need at least 3 taps for verification.")
        return
    
    current_pattern = analyze_pattern(timestamps)
    if not current_pattern:
        print("❌ Failed to analyze current pattern.")
        return
    
    stored_pattern = legitimate_users[user_id]
    
    # First check if it's a bot
    is_bot_user, reasons = is_bot(current_pattern)
    if is_bot_user:
        print(f"🚨 BOT DETECTED during user verification!")
        print("User verification failed - potential bot attack")
        return
    
    # Compare with stored pattern
    interval_diff = abs(current_pattern['avg_interval'] - stored_pattern['avg_interval'])
    speed_diff = abs(current_pattern['tap_speed'] - stored_pattern['tap_speed'])
    
    similarity = max(0, 1 - (interval_diff + speed_diff) / 2)
    
    print(f"\n=== USER VERIFICATION RESULTS ===")
    print(f"Similarity Score: {similarity:.3f}")
    
    if similarity >= 0.6:
        print("✅ USER VERIFIED: Pattern matches registered user")
        print("Action: ALLOWED")
    else:
        print("❌ USER VERIFICATION FAILED: Pattern doesn't match")
        print("Action: BLOCKED")

def view_detection_logs():
    """View all detection logs."""
    if not detection_logs:
        print("No detection logs yet.")
        return
    
    print(f"\n=== DETECTION LOGS ({len(detection_logs)} entries) ===")
    for i, log in enumerate(detection_logs, 1):
        print(f"\n{i}. Time: {log['timestamp']}")
        print(f"   Type: {log['type']}")
        print(f"   Tap Speed: {log['pattern']['tap_speed']:.2f} taps/sec")
        print(f"   Avg Interval: {log['pattern']['avg_interval']:.3f}s")

def menu():
    """Main menu."""
    while True:
        print("\n" + "="*50)
        print("    🤖 BOT DETECTION SYSTEM")
        print("="*50)
        print("1. Verify User or Bot")
        print("2. Register Human User")
        print("3. Verify Registered User")
        print("4. View Detection Logs")
        print("5. Exit")
        
        choice = input("\nChoose an option (1-5): ")
        
        if choice == "1":
            verify_user_or_bot()
        elif choice == "2":
            register_human_user()
        elif choice == "3":
            verify_registered_user()
        elif choice == "4":
            view_detection_logs()
        elif choice == "5":
            print("Goodbye! Stay safe from bots! 🤖")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    print("🤖 Welcome to Bot Detection System!")
    print("This system verifies if you're human or a bot.")
    print("Bots have too-perfect timing and unnatural speeds.")
    menu()