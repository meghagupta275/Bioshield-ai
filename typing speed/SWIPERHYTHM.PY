import time
import math
from datetime import datetime

# Store user swipe patterns
user_profiles = {}

def get_swipe_timestamps():
    """Simulate collecting swipe timestamps from user input."""
    timestamps = []
    print("Enter swipe timestamps (press Enter after each swipe, type 'done' to finish):")
    print("Format: Enter the time in seconds when you made each swipe")
    
    swipe_count = 1
    while True:
        try:
            user_input = input(f"Swipe {swipe_count} timestamp (or 'done'): ")
            if user_input.lower() == 'done':
                break
            timestamp = float(user_input)
            timestamps.append(timestamp)
            swipe_count += 1
        except ValueError:
            print("Invalid input. Please enter a number or 'done'.")
    
    return timestamps

def calculate_swipe_rhythm(timestamps):
    """Calculate swipe rhythm metrics from timestamps."""
    if len(timestamps) < 2:
        return None
    
    # Calculate intervals between swipes
    intervals = []
    for i in range(1, len(timestamps)):
        interval = timestamps[i] - timestamps[i-1]
        intervals.append(interval)
    
    # Calculate rhythm metrics
    avg_interval = sum(intervals) / len(intervals)
    variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
    std_dev = math.sqrt(variance)
    
    # Calculate swipe speed (swipes per second)
    total_time = timestamps[-1] - timestamps[0]
    swipe_speed = (len(timestamps) - 1) / total_time if total_time > 0 else 0
    
    rhythm_profile = {
        'avg_interval': avg_interval,
        'variance': variance,
        'std_dev': std_dev,
        'swipe_speed': swipe_speed,
        'intervals': intervals,
        'total_swipes': len(timestamps)
    }
    
    return rhythm_profile

def calculate_similarity(profile1, profile2):
    """Calculate similarity score between two swipe rhythm profiles."""
    if not profile1 or not profile2:
        return 0.0
    
    # Compare key metrics
    interval_diff = abs(profile1['avg_interval'] - profile2['avg_interval'])
    variance_diff = abs(profile1['variance'] - profile2['variance'])
    speed_diff = abs(profile1['swipe_speed'] - profile2['swipe_speed'])
    
    # Normalize differences (you can adjust these thresholds)
    max_interval = max(profile1['avg_interval'], profile2['avg_interval'])
    max_variance = max(profile1['variance'], profile2['variance'])
    max_speed = max(profile1['swipe_speed'], profile2['swipe_speed'])
    
    # Calculate similarity scores (0-1, where 1 is identical)
    interval_similarity = max(0, 1 - (interval_diff / max_interval if max_interval > 0 else 0))
    variance_similarity = max(0, 1 - (variance_diff / max_variance if max_variance > 0 else 0))
    speed_similarity = max(0, 1 - (speed_diff / max_speed if max_speed > 0 else 0))
    
    # Weighted average
    overall_similarity = (interval_similarity * 0.4 + variance_similarity * 0.3 + speed_similarity * 0.3)
    
    return overall_similarity

def register_user():
    """Register a new user's swipe rhythm pattern."""
    user_id = input("Enter user ID: ")
    print(f"\nRegistering swipe rhythm for user: {user_id}")
    print("Please perform a series of swipes...")
    
    timestamps = get_swipe_timestamps()
    if len(timestamps) < 3:
        print("Need at least 3 swipes for a reliable pattern.")
        return
    
    rhythm_profile = calculate_swipe_rhythm(timestamps)
    if rhythm_profile:
        user_profiles[user_id] = rhythm_profile
        print(f"\n✅ User {user_id} registered successfully!")
        print(f"Average interval: {rhythm_profile['avg_interval']:.3f}s")
        print(f"Swipe speed: {rhythm_profile['swipe_speed']:.2f} swipes/sec")
        print(f"Variance: {rhythm_profile['variance']:.3f}")
    else:
        print("❌ Failed to create rhythm profile.")

def verify_user():
    """Verify if a user's swipe rhythm matches their registered pattern."""
    user_id = input("Enter user ID to verify: ")
    
    if user_id not in user_profiles:
        print(f"❌ User {user_id} not found in database.")
        return
    
    print(f"\nVerifying swipe rhythm for user: {user_id}")
    print("Please perform the same swipe pattern...")
    
    timestamps = get_swipe_timestamps()
    if len(timestamps) < 3:
        print("Need at least 3 swipes for verification.")
        return
    
    trial_profile = calculate_swipe_rhythm(timestamps)
    if not trial_profile:
        print("❌ Failed to create trial profile.")
        return
    
    stored_profile = user_profiles[user_id]
    similarity = calculate_similarity(stored_profile, trial_profile)
    
    print(f"\n=== Verification Results ===")
    print(f"Similarity Score: {similarity:.3f}")
    
    if similarity >= 0.85:
        print("✅ HIGH CONFIDENCE: User verified successfully!")
        print("Verdict: GENUINE USER")
    elif similarity >= 0.70:
        print("⚠️  MEDIUM CONFIDENCE: User likely genuine but needs additional verification.")
        print("Verdict: PROCEED WITH CAUTION")
    else:
        print("❌ LOW CONFIDENCE: User verification failed.")
        print("Verdict: POTENTIAL IMPOSTOR")

def view_profiles():
    """View all registered user profiles."""
    if not user_profiles:
        print("No users registered yet.")
        return
    
    print("\n=== Registered User Profiles ===")
    for user_id, profile in user_profiles.items():
        print(f"\nUser: {user_id}")
        print(f"  Average interval: {profile['avg_interval']:.3f}s")
        print(f"  Swipe speed: {profile['swipe_speed']:.2f} swipes/sec")
        print(f"  Variance: {profile['variance']:.3f}")
        print(f"  Total swipes: {profile['total_swipes']}")

def menu():
    """Main menu for the swipe rhythm authentication system."""
    while True:
        print("\n" + "="*50)
        print("    SWIPE RHYTHM AUTHENTICATION SYSTEM")
        print("="*50)
        print("1. Register new user")
        print("2. Verify user")
        print("3. View all profiles")
        print("4. Exit")
        
        choice = input("\nChoose an option (1-4): ")
        
        if choice == "1":
            register_user()
        elif choice == "2":
            verify_user()
        elif choice == "3":
            view_profiles()
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    print("Welcome to Swipe Rhythm Authentication System!")
    print("This system tracks your swipe timing patterns to verify your identity.")
    menu()