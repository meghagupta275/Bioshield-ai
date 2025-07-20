from datetime import datetime

reference_sequence = []
trial_sequence = []

def input_log(logs, label=""):
    """Prompt user to enter a single navigation log and add to the given list."""
    print(f"Add a log to {label} sequence. Type 'done' as screen name to stop adding.")
    screen = input("Screen name: ")
    if screen.lower() == "done":
        return False  # Signal to stop adding
    timestamp = input("Timestamp (YYYY-MM-DDTHH:MM:SS) [leave blank for now]: ")
    if not timestamp:
        timestamp = datetime.now().isoformat(timespec='seconds')
    try:
        time_spent = float(input("Time spent on last screen (seconds): "))
    except ValueError:
        print("Invalid input. Setting time_spent=1.0")
        time_spent = 1.0
    transition_type = input("Transition type (forward/back/home/random): ")
    navigation_depth = int(input("Navigation depth (0=home, 1=settings, etc.): "))
    gesture_type = input("Gesture type (tap/swipe_back/swipe_left/swipe_right): ")
    log = {
        "screen": screen,
        "timestamp": timestamp,
        "time_spent": time_spent,
        "transition_type": transition_type,
        "navigation_depth": navigation_depth,
        "gesture_type": gesture_type
    }
    logs.append(log)
    print("Log added.\n")
    return True

def print_logs(logs):
    if not logs:
        print("No logs yet.\n")
        return
    for i, log in enumerate(logs, 1):
        print(f"{i}. {log}")
    print()

def confidence_score(ref, trial):
    """Return confidence score (0.0 to 1.0) based on matching screens in order."""
    if not ref or not trial:
        return 0.0
    min_len = min(len(ref), len(trial))
    match_count = 0
    for i in range(min_len):
        if ref[i]["screen"] == trial[i]["screen"]:
            match_count += 1
    # Penalize for length difference
    total = max(len(ref), len(trial))
    if total == 0:
        return 0.0
    return match_count / total

def is_same_user(ref, trial):
    """Check if the trial navigation matches the reference navigation (screen order) and print confidence."""
    if not ref or not trial:
        print("Both sequences must be present to check.\n")
        return False
    conf = confidence_score(ref, trial)
    print(f"Confidence score: {conf:.2f}")
    if conf == 1.0:
        print("✅ User navigation matches the reference. User is likely the same.\n")
        return True
    elif conf >= 0.7:
        print("⚠️  User navigation is similar but not identical. Possible match.\n")
        return False
    else:
        print("❌ User navigation does NOT match the reference. User is likely different.\n")
        return False

def menu():
    global reference_sequence, trial_sequence
    while True:
        print("=== Navigation User Check Menu ===")
        print("1. Add log to reference sequence")
        print("2. View reference sequence")
        print("3. Add log to trial sequence")
        print("4. View trial sequence")
        print("5. Check if user is same (compare sequences)")
        print("6. Clear both sequences")
        print("7. Exit")
        choice = input("Choose an option (1-7): ")
        print()
        if choice == "1":
            input_log(reference_sequence, label="reference")
        elif choice == "2":
            print("Reference sequence:")
            print_logs(reference_sequence)
        elif choice == "3":
            input_log(trial_sequence, label="trial")
        elif choice == "4":
            print("Trial sequence:")
            print_logs(trial_sequence)
        elif choice == "5":
            is_same_user(reference_sequence, trial_sequence)
        elif choice == "6":
            reference_sequence = []
            trial_sequence = []
            print("Both sequences cleared.\n")
        elif choice == "7":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Try again.\n")

if __name__ == "__main__":
    menu()