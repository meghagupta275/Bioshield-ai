// --- Behavioral Data Collection Utilities ---
let typingTimestamps = [];
let tapTimestamps = [];
let swipePattern = [];
let navPattern = [];
let gpsData = {};
let testActive = false;
let testTimer = null;
let testDuration = 60; // seconds

// --- Registration Behavioral Test ---
const startTestBtn = document.getElementById('startTestBtn');
const behavioralTestSection = document.getElementById('behavioralTestSection');
const typingArea = document.getElementById('typingArea');
const testStatus = document.getElementById('testStatus');
const timerDisplay = document.getElementById('timer');
const registerBtn = document.getElementById('registerBtn');
const registerForm = document.getElementById('registerForm');

if (startTestBtn) {
    startTestBtn.onclick = function() {
        startBehavioralTest();
    };
}

function startBehavioralTest() {
    testActive = true;
    typingTimestamps = [];
    tapTimestamps = [];
    swipePattern = [];
    navPattern = [];
    behavioralTestSection.style.display = 'block';
    testStatus.textContent = 'Test running...';
    registerBtn.disabled = true;
    typingArea.value = '';
    typingArea.focus();
    let timeLeft = testDuration;
    timerDisplay.textContent = `Time left: ${timeLeft}s`;
    testTimer = setInterval(() => {
        timeLeft--;
        timerDisplay.textContent = `Time left: ${timeLeft}s`;
        if (timeLeft <= 0) {
            endBehavioralTest();
        }
    }, 1000);
}

function endBehavioralTest() {
    testActive = false;
    clearInterval(testTimer);
    testStatus.textContent = 'Test complete! You can now register.';
    registerBtn.disabled = false;
    behavioralTestSection.style.display = 'none';
}

if (typingArea) {
    typingArea.addEventListener('keydown', function(e) {
        if (testActive) {
            typingTimestamps.push(Date.now());
        }
    });
}

// Tap/Swipe (for mobile)
document.addEventListener('touchstart', function(e) {
    if (testActive) {
        tapTimestamps.push(Date.now());
    }
});
document.addEventListener('touchmove', function(e) {
    if (testActive) {
        if (e.touches.length === 1) {
            swipePattern.push(e.touches[0].clientX);
        }
    }
});

// Navigation pattern (simulate by tracking focus/blur)
window.addEventListener('focus', function() {
    if (testActive) navPattern.push('focus');
});
window.addEventListener('blur', function() {
    if (testActive) navPattern.push('blur');
});

// GPS
function getGPS() {
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(function(pos) {
            gpsData = {
                lat: pos.coords.latitude,
                long: pos.coords.longitude
            };
        });
    }
}
getGPS();

// --- Registration Form Submission ---
if (registerForm) {
    registerForm.onsubmit = async function(e) {
        e.preventDefault();
        // Calculate typing speed
        let typingSpeed = 0;
        if (typingTimestamps.length > 1) {
            let intervals = typingTimestamps.slice(1).map((t, i) => t - typingTimestamps[i]);
            typingSpeed = intervals.reduce((a, b) => a + b, 0) / intervals.length;
        }
        // Calculate tap speed
        let tapSpeed = 0;
        if (tapTimestamps.length > 1) {
            let intervals = tapTimestamps.slice(1).map((t, i) => t - tapTimestamps[i]);
            tapSpeed = intervals.reduce((a, b) => a + b, 0) / intervals.length;
        }
        // Prepare baseline
        const baseline_behavior = {
            typing_speed: typingSpeed,
            tap_speed: tapSpeed,
            nav_pattern: navPattern,
            swipe_pattern: swipePattern,
            GPS: gpsData
        };
        const formData = new FormData(registerForm);
        const payload = {
            name: formData.get('name'),
            username: formData.get('username'),
            email: formData.get('email'),
            password: formData.get('password'),
            auth_type: formData.get('auth_type'),
            behavioral_data: {
                timestamps: typingTimestamps
            },
            baseline_behavior
        };
        registerBtn.disabled = true;
        testStatus.textContent = 'Registering...';
        const res = await fetch('/api/v2/register', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (res.ok && data.status === 'success') {
            testStatus.textContent = `Registration successful! Confidence: ${data.confidence.toFixed(1)}%. Please login.`;
            registerForm.reset();
        } else {
            testStatus.textContent = data.detail || data.message || 'Registration failed.';
        }
        registerBtn.disabled = false;
    };
}

// --- Login and Dashboard Behavioral Monitoring ---
const loginForm = document.getElementById('loginForm');
const loginStatus = document.getElementById('loginStatus');
const dashboard = document.getElementById('dashboard');
const dashboardStatus = document.getElementById('dashboardStatus');
const logoutBtn = document.getElementById('logoutBtn');
let accessToken = null;
let monitorInterval = null;

if (loginForm) {
    loginForm.onsubmit = async function(e) {
        e.preventDefault();
        const formData = new FormData(loginForm);
        const payload = {
            username: formData.get('username'),
            password: formData.get('password'),
            auth_type: 'typing',
            behavioral_data: { timestamps: [] }
        };
        loginStatus.textContent = 'Logging in...';
        
        try {
            const response = await fetch('/api/v2/authenticate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });
            
            const result = await response.json();
            
            if (response.ok && result.status === 'success') {
                // Store the token
                accessToken = result.access_token;
                
                // Set cookie for session management
                document.cookie = `access_token=${result.access_token}; path=/; secure; samesite=strict`;
                
                loginStatus.textContent = `Login successful! Confidence: ${result.confidence.toFixed(1)}%. Redirecting...`;
                
                // Redirect to dashboard
                setTimeout(() => {
                    window.location.href = result.redirect_url || '/dashboard';
                }, 1000);
            } else {
                loginStatus.textContent = 'Login failed: ' + (result.detail || result.message || 'Unknown error');
            }
        } catch (error) {
            loginStatus.textContent = 'Login error: ' + error.message;
        }
    };
}

function startBehavioralMonitoring() {
    monitorInterval = setInterval(sendBehavioralData, 5000);
}

async function sendBehavioralData() {
    // Collect current session data
    let typingSpeed = 0;
    if (typingTimestamps.length > 1) {
        let intervals = typingTimestamps.slice(1).map((t, i) => t - typingTimestamps[i]);
        typingSpeed = intervals.reduce((a, b) => a + b, 0) / intervals.length;
    }
    let tapSpeed = 0;
    if (tapTimestamps.length > 1) {
        let intervals = tapTimestamps.slice(1).map((t, i) => t - tapTimestamps[i]);
        tapSpeed = intervals.reduce((a, b) => a + b, 0) / intervals.length;
    }
    const behavior = {
        typing_speed: typingSpeed,
        tap_speed: tapSpeed,
        nav_pattern: navPattern,
        swipe_pattern: swipePattern,
        GPS: gpsData
    };
    // Send to /log_behavior
    await fetch('/api/v2/log_behavior', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + accessToken
        },
        body: JSON.stringify(behavior)
    });
    // Analyze
    const res = await fetch('/api/v2/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + accessToken
        },
        body: JSON.stringify(behavior)
    });
    const data = await res.json();
    if (dashboardStatus) {
        dashboardStatus.textContent = `Behavioral confidence: ${(data.confidence*100).toFixed(1)}%`;
    }
    if (data.confidence < 0.3) {
        // Lock session
        await fetch('/api/v2/lock_session', {
            method: 'POST',
            headers: {
                'Authorization': 'Bearer ' + accessToken
            }
        });
        dashboardStatus.textContent = 'Session locked due to anomaly!';
        clearInterval(monitorInterval);
    } else if (data.confidence < 0.6) {
        dashboardStatus.textContent += ' (Soft alert: unusual behavior)';
    }
}

if (logoutBtn) {
    logoutBtn.onclick = function() {
        clearInterval(monitorInterval);
        dashboard.style.display = 'none';
        loginForm.style.display = 'block';
        accessToken = null;
        loginStatus.textContent = 'Logged out.';
    };
} 