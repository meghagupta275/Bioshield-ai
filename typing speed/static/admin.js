// --- Admin Login ---
const adminLoginForm = document.getElementById('adminLoginForm');
const adminLoginStatus = document.getElementById('adminLoginStatus');
if (adminLoginForm) {
    adminLoginForm.onsubmit = async function(e) {
        e.preventDefault();
        const formData = new FormData(adminLoginForm);
        const payload = new URLSearchParams();
        payload.append('username', formData.get('username'));
        payload.append('password', formData.get('password'));
        payload.append('grant_type', 'password');
        adminLoginStatus.textContent = 'Logging in...';
        const res = await fetch('/api/v2/admin/login', {
            method: 'POST',
            headers: {'Content-Type': 'application/x-www-form-urlencoded'},
            body: payload
        });
        const data = await res.json();
        if (data.access_token) {
            localStorage.setItem('admin_jwt', data.access_token);
            window.location.href = '/admin_dashboard';
        } else {
            adminLoginStatus.textContent = data.detail || 'Login failed.';
        }
    };
}

// --- Admin Dashboard ---
if (window.location.pathname.endsWith('/admin_dashboard')) {
    const jwt = localStorage.getItem('admin_jwt');
    if (!jwt) {
        window.location.href = '/admin_login';
    }
    fetch('/api/v2/admin/logs', {
        headers: {
            'Authorization': 'Bearer ' + jwt
        }
    })
    .then(res => res.json())
    .then(data => {
        // Populate behavior logs
        const behaviorTable = document.getElementById('behaviorLogsTable').querySelector('tbody');
        behaviorTable.innerHTML = '';
        data.behavior_logs.forEach(log => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${log.user_id}</td>
                <td>${log.timestamp}</td>
                <td>${log.typing_speed}</td>
                <td>${log.tap_speed}</td>
                <td>${log.swipe_pattern}</td>
                <td>${log.nav_pattern}</td>
                <td>${log.gps}</td>
                <td>${log.anomaly_score}</td>
            `;
            behaviorTable.appendChild(row);
        });
        // Populate security logs
        const securityTable = document.getElementById('securityLogsTable').querySelector('tbody');
        securityTable.innerHTML = '';
        data.security_logs.forEach(log => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${log.user_id}</td>
                <td>${log.event_type}</td>
                <td>${log.ip_address}</td>
                <td>${log.user_agent}</td>
                <td>${log.details}</td>
                <td>${log.timestamp}</td>
            `;
            securityTable.appendChild(row);
        });
    });
    // Logout link
    const sidebar = document.querySelector('.sidebar');
    if (sidebar) {
        sidebar.addEventListener('click', function(e) {
            if (e.target.tagName === 'A' && e.target.textContent === 'Logout') {
                localStorage.removeItem('admin_jwt');
                window.location.href = '/admin_login';
            }
        });
    }
} 