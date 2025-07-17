// static/main.js (Corrected for /dashboard)
document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('role-selection-form');

    if (form) {
        form.addEventListener('submit', (event) => {
            event.preventDefault(); 
            const roleSelect = document.getElementById('role-select');
            const selectedRole = roleSelect.value;
    
            // --- THIS IS THE FIX ---
            if (selectedRole === 'passenger') {
                // It now correctly points to the /dashboard endpoint
                window.location.href = '/dashboard'; 
            } else if (selectedRole === 'driver') {
                window.location.href = '/driver';
            }
        });
    }
});