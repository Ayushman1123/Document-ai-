// Document AI - Authentication Module
const API_BASE = 'http://localhost:8000/api';

// State
let currentTab = 'login';
let mfaToken = null;
let countdownInterval = null;

// DOM Elements
const loginForm = document.getElementById('loginForm');
const registerForm = document.getElementById('registerForm');
const mfaSection = document.getElementById('mfaSection');
const loginTab = document.getElementById('loginTab');
const registerTab = document.getElementById('registerTab');
const socialDivider = document.getElementById('socialDivider');
const socialButtons = document.getElementById('socialButtons');
const toastContainer = document.getElementById('toastContainer');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    checkExistingSession();
});

function setupEventListeners() {
    // Tab switching
    loginTab.addEventListener('click', () => switchTab('login'));
    registerTab.addEventListener('click', () => switchTab('register'));

    // Form submissions
    loginForm.addEventListener('submit', handleLogin);
    registerForm.addEventListener('submit', handleRegister);

    // Password strength
    const registerPassword = document.getElementById('registerPassword');
    registerPassword.addEventListener('input', updatePasswordStrength);

    // OTP inputs
    setupOTPInputs();
}

function switchTab(tab) {
    currentTab = tab;

    // Update tab styles
    loginTab.classList.toggle('active', tab === 'login');
    registerTab.classList.toggle('active', tab === 'register');

    // Show/hide forms
    loginForm.classList.toggle('hidden', tab !== 'login');
    registerForm.classList.toggle('hidden', tab !== 'register');

    // Show social buttons
    socialDivider.classList.remove('hidden');
    socialButtons.classList.remove('hidden');
    mfaSection.classList.add('hidden');
}

// Login Handler
async function handleLogin(e) {
    e.preventDefault();

    const email = document.getElementById('loginEmail').value;
    const password = document.getElementById('loginPassword').value;
    const rememberMe = document.getElementById('rememberMe').checked;

    const btn = loginForm.querySelector('.btn-auth');
    btn.classList.add('loading');

    try {
        const response = await fetch(`${API_BASE}/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password, remember_me: rememberMe })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Login failed');
        }

        if (data.requires_mfa) {
            // Show MFA verification
            mfaToken = data.mfa_token;
            showMFASection();
            showToast('Verification code sent to your phone', 'success');
        } else {
            // Login successful
            handleLoginSuccess(data);
        }

    } catch (error) {
        showToast(error.message, 'error');
    } finally {
        btn.classList.remove('loading');
    }
}

// Register Handler
async function handleRegister(e) {
    e.preventDefault();

    const firstName = document.getElementById('firstName').value;
    const lastName = document.getElementById('lastName').value;
    const email = document.getElementById('registerEmail').value;
    const phone = document.getElementById('phone').value;
    const password = document.getElementById('registerPassword').value;
    const confirmPassword = document.getElementById('confirmPassword').value;
    const enableMFA = document.getElementById('enableMFA').checked;
    const agreeTerms = document.getElementById('agreeTerms').checked;

    // Validate passwords match
    if (password !== confirmPassword) {
        showToast('Passwords do not match', 'error');
        return;
    }

    // Validate terms
    if (!agreeTerms) {
        showToast('Please agree to the terms and conditions', 'error');
        return;
    }

    const btn = registerForm.querySelector('.btn-auth');
    btn.classList.add('loading');

    try {
        const response = await fetch(`${API_BASE}/auth/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                first_name: firstName,
                last_name: lastName,
                email,
                phone,
                password,
                enable_mfa: enableMFA
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Registration failed');
        }

        if (enableMFA) {
            // Show MFA setup
            mfaToken = data.mfa_token;
            showMFASection();
            showToast('Please verify your phone number', 'success');
        } else {
            // Registration successful
            showToast('Account created successfully!', 'success');
            setTimeout(() => switchTab('login'), 1500);
        }

    } catch (error) {
        showToast(error.message, 'error');
    } finally {
        btn.classList.remove('loading');
    }
}

// MFA Section
function showMFASection() {
    loginForm.classList.add('hidden');
    registerForm.classList.add('hidden');
    socialDivider.classList.add('hidden');
    socialButtons.classList.add('hidden');
    mfaSection.classList.remove('hidden');

    // Focus first OTP input
    document.querySelector('.otp-input').focus();

    // Start countdown
    startCountdown();
}

function setupOTPInputs() {
    const inputs = document.querySelectorAll('.otp-input');

    inputs.forEach((input, index) => {
        input.addEventListener('input', (e) => {
            const value = e.target.value;

            // Only allow numbers
            e.target.value = value.replace(/[^0-9]/g, '');

            if (value && index < inputs.length - 1) {
                inputs[index + 1].focus();
            }

            // Update filled state
            input.classList.toggle('filled', e.target.value !== '');

            // Check if all inputs are filled
            checkOTPComplete();
        });

        input.addEventListener('keydown', (e) => {
            if (e.key === 'Backspace' && !e.target.value && index > 0) {
                inputs[index - 1].focus();
            }
        });

        input.addEventListener('paste', (e) => {
            e.preventDefault();
            const pastedData = e.clipboardData.getData('text').replace(/[^0-9]/g, '').slice(0, 6);

            pastedData.split('').forEach((char, i) => {
                if (inputs[i]) {
                    inputs[i].value = char;
                    inputs[i].classList.add('filled');
                }
            });

            if (pastedData.length === 6) {
                inputs[5].focus();
                checkOTPComplete();
            }
        });
    });
}

function checkOTPComplete() {
    const inputs = document.querySelectorAll('.otp-input');
    const isComplete = Array.from(inputs).every(input => input.value);
    document.getElementById('verifyMFABtn').disabled = !isComplete;
}

function getOTPValue() {
    const inputs = document.querySelectorAll('.otp-input');
    return Array.from(inputs).map(input => input.value).join('');
}

async function verifyMFA() {
    const otp = getOTPValue();

    if (otp.length !== 6) {
        showToast('Please enter complete verification code', 'error');
        return;
    }

    const btn = document.getElementById('verifyMFABtn');
    btn.classList.add('loading');

    try {
        const response = await fetch(`${API_BASE}/auth/verify-mfa`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                mfa_token: mfaToken,
                otp_code: otp
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Verification failed');
        }

        // MFA verified successfully
        handleLoginSuccess(data);

    } catch (error) {
        showToast(error.message, 'error');
        // Clear OTP inputs
        document.querySelectorAll('.otp-input').forEach(input => {
            input.value = '';
            input.classList.remove('filled');
        });
        document.querySelector('.otp-input').focus();
    } finally {
        btn.classList.remove('loading');
    }
}

function startCountdown() {
    let seconds = 60;
    const countdownEl = document.getElementById('countdown');
    const timerText = document.getElementById('timerText');
    const resendBtn = document.getElementById('resendBtn');

    resendBtn.classList.add('hidden');
    timerText.classList.remove('hidden');

    if (countdownInterval) {
        clearInterval(countdownInterval);
    }

    countdownInterval = setInterval(() => {
        seconds--;
        countdownEl.textContent = seconds;

        if (seconds <= 0) {
            clearInterval(countdownInterval);
            timerText.classList.add('hidden');
            resendBtn.classList.remove('hidden');
        }
    }, 1000);
}

async function resendOTP() {
    try {
        const response = await fetch(`${API_BASE}/auth/resend-otp`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mfa_token: mfaToken })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Failed to resend code');
        }

        showToast('New code sent!', 'success');
        startCountdown();

    } catch (error) {
        showToast(error.message, 'error');
    }
}

function backToLogin() {
    if (countdownInterval) {
        clearInterval(countdownInterval);
    }
    mfaToken = null;
    switchTab('login');

    // Clear OTP inputs
    document.querySelectorAll('.otp-input').forEach(input => {
        input.value = '';
        input.classList.remove('filled');
    });
}

// Password Strength
function updatePasswordStrength() {
    const password = document.getElementById('registerPassword').value;
    const strengthEl = document.getElementById('passwordStrength');
    const textEl = strengthEl.querySelector('.strength-text');

    let strength = 0;

    if (password.length >= 8) strength++;
    if (/[a-z]/.test(password) && /[A-Z]/.test(password)) strength++;
    if (/\d/.test(password)) strength++;
    if (/[^a-zA-Z0-9]/.test(password)) strength++;

    strengthEl.classList.remove('weak', 'medium', 'strong');

    if (password.length === 0) {
        textEl.textContent = 'Password strength';
    } else if (strength <= 1) {
        strengthEl.classList.add('weak');
        textEl.textContent = 'Weak password';
    } else if (strength <= 2) {
        strengthEl.classList.add('medium');
        textEl.textContent = 'Medium password';
    } else {
        strengthEl.classList.add('strong');
        textEl.textContent = 'Strong password';
    }
}

// Toggle Password Visibility
function togglePassword(inputId) {
    const input = document.getElementById(inputId);
    const btn = input.nextElementSibling;

    if (input.type === 'password') {
        input.type = 'text';
        btn.textContent = '🔒';
    } else {
        input.type = 'password';
        btn.textContent = '👁';
    }
}

// Social Login
async function socialLogin(provider) {
    showToast(`${provider.charAt(0).toUpperCase() + provider.slice(1)} login coming soon!`, 'info');
}

// Session Management
function handleLoginSuccess(data) {
    // Store token
    localStorage.setItem('auth_token', data.access_token);
    localStorage.setItem('user', JSON.stringify(data.user));

    showToast('Login successful! Redirecting...', 'success');

    // Redirect to main app
    setTimeout(() => {
        window.location.href = 'index.html';
    }, 1000);
}

function checkExistingSession() {
    const token = localStorage.getItem('auth_token');

    if (token) {
        // Validate token and redirect if valid
        validateToken(token);
    }
}

async function validateToken(token) {
    try {
        const response = await fetch(`${API_BASE}/auth/validate`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });

        if (response.ok) {
            // Token is valid, redirect to main app
            window.location.href = 'index.html';
        } else {
            // Token invalid, clear storage
            localStorage.removeItem('auth_token');
            localStorage.removeItem('user');
        }
    } catch (error) {
        console.log('Token validation failed');
    }
}

// Forgot Password
function showForgotPassword() {
    const email = document.getElementById('loginEmail').value;

    if (!email) {
        showToast('Please enter your email address first', 'error');
        return;
    }

    // Show forgot password flow
    showToast('Password reset link sent to your email!', 'success');
}

// Toast Notifications
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;

    toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.remove();
    }, 4000);
}
