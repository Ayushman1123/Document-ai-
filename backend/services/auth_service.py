"""
Authentication Service with Multi-Factor Authentication
Handles user registration, login, and MFA verification
"""
import os
import re
import json
import uuid
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict
from pathlib import Path
import random

logger = logging.getLogger(__name__)

# In production, use a proper database. This is for demo purposes.
USERS_FILE = Path(__file__).parent.parent.parent / "data" / "users.json"
SESSIONS_FILE = Path(__file__).parent.parent.parent / "data" / "sessions.json"

class AuthService:
    """
    Authentication service with MFA support
    
    Features:
    - User registration with email/password
    - Secure password hashing with salt
    - Multi-factor authentication via OTP
    - Session management
    - Rate limiting
    """
    
    def __init__(self):
        self.users: Dict = {}
        self.sessions: Dict = {}
        self.mfa_tokens: Dict = {}  # Temporary MFA token storage
        self.otp_codes: Dict = {}   # OTP codes storage
        self.rate_limits: Dict = {} # Rate limiting
        
        self._load_data()
    
    def _load_data(self):
        """Load user and session data from files"""
        USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        if USERS_FILE.exists():
            try:
                with open(USERS_FILE, 'r') as f:
                    self.users = json.load(f)
            except:
                self.users = {}
        
        if SESSIONS_FILE.exists():
            try:
                with open(SESSIONS_FILE, 'r') as f:
                    self.sessions = json.load(f)
            except:
                self.sessions = {}
    
    def _save_users(self):
        """Save users to file"""
        with open(USERS_FILE, 'w') as f:
            json.dump(self.users, f, indent=2)
    
    def _save_sessions(self):
        """Save sessions to file"""
        with open(SESSIONS_FILE, 'w') as f:
            json.dump(self.sessions, f, indent=2)
    
    def _hash_password(self, password: str, salt: str = None) -> tuple:
        """Hash password with salt using SHA-256"""
        if salt is None:
            salt = secrets.token_hex(16)
        
        salted = f"{salt}{password}"
        hashed = hashlib.sha256(salted.encode()).hexdigest()
        
        return hashed, salt
    
    def _verify_password(self, password: str, stored_hash: str, salt: str) -> bool:
        """Verify password against stored hash"""
        hashed, _ = self._hash_password(password, salt)
        return secrets.compare_digest(hashed, stored_hash)
    
    def _generate_otp(self) -> str:
        """Generate 6-digit OTP code"""
        return ''.join([str(random.randint(0, 9)) for _ in range(6)])
    
    def _validate_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def _validate_phone(self, phone: str) -> bool:
        """Validate phone number format"""
        # Remove spaces, dashes, and +
        cleaned = re.sub(r'[\s\-\+]', '', phone)
        return len(cleaned) >= 10 and cleaned.isdigit()
    
    def _validate_password(self, password: str) -> tuple:
        """Validate password strength"""
        if len(password) < 8:
            return False, "Password must be at least 8 characters"
        if not re.search(r'[a-z]', password):
            return False, "Password must contain lowercase letters"
        if not re.search(r'[A-Z]', password):
            return False, "Password must contain uppercase letters"
        if not re.search(r'\d', password):
            return False, "Password must contain numbers"
        return True, "Password is strong"
    
    def _check_rate_limit(self, identifier: str, action: str, max_attempts: int = 5) -> bool:
        """Check if action is rate limited"""
        key = f"{identifier}:{action}"
        current = self.rate_limits.get(key, {"count": 0, "reset_at": datetime.now().isoformat()})
        
        reset_at = datetime.fromisoformat(current["reset_at"])
        
        if datetime.now() > reset_at:
            # Reset the counter
            self.rate_limits[key] = {
                "count": 1,
                "reset_at": (datetime.now() + timedelta(minutes=15)).isoformat()
            }
            return True
        
        if current["count"] >= max_attempts:
            return False
        
        current["count"] += 1
        self.rate_limits[key] = current
        return True
    
    def register(
        self,
        email: str,
        password: str,
        first_name: str,
        last_name: str,
        phone: str,
        enable_mfa: bool = True
    ) -> Dict:
        """
        Register a new user
        
        Returns:
            Dict with status and user info or error
        """
        # Validate email
        if not self._validate_email(email):
            return {"success": False, "error": "Invalid email format"}
        
        # Check if email exists
        if email.lower() in self.users:
            return {"success": False, "error": "Email already registered"}
        
        # Validate phone
        if enable_mfa and not self._validate_phone(phone):
            return {"success": False, "error": "Invalid phone number"}
        
        # Validate password
        is_valid, message = self._validate_password(password)
        if not is_valid:
            return {"success": False, "error": message}
        
        # Hash password
        password_hash, salt = self._hash_password(password)
        
        # Create user
        user_id = str(uuid.uuid4())[:8]
        user = {
            "id": user_id,
            "email": email.lower(),
            "first_name": first_name,
            "last_name": last_name,
            "phone": phone,
            "password_hash": password_hash,
            "password_salt": salt,
            "mfa_enabled": enable_mfa,
            "created_at": datetime.now().isoformat(),
            "last_login": None,
            "is_verified": not enable_mfa  # If MFA disabled, user is verified
        }
        
        self.users[email.lower()] = user
        self._save_users()
        
        if enable_mfa:
            # Generate MFA token and OTP
            mfa_token = secrets.token_urlsafe(32)
            otp = self._generate_otp()
            
            self.mfa_tokens[mfa_token] = {
                "email": email.lower(),
                "type": "registration",
                "expires_at": (datetime.now() + timedelta(minutes=10)).isoformat()
            }
            
            self.otp_codes[mfa_token] = {
                "code": otp,
                "expires_at": (datetime.now() + timedelta(minutes=5)).isoformat(),
                "attempts": 0
            }
            
            # In production, send OTP via SMS
            logger.info(f"OTP for {email}: {otp}")  # Remove in production!
            
            return {
                "success": True,
                "requires_mfa": True,
                "mfa_token": mfa_token,
                "message": "Verification code sent to your phone"
            }
        
        return {
            "success": True,
            "message": "Registration successful",
            "user": {
                "id": user_id,
                "email": email.lower(),
                "first_name": first_name,
                "last_name": last_name
            }
        }
    
    def login(self, email: str, password: str, remember_me: bool = False) -> Dict:
        """
        Authenticate user
        
        Returns:
            Dict with access token or MFA requirement
        """
        email = email.lower()
        
        # Rate limiting
        if not self._check_rate_limit(email, "login"):
            return {
                "success": False,
                "error": "Too many login attempts. Please try again later."
            }
        
        # Check if user exists
        user = self.users.get(email)
        if not user:
            return {"success": False, "error": "Invalid email or password"}
        
        # Verify password
        if not self._verify_password(password, user["password_hash"], user["password_salt"]):
            return {"success": False, "error": "Invalid email or password"}
        
        # Check if MFA enabled
        if user.get("mfa_enabled"):
            # Generate MFA token and OTP
            mfa_token = secrets.token_urlsafe(32)
            otp = self._generate_otp()
            
            self.mfa_tokens[mfa_token] = {
                "email": email,
                "type": "login",
                "remember_me": remember_me,
                "expires_at": (datetime.now() + timedelta(minutes=10)).isoformat()
            }
            
            self.otp_codes[mfa_token] = {
                "code": otp,
                "expires_at": (datetime.now() + timedelta(minutes=5)).isoformat(),
                "attempts": 0
            }
            
            # In production, send OTP via SMS
            logger.info(f"Login OTP for {email}: {otp}")  # Remove in production!
            
            return {
                "success": True,
                "requires_mfa": True,
                "mfa_token": mfa_token,
                "message": "Verification code sent to your phone"
            }
        
        # Create session
        return self._create_session(user, remember_me)
    
    def verify_mfa(self, mfa_token: str, otp_code: str) -> Dict:
        """
        Verify MFA code
        
        Returns:
            Dict with access token on success
        """
        # Check if MFA token exists
        mfa_data = self.mfa_tokens.get(mfa_token)
        if not mfa_data:
            return {"success": False, "error": "Invalid or expired verification session"}
        
        # Check expiration
        if datetime.now() > datetime.fromisoformat(mfa_data["expires_at"]):
            del self.mfa_tokens[mfa_token]
            return {"success": False, "error": "Verification session expired"}
        
        # Get OTP data
        otp_data = self.otp_codes.get(mfa_token)
        if not otp_data:
            return {"success": False, "error": "Invalid verification code"}
        
        # Check OTP expiration
        if datetime.now() > datetime.fromisoformat(otp_data["expires_at"]):
            del self.otp_codes[mfa_token]
            return {"success": False, "error": "Verification code expired. Please request a new one."}
        
        # Check attempts
        if otp_data["attempts"] >= 3:
            del self.otp_codes[mfa_token]
            del self.mfa_tokens[mfa_token]
            return {"success": False, "error": "Too many incorrect attempts. Please start over."}
        
        # Verify OTP
        if otp_code != otp_data["code"]:
            otp_data["attempts"] += 1
            self.otp_codes[mfa_token] = otp_data
            remaining = 3 - otp_data["attempts"]
            return {"success": False, "error": f"Incorrect code. {remaining} attempts remaining."}
        
        # OTP verified!
        email = mfa_data["email"]
        user = self.users.get(email)
        
        if not user:
            return {"success": False, "error": "User not found"}
        
        # If registration, mark as verified
        if mfa_data["type"] == "registration":
            user["is_verified"] = True
            self.users[email] = user
            self._save_users()
        
        # Clean up
        del self.otp_codes[mfa_token]
        del self.mfa_tokens[mfa_token]
        
        # Create session
        remember_me = mfa_data.get("remember_me", False)
        return self._create_session(user, remember_me)
    
    def resend_otp(self, mfa_token: str) -> Dict:
        """Resend OTP code"""
        mfa_data = self.mfa_tokens.get(mfa_token)
        if not mfa_data:
            return {"success": False, "error": "Invalid session"}
        
        # Rate limiting
        email = mfa_data["email"]
        if not self._check_rate_limit(email, "resend_otp", max_attempts=3):
            return {"success": False, "error": "Too many resend attempts. Please wait."}
        
        # Generate new OTP
        otp = self._generate_otp()
        
        self.otp_codes[mfa_token] = {
            "code": otp,
            "expires_at": (datetime.now() + timedelta(minutes=5)).isoformat(),
            "attempts": 0
        }
        
        # In production, send OTP via SMS
        logger.info(f"New OTP for {email}: {otp}")  # Remove in production!
        
        return {
            "success": True,
            "message": "New verification code sent"
        }
    
    def _create_session(self, user: Dict, remember_me: bool = False) -> Dict:
        """Create a new session for user"""
        session_id = secrets.token_urlsafe(32)
        
        # Session duration: 7 days if remember_me, else 24 hours
        duration = timedelta(days=7) if remember_me else timedelta(hours=24)
        expires_at = datetime.now() + duration
        
        session = {
            "user_id": user["id"],
            "email": user["email"],
            "created_at": datetime.now().isoformat(),
            "expires_at": expires_at.isoformat()
        }
        
        self.sessions[session_id] = session
        self._save_sessions()
        
        # Update last login
        user["last_login"] = datetime.now().isoformat()
        self.users[user["email"]] = user
        self._save_users()
        
        return {
            "success": True,
            "access_token": session_id,
            "expires_at": expires_at.isoformat(),
            "user": {
                "id": user["id"],
                "email": user["email"],
                "first_name": user["first_name"],
                "last_name": user["last_name"],
                "mfa_enabled": user.get("mfa_enabled", False)
            }
        }
    
    def validate_token(self, token: str) -> Dict:
        """Validate access token"""
        session = self.sessions.get(token)
        
        if not session:
            return {"valid": False, "error": "Invalid token"}
        
        if datetime.now() > datetime.fromisoformat(session["expires_at"]):
            del self.sessions[token]
            self._save_sessions()
            return {"valid": False, "error": "Token expired"}
        
        user = self.users.get(session["email"])
        if not user:
            return {"valid": False, "error": "User not found"}
        
        return {
            "valid": True,
            "user": {
                "id": user["id"],
                "email": user["email"],
                "first_name": user["first_name"],
                "last_name": user["last_name"]
            }
        }
    
    def logout(self, token: str) -> Dict:
        """Logout user and invalidate session"""
        if token in self.sessions:
            del self.sessions[token]
            self._save_sessions()
        
        return {"success": True, "message": "Logged out successfully"}
    
    def get_user_stats(self) -> Dict:
        """Get authentication statistics"""
        return {
            "total_users": len(self.users),
            "mfa_enabled_users": sum(1 for u in self.users.values() if u.get("mfa_enabled")),
            "active_sessions": len(self.sessions)
        }


# Singleton instance
_auth_service = None

def get_auth_service() -> AuthService:
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService()
    return _auth_service
