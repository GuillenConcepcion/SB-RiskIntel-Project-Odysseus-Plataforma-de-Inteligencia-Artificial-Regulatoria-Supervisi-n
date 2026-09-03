"""Odysseus SupTech RBAC & OAuth2/JWT Authentication Module.

Implements banking-grade cryptographic authentication:
1. PBKDF2-HMAC-SHA256 password hashing with individual salt.
2. JSON Web Token (JWT) encoding, decoding, and expiration validation (HMAC-SHA256).
3. Role-Based Access Control (RBAC) with hierarchical supervisory roles:
   - AUDITOR_INSPECTOR: Read-only access to alerts, forecasts, XAI, and reports.
   - DATA_SCIENTIST: Access to training, tuning, caching, and drift evaluation.
   - SUPERVISORY_ADMIN: Complete administrative privileges.
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import sys
import time
from typing import Any, Dict, List, Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# Security Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "sb-riskintel-supervisory-secret-key-2026-fips-compliant")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 8  # 8-hour supervisory session

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")


# --- Cryptographic Helpers (PBKDF2 & JWT) ---

def hash_password(password: str, salt: Optional[str] = None) -> str:
    """Hash password using PBKDF2-HMAC-SHA256 with 100,000 iterations."""
    if salt is None:
        salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
    return f"{salt}${key.hex()}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plain password against stored salt$hash."""
    try:
        salt, stored_hash = hashed_password.split("$")
        computed_hash = hashlib.pbkdf2_hmac("sha256", plain_password.encode("utf-8"), salt.encode("utf-8"), 100000).hex()
        return secrets.compare_digest(stored_hash, computed_hash)
    except Exception:
        return False


def _base64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _base64url_decode(data_str: str) -> bytes:
    padding = "=" * ((4 - len(data_str) % 4) % 4)
    return base64.urlsafe_b64decode(data_str + padding)


def create_access_token(data: Dict[str, Any], expires_delta_seconds: Optional[int] = None) -> str:
    """Generate signed JWT token."""
    to_encode = data.copy()
    expire_time = time.time() + (expires_delta_seconds or (ACCESS_TOKEN_EXPIRE_MINUTES * 60))
    to_encode.update({"exp": expire_time, "iat": time.time()})

    header = {"alg": ALGORITHM, "typ": "JWT"}
    header_b64 = _base64url_encode(json.dumps(header, sort_keys=True).encode("utf-8"))
    payload_b64 = _base64url_encode(json.dumps(to_encode, sort_keys=True, default=str).encode("utf-8"))

    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    signature = hmac.new(SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256).digest()
    signature_b64 = _base64url_encode(signature)

    return f"{header_b64}.{payload_b64}.{signature_b64}"


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decode and verify JWT signature and expiration timestamp."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid JWT format")

        header_b64, payload_b64, signature_b64 = parts
        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
        expected_sig = hmac.new(SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256).digest()
        provided_sig = _base64url_decode(signature_b64)

        if not secrets.compare_digest(expected_sig, provided_sig):
            raise ValueError("Invalid signature")

        payload_bytes = _base64url_decode(payload_b64)
        payload = json.loads(payload_bytes.decode("utf-8"))

        if time.time() > payload.get("exp", 0):
            raise ValueError("Token expired")

        return payload
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not validate credentials: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


# --- User Registry & Roles ---

class User(BaseModel):
    username: str
    email: str
    full_name: str
    role: str
    disabled: bool = False


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str
    expires_in_minutes: int


# Pre-configured institutional users for SB-RiskIntel
USERS_DB = {
    "inspector": {
        "username": "inspector",
        "email": "inspector.bancario@sb.gob.do",
        "full_name": "Auditor / Inspector de Entidades Financieras",
        "hashed_password": hash_password("inspector123", salt="salt_inspector_2026"),
        "role": "AUDITOR_INSPECTOR",
        "disabled": False,
    },
    "datascientist": {
        "username": "datascientist",
        "email": "guillenconcepcion@gmail.com",
        "full_name": "Guillén Concepción (Senior Data Scientist & MLOps)",
        "hashed_password": hash_password("mlops123", salt="salt_mlops_2026"),
        "role": "DATA_SCIENTIST",
        "disabled": False,
    },
    "admin": {
        "username": "admin",
        "email": "supervision.riesgos@sb.gob.do",
        "full_name": "Supervisory Security Admin",
        "hashed_password": hash_password("admin123", salt="salt_admin_2026"),
        "role": "SUPERVISORY_ADMIN",
        "disabled": False,
    },
}


# --- FastAPI RBAC Dependencies ---

def authenticate_user(username: str, password: str) -> Optional[User]:
    """Authenticate user credentials against security directory."""
    user_dict = USERS_DB.get(username)
    if not user_dict:
        return None
    if not verify_password(password, user_dict["hashed_password"]):
        return None
    return User(**user_dict)


def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Validate bearer token and retrieve authenticated user."""
    payload = decode_access_token(token)
    username = payload.get("sub")
    if username is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")

    user_dict = USERS_DB.get(username)
    if user_dict is None or user_dict.get("disabled", False):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or disabled")

    return User(**user_dict)


def require_roles(allowed_roles: List[str]):
    """Enforce RBAC role requirement on protected endpoints."""
    def role_checker(current_user: User = Security(get_current_user)) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: User role '{current_user.role}' lacks permissions for this operation. Allowed roles: {allowed_roles}",
            )
        return current_user
    return role_checker
