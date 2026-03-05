"""
Glass Expert AI — JWT Authentication Module
Handles user registration, login, and JWT token validation.
Roles: admin, engineer, viewer
"""
import os
from datetime import datetime, timedelta
from typing import Optional

import psycopg2
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# ── Config ─────────────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "glass-expert-ai-secret-key-change-in-production-2024")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))  # 24 hours

# ── Password hashing ───────────────────────────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


# ── Pydantic models ────────────────────────────────────────────────────────────
class TokenData(BaseModel):
    user_id: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None


class UserInDB(BaseModel):
    id: str
    email: str
    full_name: Optional[str] = None
    role: str
    plant_location: Optional[str] = None
    language_pref: str = "en"
    is_active: bool = True


# ── Helpers ────────────────────────────────────────────────────────────────────
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_db_connection():
    db_url = os.getenv("DATABASE_URL", "postgresql://glassai:glassai_secure_2024@localhost:5432/glass_expert_ai")
    return psycopg2.connect(db_url)


def get_user_by_email(email: str) -> Optional[dict]:
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, email, hashed_password, full_name, role, plant_location, language_pref, is_active FROM users WHERE email = %s",
            (email,)
        )
        row = cur.fetchone()
        conn.close()
        if row:
            return {
                "id": str(row[0]), "email": row[1], "hashed_password": row[2],
                "full_name": row[3], "role": row[4], "plant_location": row[5],
                "language_pref": row[6], "is_active": row[7]
            }
        return None
    except Exception:
        return None


def authenticate_user(email: str, password: str) -> Optional[dict]:
    user = get_user_by_email(email)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return user


# ── Dependency: get current user from JWT token ────────────────────────────────
async def get_current_user(token: Optional[str] = Depends(oauth2_scheme)) -> Optional[UserInDB]:
    """Returns current user if token is valid, else None (for optional auth)."""
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        email: str = payload.get("email")
        role: str = payload.get("role")
        if not user_id:
            return None
        return UserInDB(id=user_id, email=email or "", role=role or "viewer")
    except JWTError:
        return None


async def require_auth(token: str = Depends(oauth2_scheme)) -> UserInDB:
    """Strict auth — raises 401 if not authenticated."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated. Please login at /api/v1/auth/login",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not token:
        raise credentials_exception
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        email: str = payload.get("email")
        role: str = payload.get("role")
        if not user_id:
            raise credentials_exception
        return UserInDB(id=user_id, email=email or "", role=role or "viewer")
    except JWTError:
        raise credentials_exception


async def require_admin(current_user: UserInDB = Depends(require_auth)) -> UserInDB:
    """Requires admin role."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user