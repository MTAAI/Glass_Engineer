"""
Glass Expert AI — Authentication Module
JWT-based auth with password hashing. Users table in PostgreSQL.
"""
import os
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from loguru import logger
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field

# ── Configuration ─────────────────────────────────────────────────────────────

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
if not SECRET_KEY:
    import secrets
    SECRET_KEY = secrets.token_hex(32)
    logger.warning("JWT_SECRET_KEY not set! Using random key — tokens will invalidate on restart. Set JWT_SECRET_KEY in .env for production.")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "480"))  # 8 hours

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)

router = APIRouter(prefix="/auth")

# ── Schemas ───────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = None
    plant_location: Optional[str] = None
    language_pref: str = Field("en", pattern="^(en|fa)$")

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str
    role: str
    full_name: Optional[str] = None

class UserResponse(BaseModel):
    id: str
    email: str
    full_name: Optional[str]
    role: str
    plant_location: Optional[str]
    language_pref: str
    is_active: bool

class UserInToken(BaseModel):
    """Decoded user info from JWT."""
    user_id: str
    email: str
    role: str

# ── Helpers ───────────────────────────────────────────────────────────────────

def _hash_password(password: str) -> str:
    return pwd_context.hash(password)

def _verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def _create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode["exp"] = expire
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def _get_db_connection():
    """Get a pooled DB connection."""
    from api.database import get_db_conn
    return get_db_conn()


def _return_db_connection(conn):
    """Return connection to pool instead of closing."""
    from api.database import return_db_conn
    return_db_conn(conn)

# ── Dependencies ──────────────────────────────────────────────────────────────

async def get_current_user(token: Optional[str] = Depends(oauth2_scheme)) -> UserInToken:
    """Extract and validate user from JWT token. Returns anonymous user if no token."""
    if token is None:
        # Allow anonymous access with limited identity
        return UserInToken(
            user_id="00000000-0000-0000-0000-000000000001",
            email="anonymous@glass-expert.ai",
            role="anonymous",
        )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        email: str = payload.get("email", "")
        role: str = payload.get("role", "engineer")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return UserInToken(user_id=user_id, email=email, role=role)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


async def require_auth(token: Optional[str] = Depends(oauth2_scheme)) -> UserInToken:
    """Strict auth — rejects anonymous users."""
    if token is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return await get_current_user(token)


async def require_admin(user: UserInToken = Depends(require_auth)) -> UserInToken:
    """Requires admin role."""
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse)
async def register(req: RegisterRequest):
    """Register a new user account."""
    conn = _get_db_connection()
    try:
        cur = conn.cursor()
        # Check if email already exists
        cur.execute("SELECT id FROM users WHERE email = %s", (req.email,))
        if cur.fetchone():
            raise HTTPException(status_code=409, detail="Email already registered")

        hashed = _hash_password(req.password)
        cur.execute(
            """INSERT INTO users (email, hashed_password, full_name, plant_location, language_pref)
               VALUES (%s, %s, %s, %s, %s) RETURNING id, role""",
            (req.email, hashed, req.full_name, req.plant_location, req.language_pref),
        )
        row = cur.fetchone()
        conn.commit()

        user_id = str(row[0])
        role = row[1]

        token = _create_access_token({"sub": user_id, "email": req.email, "role": role})
        logger.info(f"User registered: {req.email} (id={user_id})")

        return TokenResponse(
            access_token=token,
            user_id=user_id,
            email=req.email,
            role=role,
            full_name=req.full_name,
        )
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")
    finally:
        _return_db_connection(conn)


@router.post("/login", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login with email + password. Returns JWT token.
    Uses OAuth2PasswordRequestForm for Swagger UI compatibility (username = email).
    """
    conn = _get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, email, hashed_password, full_name, role, is_active FROM users WHERE email = %s",
            (form_data.username,),  # OAuth2 form uses 'username' field
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        user_id, email, hashed_pw, full_name, role, is_active = row

        if not is_active:
            raise HTTPException(status_code=403, detail="Account is deactivated")

        if not _verify_password(form_data.password, hashed_pw):
            raise HTTPException(status_code=401, detail="Invalid email or password")

        token = _create_access_token({"sub": str(user_id), "email": email, "role": role})
        logger.info(f"User logged in: {email}")

        return TokenResponse(
            access_token=token,
            user_id=str(user_id),
            email=email,
            role=role,
            full_name=full_name,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")
    finally:
        _return_db_connection(conn)


@router.get("/me", response_model=UserResponse)
async def get_me(user: UserInToken = Depends(require_auth)):
    """Get current user profile."""
    conn = _get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """SELECT id, email, full_name, role, plant_location, language_pref, is_active
               FROM users WHERE id = %s""",
            (user.user_id,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="User not found")

        return UserResponse(
            id=str(row[0]),
            email=row[1],
            full_name=row[2],
            role=row[3],
            plant_location=row[4],
            language_pref=row[5],
            is_active=row[6],
        )
    finally:
        _return_db_connection(conn)
