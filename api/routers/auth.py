"""
Glass Expert AI — Auth Router
Endpoints: /register, /login, /me, /users (admin only)
"""
import os
import psycopg2
from datetime import timedelta
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field

from api.auth import (
    get_password_hash, authenticate_user, create_access_token,
    require_auth, require_admin, get_db_connection,
    UserInDB, ACCESS_TOKEN_EXPIRE_MINUTES
)

router = APIRouter()


# ── Request / Response schemas ─────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=6, description="Password (min 6 chars)")
    full_name: Optional[str] = Field(None, description="Full name")
    role: Optional[str] = Field("engineer", description="Role: admin, engineer, viewer")
    plant_location: Optional[str] = Field(None, description="Plant/factory location")
    language_pref: Optional[str] = Field("en", description="Language preference: en or fa")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: dict


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: Optional[str]
    role: str
    plant_location: Optional[str]
    language_pref: str
    is_active: bool


# ── Register ───────────────────────────────────────────────────────────────────
@router.post("/register", response_model=UserResponse, status_code=201)
async def register(req: RegisterRequest):
    """Register a new user account."""
    valid_roles = {"admin", "engineer", "viewer"}
    if req.role not in valid_roles:
        raise HTTPException(status_code=400, detail=f"Role must be one of: {valid_roles}")

    hashed_pw = get_password_hash(req.password)

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO users (email, hashed_password, full_name, role, plant_location, language_pref)
               VALUES (%s, %s, %s, %s, %s, %s)
               RETURNING id, email, full_name, role, plant_location, language_pref, is_active""",
            (req.email, hashed_pw, req.full_name, req.role, req.plant_location, req.language_pref)
        )
        row = cur.fetchone()
        conn.commit()
        conn.close()
        return UserResponse(
            id=str(row[0]), email=row[1], full_name=row[2],
            role=row[3], plant_location=row[4], language_pref=row[5], is_active=row[6]
        )
    except psycopg2.errors.UniqueViolation:
        raise HTTPException(status_code=409, detail="Email already registered")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


# ── Login ──────────────────────────────────────────────────────────────────────
@router.post("/login", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login and receive a JWT access token."""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user["is_active"]:
        raise HTTPException(status_code=403, detail="Account is disabled")

    token = create_access_token(
        data={"sub": user["id"], "email": user["email"], "role": user["role"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user={
            "id": user["id"],
            "email": user["email"],
            "full_name": user["full_name"],
            "role": user["role"],
            "plant_location": user["plant_location"],
            "language_pref": user["language_pref"]
        }
    )


# ── Get current user profile ───────────────────────────────────────────────────
@router.get("/me", response_model=UserResponse)
async def get_me(current_user: UserInDB = Depends(require_auth)):
    """Get the currently authenticated user's profile."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, email, full_name, role, plant_location, language_pref, is_active FROM users WHERE id = %s",
            (current_user.id,)
        )
        row = cur.fetchone()
        conn.close()
        if not row:
            raise HTTPException(status_code=404, detail="User not found")
        return UserResponse(
            id=str(row[0]), email=row[1], full_name=row[2],
            role=row[3], plant_location=row[4], language_pref=row[5], is_active=row[6]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── List all users (admin only) ────────────────────────────────────────────────
@router.get("/users", response_model=List[UserResponse])
async def list_users(admin: UserInDB = Depends(require_admin)):
    """List all users — admin only."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, email, full_name, role, plant_location, language_pref, is_active FROM users ORDER BY created_at DESC")
        rows = cur.fetchall()
        conn.close()
        return [
            UserResponse(
                id=str(r[0]), email=r[1], full_name=r[2],
                role=r[3], plant_location=r[4], language_pref=r[5], is_active=r[6]
            ) for r in rows
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Deactivate user (admin only) ───────────────────────────────────────────────
@router.delete("/users/{user_id}")
async def deactivate_user(user_id: str, admin: UserInDB = Depends(require_admin)):
    """Deactivate a user account — admin only."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("UPDATE users SET is_active = false WHERE id = %s RETURNING id", (user_id,))
        row = cur.fetchone()
        conn.commit()
        conn.close()
        if not row:
            raise HTTPException(status_code=404, detail="User not found")
        return {"message": f"User {user_id} deactivated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))