from pydantic import BaseModel, EmailStr, Field
from uuid import UUID

from app.models.user import UserRole


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=4)


class RegisterRequest(BaseModel):
    email: EmailStr
    name: str = Field(min_length=2, max_length=200)
    password: str = Field(min_length=6, max_length=128)
    role: UserRole = UserRole.manager


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: UUID
    name: str
    email: EmailStr
    role: UserRole


class UserOut(BaseModel):
    id: UUID
    email: EmailStr
    name: str
    role: UserRole
    is_active: bool

    model_config = {"from_attributes": True}
