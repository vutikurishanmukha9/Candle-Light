"""
Authentication Schemas

Pydantic models for authentication-related request/response validation.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class LoginRequest(BaseModel):
    """Schema for login request."""
    email: EmailStr
    password: str = Field(..., min_length=1)


class RegisterRequest(BaseModel):
    """Schema for user registration."""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    full_name: Optional[str] = Field(None, max_length=255)


class RefreshTokenRequest(BaseModel):
    """Schema for token refresh request."""
    refresh_token: str


class Token(BaseModel):
    """Schema for token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class TokenPayload(BaseModel):
    """Schema for JWT token payload."""
    sub: str  # user_id
    email: str
    exp: datetime
    iat: datetime
    type: str = "access"  # "access" or "refresh"


class GoogleAuthCallback(BaseModel):
    """Schema for Google OAuth callback."""
    code: str
    state: Optional[str] = None
