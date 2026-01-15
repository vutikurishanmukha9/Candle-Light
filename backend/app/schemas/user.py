"""
User Schemas

Pydantic models for user-related request/response validation.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class UserBase(BaseModel):
    """Base user schema with common fields."""
    email: EmailStr
    full_name: Optional[str] = None


class UserCreate(UserBase):
    """Schema for creating a new user."""
    password: str = Field(..., min_length=8, max_length=100)


class UserUpdate(BaseModel):
    """Schema for updating user profile."""
    full_name: Optional[str] = Field(None, max_length=255)
    avatar_url: Optional[str] = None


class UserResponse(UserBase):
    """Schema for user response (public data)."""
    id: str
    avatar_url: Optional[str] = None
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime
    
    model_config = {"from_attributes": True}


class UserInDB(UserResponse):
    """Schema for user with private data (internal use)."""
    hashed_password: Optional[str] = None
    oauth_provider: Optional[str] = None
    oauth_id: Optional[str] = None
    is_superuser: bool = False
    updated_at: datetime
    last_login: Optional[datetime] = None
    
    model_config = {"from_attributes": True}
