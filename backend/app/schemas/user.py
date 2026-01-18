"""
User Schemas

Pydantic models for user-related request/response validation.
"""

from datetime import datetime
from typing import Optional, Dict, Any
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


class UserPreferences(BaseModel):
    """Schema for user preferences."""
    theme: str = "dark"
    ai_provider: str = "inhouse"
    notifications_enabled: bool = True
    email_notifications: bool = False
    default_timeframe: str = "1h"
    show_confidence_scores: bool = True
    auto_analyze: bool = False


class UserResponse(UserBase):
    """Schema for user response (public data)."""
    id: str
    avatar_url: Optional[str] = None
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime
    oauth_provider: Optional[str] = None
    preferences: Optional[UserPreferences] = None
    
    model_config = {"from_attributes": True}


class UserWithPreferences(UserResponse):
    """Schema for user with full preferences."""
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    
    model_config = {"from_attributes": True}


class PasswordChange(BaseModel):
    """Schema for changing password."""
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8, max_length=100)


class PreferencesUpdate(BaseModel):
    """Schema for updating user preferences."""
    theme: Optional[str] = None
    ai_provider: Optional[str] = None
    notifications_enabled: Optional[bool] = None
    email_notifications: Optional[bool] = None
    analysis_notifications: Optional[bool] = None
    auto_save: Optional[bool] = None
    confidence_threshold: Optional[int] = None
    compact_view: Optional[bool] = None


class UserInDB(UserResponse):
    """Schema for user with private data (internal use)."""
    hashed_password: Optional[str] = None
    oauth_id: Optional[str] = None
    is_superuser: bool = False
    updated_at: datetime
    last_login: Optional[datetime] = None
    
    model_config = {"from_attributes": True}


# ============================================================
# OAuth Schemas
# ============================================================

class OAuthProvider(BaseModel):
    """Schema for OAuth provider info."""
    name: str
    display_name: str
    enabled: bool


class OAuthProvidersResponse(BaseModel):
    """Schema for list of OAuth providers."""
    providers: list[OAuthProvider]


class OAuthCallbackResponse(BaseModel):
    """Schema for OAuth callback success."""
    user: UserResponse
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

