"""
Settings Router

Handles user preferences and settings management.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.database import get_db
from app.core.dependencies import get_current_user
from app.models.user import User


router = APIRouter(prefix="/settings", tags=["Settings"])


# ============================================================================
# Schemas
# ============================================================================

class UserPreferences(BaseModel):
    """User preferences schema"""
    theme: Optional[str] = None  # "dark" or "light"
    ai_provider: Optional[str] = None  # "inhouse", "openai", "gemini"
    notifications_enabled: Optional[bool] = None
    email_notifications: Optional[bool] = None
    default_timeframe: Optional[str] = None  # "1m", "5m", "1h", "4h", "1d"
    show_confidence_scores: Optional[bool] = None
    auto_analyze: Optional[bool] = None

    class Config:
        json_schema_extra = {
            "example": {
                "theme": "dark",
                "ai_provider": "inhouse",
                "notifications_enabled": True,
                "default_timeframe": "1h"
            }
        }


class PreferencesResponse(BaseModel):
    """Response containing all user preferences"""
    theme: str
    ai_provider: str
    notifications_enabled: bool
    email_notifications: bool
    default_timeframe: str
    show_confidence_scores: bool
    auto_analyze: bool


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/preferences", response_model=PreferencesResponse)
async def get_preferences(
    current_user: User = Depends(get_current_user)
):
    """
    Get current user's preferences.
    
    Returns all preferences with defaults applied for any unset values.
    """
    prefs = current_user.user_preferences
    return PreferencesResponse(**prefs)


@router.patch("/preferences", response_model=PreferencesResponse)
async def update_preferences(
    preferences: UserPreferences,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update user preferences.
    
    Only updates the fields that are provided (partial update).
    """
    # Get non-None values
    updates = {k: v for k, v in preferences.model_dump().items() if v is not None}
    
    if not updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No preferences to update"
        )
    
    # Validate theme
    if "theme" in updates and updates["theme"] not in ("dark", "light"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Theme must be 'dark' or 'light'"
        )
    
    # Validate ai_provider
    if "ai_provider" in updates and updates["ai_provider"] not in ("inhouse", "openai", "gemini", "demo"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid AI provider"
        )
    
    # Validate timeframe
    valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
    if "default_timeframe" in updates and updates["default_timeframe"] not in valid_timeframes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid timeframe. Must be one of: {', '.join(valid_timeframes)}"
        )
    
    # Update preferences
    current_user.set_preferences(updates)
    await db.commit()
    await db.refresh(current_user)
    
    return PreferencesResponse(**current_user.user_preferences)


@router.delete("/preferences")
async def reset_preferences(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Reset all preferences to defaults.
    """
    current_user.preferences = None
    await db.commit()
    
    return {
        "message": "Preferences reset to defaults",
        "preferences": current_user.user_preferences
    }
