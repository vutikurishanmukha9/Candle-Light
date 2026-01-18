"""
User Router

Handles user profile operations, password change, preferences, and account deletion.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete
from datetime import datetime, timedelta

from app.database import get_db
from app.schemas.user import UserResponse, UserUpdate, PasswordChange, PreferencesUpdate, UserPreferences
from app.core.dependencies import get_current_user
from app.models.user import User
from app.models.analysis import Analysis


router = APIRouter(prefix="/users", tags=["Users"])


@router.get("/profile", response_model=UserResponse)
async def get_profile(
    current_user: User = Depends(get_current_user)
):
    """
    Get current user's profile.
    """
    return current_user


@router.put("/profile", response_model=UserResponse)
async def update_profile(
    updates: UserUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update current user's profile.
    
    - **full_name**: User's display name
    - **avatar_url**: URL to avatar image
    """
    if updates.full_name is not None:
        current_user.full_name = updates.full_name
    
    if updates.avatar_url is not None:
        current_user.avatar_url = updates.avatar_url
    
    await db.commit()
    await db.refresh(current_user)
    
    return current_user


@router.put("/password")
async def change_password(
    password_data: PasswordChange,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Change current user's password.
    
    Requires current password for verification.
    """
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    # Verify current password
    if not pwd_context.verify(password_data.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    # Hash and set new password
    current_user.hashed_password = pwd_context.hash(password_data.new_password)
    current_user.updated_at = datetime.utcnow()
    
    await db.commit()
    
    return {"message": "Password changed successfully"}


@router.get("/preferences", response_model=UserPreferences)
async def get_preferences(
    current_user: User = Depends(get_current_user)
):
    """
    Get current user's preferences.
    """
    if current_user.preferences:
        return UserPreferences(**current_user.preferences)
    return UserPreferences()


@router.put("/preferences", response_model=UserPreferences)
async def update_preferences(
    prefs: PreferencesUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update current user's preferences.
    """
    current_prefs = current_user.preferences or {}
    
    # Update only provided fields
    updates = prefs.model_dump(exclude_unset=True)
    current_prefs.update(updates)
    
    current_user.preferences = current_prefs
    current_user.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(current_user)
    
    return UserPreferences(**current_user.preferences)


@router.delete("/account")
async def delete_account(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Permanently delete the current user's account and all associated data.
    
    WARNING: This action is irreversible.
    """
    user_id = current_user.id
    
    # Delete all user's analyses
    await db.execute(
        delete(Analysis).where(Analysis.user_id == user_id)
    )
    
    # Delete user
    await db.delete(current_user)
    await db.commit()
    
    return {"message": "Account deleted successfully"}


@router.delete("/history")
async def clear_history(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete all analysis history for the current user.
    """
    result = await db.execute(
        delete(Analysis).where(Analysis.user_id == current_user.id)
    )
    await db.commit()
    
    return {"message": f"Deleted {result.rowcount} analyses"}


@router.get("/stats")
async def get_user_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get user's analysis statistics.
    """
    # Get total analyses
    result = await db.execute(
        select(func.count(Analysis.id)).where(
            Analysis.user_id == current_user.id
        )
    )
    total_analyses = result.scalar() or 0
    
    # Get bias distribution
    result = await db.execute(
        select(Analysis.market_bias, func.count(Analysis.id))
        .where(Analysis.user_id == current_user.id)
        .group_by(Analysis.market_bias)
    )
    bias_distribution = {row[0]: row[1] for row in result}
    
    # Get average confidence
    result = await db.execute(
        select(func.avg(Analysis.confidence)).where(
            Analysis.user_id == current_user.id
        )
    )
    avg_confidence = result.scalar() or 0
    
    # Get this week count
    one_week_ago = datetime.utcnow() - timedelta(days=7)
    result = await db.execute(
        select(func.count(Analysis.id)).where(
            Analysis.user_id == current_user.id,
            Analysis.created_at >= one_week_ago
        )
    )
    this_week_count = result.scalar() or 0
    
    return {
        "total_analyses": total_analyses,
        "bias_distribution": bias_distribution,
        "avg_confidence": round(avg_confidence) if avg_confidence else 0,
        "this_week_count": this_week_count,
        "member_since": current_user.created_at.isoformat(),
    }
