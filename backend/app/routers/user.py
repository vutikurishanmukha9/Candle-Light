"""
User Router

Handles user profile operations.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.user import UserResponse, UserUpdate
from app.core.dependencies import get_current_user
from app.models.user import User


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


@router.get("/stats")
async def get_user_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get user's analysis statistics.
    """
    from sqlalchemy import select, func
    from app.models.analysis import Analysis
    
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
    
    return {
        "total_analyses": total_analyses,
        "bias_distribution": bias_distribution,
        "member_since": current_user.created_at.isoformat(),
    }
