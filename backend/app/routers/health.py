"""
Health Check Router

Simple health check endpoints for monitoring.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.database import get_db
from app.config import settings


router = APIRouter(prefix="/health", tags=["Health"])


@router.get("")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
    }


@router.get("/db")
async def database_health(db: AsyncSession = Depends(get_db)):
    """Check database connectivity."""
    try:
        await db.execute(text("SELECT 1"))
        return {
            "status": "healthy",
            "database": "connected",
            "type": "sqlite" if settings.is_sqlite else "postgresql"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e)
        }


@router.get("/ai")
async def ai_health():
    """Check AI service configuration."""
    return {
        "status": "healthy",
        "provider": settings.ai_provider,
        "configured": settings.ai_enabled,
        "inhouse_first": settings.ai_inhouse_first,
    }
