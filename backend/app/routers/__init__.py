"""API Routers Package"""

from app.routers.auth import router as auth_router
from app.routers.analysis import router as analysis_router
from app.routers.user import router as user_router
from app.routers.health import router as health_router
from app.routers.settings import router as settings_router

__all__ = [
    "auth_router",
    "analysis_router",
    "user_router",
    "health_router",
    "settings_router",
]

