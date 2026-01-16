"""
Candle-Light Backend - FastAPI Application

AI-Powered Candlestick Pattern Analysis API.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os

from app.config import settings
from app.database import init_db, close_db
from app.core.exceptions import AppException
from app.core.rate_limit import RateLimitMiddleware
from app.routers import (
    auth_router,
    analysis_router,
    user_router,
    health_router,
    settings_router,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    print(f"[START] Starting {settings.app_name} v{settings.app_version}")
    print(f"[DB] Database: {'SQLite' if settings.is_sqlite else 'PostgreSQL'}")
    print(f"[AI] AI Provider: {settings.ai_provider}")
    
    # Initialize database
    await init_db()
    print("[OK] Database initialized")
    
    # Ensure upload directory exists
    os.makedirs(settings.upload_dir, exist_ok=True)
    print(f"[DIR] Upload directory: {settings.upload_dir}")
    
    yield
    
    # Shutdown
    print("[STOP] Shutting down...")
    await close_db()
    print("[OK] Database connections closed")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware
app.add_middleware(
    RateLimitMiddleware,
    rate_per_minute=settings.rate_limit_per_minute,
    burst=settings.rate_limit_burst,
)

# Add session middleware for OAuth state storage
from starlette.middleware.sessions import SessionMiddleware
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.session_secret_key,
    max_age=3600,  # 1 hour session expiry
)


# Mount static files for uploads
app.mount(
    "/uploads",
    StaticFiles(directory=settings.upload_dir),
    name="uploads"
)


# Exception handlers
@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    """Handle custom application exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.message,
            "details": exc.details,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    if settings.debug:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "details": {"message": str(exc)},
            },
        )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )


# Include routers
app.include_router(health_router)
app.include_router(auth_router)
app.include_router(analysis_router)
app.include_router(user_router)
app.include_router(settings_router)


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """API root endpoint with welcome message."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
    }


# For running with uvicorn directly
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
