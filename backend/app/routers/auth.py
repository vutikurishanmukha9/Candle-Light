"""
Authentication Router

Handles user registration, login, and token management.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.auth import (
    LoginRequest,
    RegisterRequest,
    RefreshTokenRequest,
    Token,
)
from app.schemas.user import UserResponse
from app.services.auth_service import AuthService
from app.core.dependencies import get_current_user
from app.core.exceptions import AuthenticationError, ValidationError
from app.models.user import User


router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    request: RegisterRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Register a new user account.
    
    - **email**: Valid email address
    - **password**: Password (min 8 characters)
    - **full_name**: Optional full name
    """
    try:
        auth_service = AuthService(db)
        user = await auth_service.register(
            email=request.email,
            password=request.password,
            full_name=request.full_name
        )
        return user
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message
        )


@router.post("/login", response_model=Token)
async def login(
    request: LoginRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Login with email and password.
    
    Returns access and refresh tokens.
    """
    try:
        auth_service = AuthService(db)
        tokens = await auth_service.login(
            email=request.email,
            password=request.password
        )
        return tokens
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=e.message,
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post("/refresh", response_model=Token)
async def refresh_token(
    request: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Refresh access token using refresh token.
    """
    try:
        auth_service = AuthService(db)
        tokens = await auth_service.refresh_tokens(request.refresh_token)
        return tokens
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=e.message,
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """
    Get current authenticated user's information.
    
    Requires valid access token.
    """
    return current_user


@router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_user)
):
    """
    Logout current user.
    
    Note: JWT tokens are stateless, so this endpoint is mainly
    for client-side token cleanup. For full logout, client should
    discard the tokens.
    """
    return {
        "message": "Logged out successfully",
        "detail": "Please discard your access and refresh tokens"
    }
