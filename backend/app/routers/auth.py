"""
Authentication Router

Handles user registration, login, and token management.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
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


# ============================================================
# OAuth Endpoints
# ============================================================

@router.get("/providers")
async def get_oauth_providers():
    """
    Get list of available OAuth providers.
    
    Returns enabled OAuth providers for the frontend to display.
    """
    from app.services.oauth_service import oauth_service
    
    return {
        "providers": oauth_service.get_available_providers()
    }


@router.get("/google")
async def google_oauth_login(
    request: Request,
    redirect_url: str = None
):
    """
    Initiate Google OAuth login.
    
    Redirects user to Google's authorization page.
    
    - **redirect_url**: Optional URL to redirect after successful login
    """
    from fastapi.responses import RedirectResponse
    from app.services.oauth_service import oauth_service
    from app.config import settings
    
    if not oauth_service.google_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google OAuth is not configured"
        )
    
    # Build callback URL
    callback_url = f"{settings.app_url}/auth/google/callback"
    
    # Store redirect URL in session for after callback
    if redirect_url:
        request.session['oauth_redirect'] = redirect_url
    
    # Get Google authorization URL
    google = oauth_service.oauth.create_client('google')
    redirect = await google.authorize_redirect(request, callback_url)
    
    return redirect


@router.get("/google/callback")
async def google_oauth_callback(
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Handle Google OAuth callback.
    
    Exchanges authorization code for tokens and creates/updates user.
    """
    from fastapi.responses import RedirectResponse
    from app.services.oauth_service import oauth_service, OAuthUserInfo
    from app.config import settings
    from datetime import datetime, timezone
    
    if not oauth_service.google_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google OAuth is not configured"
        )
    
    try:
        # Get Google client and exchange code for token
        google = oauth_service.oauth.create_client('google')
        token = await google.authorize_access_token(request)
        
        # Get user info from token
        user_info = token.get('userinfo')
        if not user_info:
            user_info = await google.userinfo()
        
        email = user_info.get('email')
        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email not provided by Google"
            )
        
        # Check if user exists
        from sqlalchemy import select
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
        
        if user:
            # Update existing user with OAuth info
            user.oauth_provider = 'google'
            user.oauth_id = user_info.get('sub', '')
            user.last_login = datetime.now(timezone.utc)
            if not user.full_name and user_info.get('name'):
                user.full_name = user_info.get('name')
            if not user.avatar_url and user_info.get('picture'):
                user.avatar_url = user_info.get('picture')
            user.is_verified = True  # Google verified the email
        else:
            # Create new user
            user = User(
                email=email,
                full_name=user_info.get('name'),
                avatar_url=user_info.get('picture'),
                oauth_provider='google',
                oauth_id=user_info.get('sub', ''),
                is_verified=True,
                is_active=True
            )
            db.add(user)
        
        await db.commit()
        await db.refresh(user)
        
        # Generate tokens
        auth_service = AuthService(db)
        tokens = auth_service._create_tokens(user)
        
        # Get redirect URL from session
        redirect_url = request.session.pop('oauth_redirect', None)
        
        # Redirect to frontend with tokens
        frontend_url = redirect_url or settings.frontend_url or "http://localhost:5173"
        callback_params = f"?access_token={tokens['access_token']}&refresh_token={tokens['refresh_token']}"
        
        return RedirectResponse(url=f"{frontend_url}/auth/callback{callback_params}")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"OAuth authentication failed: {str(e)}"
        )

