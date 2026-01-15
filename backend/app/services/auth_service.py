"""
Authentication Service

Handles user authentication, registration, and token management.
"""

from datetime import datetime
from typing import Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.user import User
from app.schemas.auth import Token
from app.core.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_token_expiry_seconds,
)
from app.core.exceptions import AuthenticationError, ValidationError


class AuthService:
    """Service for handling authentication operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def register(
        self,
        email: str,
        password: str,
        full_name: Optional[str] = None
    ) -> User:
        """
        Register a new user with email and password.
        
        Args:
            email: User's email address
            password: Plain text password
            full_name: Optional full name
            
        Returns:
            The created User object
            
        Raises:
            ValidationError: If email already exists
        """
        # Check if email exists
        existing = await self.get_user_by_email(email)
        if existing:
            raise ValidationError(
                message="Email already registered",
                details={"email": email}
            )
        
        # Create user
        user = User(
            email=email.lower(),
            hashed_password=get_password_hash(password),
            full_name=full_name,
            is_active=True,
            is_verified=False,  # Could implement email verification later
        )
        
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        
        return user
    
    async def login(self, email: str, password: str) -> Token:
        """
        Authenticate user and return tokens.
        
        Args:
            email: User's email address
            password: Plain text password
            
        Returns:
            Token object with access and refresh tokens
            
        Raises:
            AuthenticationError: If credentials are invalid
        """
        user = await self.authenticate(email, password)
        
        if not user:
            raise AuthenticationError(
                message="Invalid email or password"
            )
        
        if not user.is_active:
            raise AuthenticationError(
                message="User account is deactivated"
            )
        
        # Update last login
        user.last_login = datetime.utcnow()
        await self.db.commit()
        
        # Generate tokens
        access_token = create_access_token(user.id, user.email)
        refresh_token = create_refresh_token(user.id, user.email)
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=get_token_expiry_seconds()
        )
    
    async def refresh_tokens(self, refresh_token: str) -> Token:
        """
        Refresh access token using refresh token.
        
        Args:
            refresh_token: The refresh token
            
        Returns:
            New Token object with fresh tokens
            
        Raises:
            AuthenticationError: If refresh token is invalid
        """
        payload = decode_token(refresh_token)
        
        if not payload:
            raise AuthenticationError(
                message="Invalid refresh token"
            )
        
        if payload.get("type") != "refresh":
            raise AuthenticationError(
                message="Invalid token type"
            )
        
        user_id = payload.get("sub")
        email = payload.get("email")
        
        # Verify user still exists and is active
        user = await self.get_user_by_id(user_id)
        if not user or not user.is_active:
            raise AuthenticationError(
                message="User not found or inactive"
            )
        
        # Generate new tokens
        new_access_token = create_access_token(user.id, user.email)
        new_refresh_token = create_refresh_token(user.id, user.email)
        
        return Token(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            token_type="bearer",
            expires_in=get_token_expiry_seconds()
        )
    
    async def authenticate(
        self,
        email: str,
        password: str
    ) -> Optional[User]:
        """
        Verify email and password, return user if valid.
        
        Args:
            email: User's email
            password: Plain text password
            
        Returns:
            User object if valid, None otherwise
        """
        user = await self.get_user_by_email(email)
        
        if not user:
            return None
        
        if not user.hashed_password:
            return None  # OAuth-only user
        
        if not verify_password(password, user.hashed_password):
            return None
        
        return user
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email address."""
        result = await self.db.execute(
            select(User).where(User.email == email.lower())
        )
        return result.scalar_one_or_none()
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()
    
    async def create_oauth_user(
        self,
        email: str,
        full_name: Optional[str],
        avatar_url: Optional[str],
        provider: str,
        oauth_id: str
    ) -> Tuple[User, bool]:
        """
        Create or get user from OAuth login.
        
        Returns:
            Tuple of (User, created: bool)
        """
        # Check if user exists
        user = await self.get_user_by_email(email)
        
        if user:
            # Update OAuth info if not set
            if not user.oauth_provider:
                user.oauth_provider = provider
                user.oauth_id = oauth_id
                if not user.avatar_url and avatar_url:
                    user.avatar_url = avatar_url
                await self.db.commit()
            return user, False
        
        # Create new OAuth user
        user = User(
            email=email.lower(),
            full_name=full_name,
            avatar_url=avatar_url,
            oauth_provider=provider,
            oauth_id=oauth_id,
            is_active=True,
            is_verified=True,  # OAuth users are verified
        )
        
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        
        return user, True
