"""
OAuth Service

Handles OAuth authentication with external providers (Google, GitHub, etc.)
using the Authlib library.
"""

import logging
from typing import Optional, Tuple
from dataclasses import dataclass

from authlib.integrations.starlette_client import OAuth
from starlette.requests import Request

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class OAuthUserInfo:
    """User information from OAuth provider"""
    email: str
    name: Optional[str] = None
    picture: Optional[str] = None
    provider: str = "google"
    provider_id: str = ""


class OAuthService:
    """
    OAuth authentication service using Authlib.
    
    Supports:
    - Google OAuth 2.0
    - GitHub OAuth (future)
    """
    
    def __init__(self):
        self.oauth = OAuth()
        self._configured = False
        self._configure_providers()
    
    def _configure_providers(self) -> None:
        """Configure OAuth providers based on settings."""
        
        # Configure Google OAuth
        if settings.google_oauth_enabled:
            self.oauth.register(
                name='google',
                client_id=settings.google_client_id,
                client_secret=settings.google_client_secret,
                server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
                client_kwargs={
                    'scope': 'openid email profile'
                }
            )
            self._configured = True
            logger.info("Google OAuth configured successfully")
        else:
            logger.warning("Google OAuth not configured (missing credentials)")
    
    @property
    def is_configured(self) -> bool:
        """Check if any OAuth provider is configured."""
        return self._configured
    
    @property
    def google_enabled(self) -> bool:
        """Check if Google OAuth is enabled."""
        return settings.google_oauth_enabled
    
    async def get_google_redirect_url(self, request: Request, redirect_uri: str) -> str:
        """
        Get the Google OAuth authorization URL.
        
        Args:
            request: Starlette request object
            redirect_uri: Callback URL after authentication
            
        Returns:
            Authorization URL to redirect user to
        """
        if not self.google_enabled:
            raise ValueError("Google OAuth is not configured")
        
        google = self.oauth.create_client('google')
        redirect = await google.create_authorization_url(redirect_uri)
        
        # Store state in session for CSRF protection
        request.session['oauth_state'] = redirect.get('state', '')
        
        return redirect['url']
    
    async def handle_google_callback(
        self, 
        request: Request,
        redirect_uri: str
    ) -> OAuthUserInfo:
        """
        Handle the Google OAuth callback.
        
        Args:
            request: Starlette request with OAuth callback params
            redirect_uri: The callback URL used in initial request
            
        Returns:
            OAuthUserInfo with user details from Google
            
        Raises:
            ValueError: If OAuth fails or user info cannot be retrieved
        """
        if not self.google_enabled:
            raise ValueError("Google OAuth is not configured")
        
        google = self.oauth.create_client('google')
        
        try:
            # Exchange code for token
            token = await google.authorize_access_token(request)
            
            # Get user info from ID token or userinfo endpoint
            user_info = token.get('userinfo')
            
            if not user_info:
                # Fallback to userinfo endpoint
                resp = await google.get('https://www.googleapis.com/oauth2/v3/userinfo')
                user_info = resp.json()
            
            email = user_info.get('email')
            if not email:
                raise ValueError("Email not provided by Google")
            
            return OAuthUserInfo(
                email=email,
                name=user_info.get('name'),
                picture=user_info.get('picture'),
                provider='google',
                provider_id=user_info.get('sub', '')
            )
            
        except Exception as e:
            logger.error(f"Google OAuth callback failed: {e}")
            raise ValueError(f"OAuth authentication failed: {str(e)}")
    
    def get_available_providers(self) -> list:
        """Get list of available OAuth providers."""
        providers = []
        
        if self.google_enabled:
            providers.append({
                "name": "google",
                "display_name": "Google",
                "enabled": True
            })
        
        return providers


# Singleton instance
oauth_service = OAuthService()
