"""
Candle-Light Backend Configuration

Uses pydantic-settings for environment variable management.
Supports SQLite for local development and PostgreSQL for production.
"""

import secrets
import warnings
from functools import lru_cache
from typing import List, Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, model_validator
import json


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # ===================
    # Database
    # ===================
    database_url: str = "sqlite+aiosqlite:///./candlelight.db"
    
    @property
    def is_sqlite(self) -> bool:
        """Check if using SQLite database."""
        return "sqlite" in self.database_url.lower()
    
    @property
    def is_postgres(self) -> bool:
        """Check if using PostgreSQL database."""
        return "postgresql" in self.database_url.lower()
    
    # ===================
    # Authentication
    # ===================
    # JWT secret - MUST be set in production via environment variable
    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    @field_validator("jwt_secret_key", mode="before")
    @classmethod
    def generate_secret_if_empty(cls, v):
        """Generate a secure secret in development if not provided."""
        if not v:
            warnings.warn(
                "JWT_SECRET_KEY not set! Using auto-generated key. "
                "Set JWT_SECRET_KEY environment variable in production.",
                UserWarning
            )
            return secrets.token_urlsafe(32)
        return v
    
    # Google OAuth
    google_client_id: str = ""
    google_client_secret: str = ""
    
    @property
    def google_oauth_enabled(self) -> bool:
        """Check if Google OAuth is configured."""
        return bool(self.google_client_id and self.google_client_secret)
    
    # ===================
    # AI/ML Services
    # ===================
    openai_api_key: str = ""
    google_ai_api_key: str = ""
    ai_provider: Literal["openai", "gemini", "inhouse", "demo"] = "demo"
    
    @property
    def ai_enabled(self) -> bool:
        """Check if any AI provider is configured."""
        if self.ai_provider == "openai":
            return bool(self.openai_api_key)
        elif self.ai_provider == "gemini":
            return bool(self.google_ai_api_key)
        return True  # Demo mode is always available
    
    # ===================
    # Storage
    # ===================
    storage_type: Literal["local", "s3"] = "local"
    upload_dir: str = "./uploads"
    
    # S3 Configuration (for production cloud deployments)
    # Set STORAGE_TYPE=s3 to enable S3 storage
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    s3_bucket_name: str = ""  # e.g., "candle-light-uploads"
    s3_region: str = "us-east-1"
    s3_endpoint_url: str = ""  # For S3-compatible services (MinIO, DigitalOcean Spaces)
    
    @property
    def s3_enabled(self) -> bool:
        """Check if S3 storage is properly configured."""
        return (
            self.storage_type == "s3" and 
            bool(self.s3_bucket_name) and
            bool(self.aws_access_key_id)
        )

    
    # ===================
    # Server
    # ===================
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False  # SECURITY: Default to False for production safety
    
    # ===================
    # Rate Limiting
    # ===================
    rate_limit_per_minute: int = 60
    rate_limit_burst: int = 10
    
    # ===================
    # URLs (for OAuth redirects)
    # ===================
    app_url: str = "http://localhost:8000"  # Backend URL
    frontend_url: str = "http://localhost:5173"  # Frontend URL
    
    # ===================
    # Session (for OAuth state)
    # ===================
    session_secret_key: str = ""
    
    @field_validator("session_secret_key", mode="before")
    @classmethod
    def generate_session_secret_if_empty(cls, v):
        """Generate a secure session secret if not provided."""
        if not v:
            return secrets.token_urlsafe(32)
        return v
    
    # ===================
    # CORS
    # ===================
    cors_origins: List[str] = [
        "http://localhost:8080",
        "http://localhost:3000",
        "http://127.0.0.1:8080",
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ]
    
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from JSON string or list."""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [origin.strip() for origin in v.split(",")]
        return v
    
    # ===================
    # Application Info
    # ===================
    app_name: str = "Candle-Light API"
    app_version: str = "1.0.0"
    app_description: str = "AI-Powered Candlestick Pattern Analysis API"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses lru_cache to ensure settings are loaded only once.
    """
    return Settings()


# Convenience instance
settings = get_settings()
