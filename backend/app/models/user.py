"""
User Model

Represents a user in the Candle-Light application.
Supports both email/password and OAuth authentication.
"""

from datetime import datetime, timezone
from typing import Optional, List, TYPE_CHECKING
from sqlalchemy import String, Boolean, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
import uuid

from app.database import Base

if TYPE_CHECKING:
    from app.models.analysis import Analysis


class User(Base):
    """User account model."""
    
    __tablename__ = "users"
    
    # Primary key
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )
    
    # Authentication
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
        nullable=False
    )
    hashed_password: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True  # Null for OAuth-only users
    )
    
    # Profile
    full_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True
    )
    avatar_url: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )
    
    # OAuth
    oauth_provider: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True  # "google", "github", etc.
    )
    oauth_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True
    )
    
    # Status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True
    )
    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False
    )
    is_superuser: Mapped[bool] = mapped_column(
        Boolean,
        default=False
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc)
    )
    last_login: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
        nullable=True
    )
    
    # User preferences (JSON)
    preferences: Mapped[Optional[dict]] = mapped_column(
        Text,  # Stored as JSON string
        nullable=True,
        default=None
    )
    
    @property
    def user_preferences(self) -> dict:
        """Get user preferences with defaults."""
        import json
        defaults = {
            "theme": "dark",
            "ai_provider": "inhouse",
            "notifications_enabled": True,
            "email_notifications": False,
            "default_timeframe": "1h",
            "show_confidence_scores": True,
            "auto_analyze": False,
        }
        if self.preferences:
            try:
                stored = json.loads(self.preferences) if isinstance(self.preferences, str) else self.preferences
                return {**defaults, **stored}
            except (json.JSONDecodeError, TypeError):
                return defaults
        return defaults
    
    def set_preferences(self, prefs: dict) -> None:
        """Update user preferences."""
        import json
        current = self.user_preferences
        current.update(prefs)
        self.preferences = json.dumps(current)
    
    # Relationships
    analyses: Mapped[List["Analysis"]] = relationship(
        "Analysis",
        back_populates="user",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email})>"
    
    @property
    def display_name(self) -> str:
        """Get user's display name."""
        return self.full_name or self.email.split("@")[0]
