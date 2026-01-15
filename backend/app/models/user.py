"""
User Model

Represents a user in the Candle-Light application.
Supports both email/password and OAuth authentication.
"""

from datetime import datetime
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
        DateTime,
        default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )
    last_login: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
        nullable=True
    )
    
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
