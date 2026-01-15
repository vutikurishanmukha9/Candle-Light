"""
Analysis Model

Represents a chart analysis result in the Candle-Light application.
Stores the image path, detected patterns, and AI-generated insights.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from sqlalchemy import String, Integer, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
import uuid

from app.database import Base

if TYPE_CHECKING:
    from app.models.user import User


class Analysis(Base):
    """Chart analysis result model."""
    
    __tablename__ = "analyses"
    
    # Primary key
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )
    
    # Foreign key to user
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Image storage
    image_filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )
    image_path: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )
    image_url: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True  # URL for accessing the image (local or S3)
    )
    
    # Original image metadata
    original_filename: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True
    )
    image_size: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True  # Size in bytes
    )
    image_width: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True
    )
    image_height: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True
    )
    
    # Analysis results
    patterns: Mapped[List[Dict[str, Any]]] = mapped_column(
        JSON,
        default=list,
        nullable=False
    )
    """
    List of detected patterns:
    [
        {
            "name": "Double Bottom",
            "type": "bullish",
            "confidence": 85,
            "description": "..."
        },
        ...
    ]
    """
    
    market_bias: Mapped[str] = mapped_column(
        String(20),
        default="neutral"  # bullish, bearish, neutral
    )
    
    confidence: Mapped[int] = mapped_column(
        Integer,
        default=0  # 0-100
    )
    
    reasoning: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )
    
    # AI provider info
    ai_provider: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True  # "openai", "gemini", "demo"
    )
    ai_model: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True  # "gpt-4-vision-preview", "gemini-pro-vision"
    )
    
    # Processing info
    processing_time_ms: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True
    )
    
    # Status
    status: Mapped[str] = mapped_column(
        String(20),
        default="completed"  # pending, processing, completed, failed
    )
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        index=True
    )
    
    # Relationships
    user: Mapped["User"] = relationship(
        "User",
        back_populates="analyses"
    )
    
    def __repr__(self) -> str:
        return f"<Analysis(id={self.id}, bias={self.market_bias}, confidence={self.confidence})>"
    
    @property
    def pattern_count(self) -> int:
        """Get number of detected patterns."""
        return len(self.patterns) if self.patterns else 0
    
    @property
    def pattern_names(self) -> List[str]:
        """Get list of pattern names."""
        return [p.get("name", "Unknown") for p in (self.patterns or [])]
