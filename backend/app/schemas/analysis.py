"""
Analysis Schemas

Pydantic models for chart analysis request/response validation.
"""

from datetime import datetime
from typing import Optional, List, Literal
from pydantic import BaseModel, Field


class PatternResult(BaseModel):
    """Schema for a detected pattern."""
    name: str = Field(..., description="Pattern name (e.g., 'Double Bottom', 'Hammer')")
    type: Literal["bullish", "bearish", "neutral"] = Field(
        ..., 
        description="Pattern bias type"
    )
    confidence: int = Field(
        ..., 
        ge=0, 
        le=100, 
        description="Confidence score 0-100"
    )
    description: Optional[str] = Field(
        None, 
        description="Brief description of the pattern"
    )


class EntryTiming(BaseModel):
    """Schema for market entry timing prediction."""
    signal: Literal["wait", "prepare", "ready", "now"] = Field(
        "wait",
        description="Entry signal: wait, prepare, ready, or now"
    )
    timing_description: Optional[str] = Field(
        None,
        description="Clear explanation of when to enter"
    )
    conditions: List[str] = Field(
        default_factory=list,
        description="Conditions that need to be met before entry"
    )
    entry_price_zone: Optional[str] = Field(
        None,
        description="Price range or level for entry"
    )
    stop_loss: Optional[str] = Field(
        None,
        description="Where to place stop loss"
    )
    take_profit: Optional[str] = Field(
        None,
        description="Target price levels"
    )
    risk_reward: Optional[str] = Field(
        None,
        description="Estimated risk:reward ratio"
    )
    timeframe: Optional[str] = Field(
        None,
        description="Expected timeframe for the trade"
    )


class AnalysisCreate(BaseModel):
    """Schema for creating an analysis (internal use)."""
    image_filename: str
    image_path: str
    original_filename: Optional[str] = None
    image_size: Optional[int] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None


class AnalysisResult(BaseModel):
    """Schema for AI analysis result."""
    patterns: List[PatternResult] = Field(default_factory=list)
    market_bias: Literal["bullish", "bearish", "neutral"] = "neutral"
    confidence: int = Field(0, ge=0, le=100)
    reasoning: Optional[str] = None
    entry_timing: Optional[EntryTiming] = None
    ai_provider: Optional[str] = None
    ai_model: Optional[str] = None
    processing_time_ms: Optional[int] = None


class AnalysisResponse(BaseModel):
    """Schema for analysis response."""
    id: str
    image_url: Optional[str] = None
    patterns: List[PatternResult] = Field(default_factory=list)
    market_bias: str = "neutral"
    confidence: int = 0
    reasoning: Optional[str] = None
    status: str = "completed"
    created_at: datetime
    
    # Metadata
    ai_provider: Optional[str] = None
    processing_time_ms: Optional[int] = None
    
    model_config = {"from_attributes": True}


class AnalysisListResponse(BaseModel):
    """Schema for paginated analysis list."""
    items: List[AnalysisResponse]
    total: int
    page: int
    page_size: int
    pages: int


class AnalysisUploadResponse(BaseModel):
    """Schema for upload and analysis response."""
    message: str = "Analysis completed successfully"
    analysis: AnalysisResponse


# ============================================================
# Export Schemas
# ============================================================

class ExportFormat(BaseModel):
    """Schema for export format options."""
    format: Literal["json", "csv", "txt"] = "json"


class ExportResponse(BaseModel):
    """Schema for export metadata (actual file is returned as download)."""
    filename: str
    content_type: str
    size_bytes: int


class HistoryExportRequest(BaseModel):
    """Schema for history export request."""
    format: Literal["json", "csv"] = "csv"
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None

