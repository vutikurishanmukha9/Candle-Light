"""
ML Types Module

Centralized type definitions for the ML pattern detection system.
This module contains all shared dataclasses and enums used across
image_processor.py, pattern_detector.py, and patterns.py.

Import from this module rather than individual files to ensure consistency.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class TrendDirection(Enum):
    """Trend direction enumeration"""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"


class CandleColor(Enum):
    """Candle color enumeration"""
    GREEN = "green"
    RED = "red"
    MIXED = "mixed"
    NEUTRAL = "neutral"


class MarketBias(Enum):
    """Market bias enumeration"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class PatternLocation(Enum):
    """Where a pattern was found in the chart"""
    RECENT = "recent"      # Last 5 candles
    MIDDLE = "middle"      # Middle portion
    HISTORICAL = "historical"  # Beginning of chart


# ============================================================================
# Core Candlestick Dataclass
# ============================================================================

@dataclass
class Candlestick:
    """
    Represents a single candlestick with OHLC data and derived metrics.
    
    This is the canonical Candlestick class used throughout the ML system.
    """
    index: int
    x_position: int
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    is_bullish: bool
    
    # Derived properties (calculated in __post_init__)
    body_height: float = 0.0
    upper_shadow: float = 0.0
    lower_shadow: float = 0.0
    total_range: float = 0.0
    
    # Optional metadata
    volume: Optional[float] = None
    timestamp: Optional[str] = None
    confidence: float = 1.0
    detected_patterns: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate derived values after initialization."""
        if self.total_range == 0:
            self.total_range = self.high_price - self.low_price
        if self.body_height == 0:
            self.body_height = abs(self.close_price - self.open_price)
        if self.upper_shadow == 0:
            self.upper_shadow = self.high_price - max(self.open_price, self.close_price)
        if self.lower_shadow == 0:
            self.lower_shadow = min(self.open_price, self.close_price) - self.low_price
    
    @property
    def body_ratio(self) -> float:
        """Ratio of body to total range"""
        if self.total_range == 0:
            return 0
        return self.body_height / self.total_range
    
    @property
    def upper_shadow_ratio(self) -> float:
        """Ratio of upper shadow to total range"""
        if self.total_range == 0:
            return 0
        return self.upper_shadow / self.total_range
    
    @property
    def lower_shadow_ratio(self) -> float:
        """Ratio of lower shadow to total range"""
        if self.total_range == 0:
            return 0
        return self.lower_shadow / self.total_range
    
    def is_doji(self, threshold: float = 0.1) -> bool:
        """Check if this candle is a doji (very small body)"""
        return self.body_ratio < threshold
    
    def has_long_lower_shadow(self, threshold: float = 0.6) -> bool:
        """Check if lower shadow is significant"""
        return self.lower_shadow_ratio > threshold
    
    def has_long_upper_shadow(self, threshold: float = 0.6) -> bool:
        """Check if upper shadow is significant"""
        return self.upper_shadow_ratio > threshold
    
    def is_hammer_shape(self) -> bool:
        """Check if candle has hammer/pin bar shape"""
        return (
            self.lower_shadow_ratio > 0.6 and
            self.upper_shadow_ratio < 0.1 and
            self.body_ratio < 0.3
        )
    
    def is_shooting_star_shape(self) -> bool:
        """Check if candle has shooting star shape"""
        return (
            self.upper_shadow_ratio > 0.6 and
            self.lower_shadow_ratio < 0.1 and
            self.body_ratio < 0.3
        )
    
    def is_marubozu(self) -> bool:
        """Check if candle is a marubozu (no shadows)"""
        return (
            self.upper_shadow_ratio < 0.05 and
            self.lower_shadow_ratio < 0.05 and
            self.body_ratio > 0.9
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'index': self.index,
            'x_position': self.x_position,
            'open': self.open_price,
            'high': self.high_price,
            'low': self.low_price,
            'close': self.close_price,
            'is_bullish': self.is_bullish,
            'body_ratio': round(self.body_ratio, 4),
            'upper_shadow_ratio': round(self.upper_shadow_ratio, 4),
            'lower_shadow_ratio': round(self.lower_shadow_ratio, 4),
            'volume': self.volume,
            'confidence': self.confidence,
        }


# ============================================================================
# Support/Resistance
# ============================================================================

@dataclass
class SupportResistance:
    """Support or resistance level"""
    level: float
    strength: float  # 0.0 to 1.0
    touches: int
    level_type: str  # 'support' or 'resistance'


# ============================================================================
# Chart Analysis Result
# ============================================================================

@dataclass
class ChartMetrics:
    """Metrics calculated from chart analysis"""
    volatility: float = 0.0
    momentum: float = 0.0
    support_level: float = 0.0
    resistance_level: float = 0.0
    average_body_size: float = 0.0
    bullish_percentage: float = 50.0
    pattern_density: float = 0.0
    quality_score: float = 0.0


@dataclass
class ChartAnalysis:
    """Results of chart image analysis"""
    candlesticks: List[Candlestick]
    trend_direction: TrendDirection
    trend_strength: float  # 0.0 to 1.0
    dominant_color: CandleColor
    chart_quality: float  # 0.0 to 1.0
    image_dimensions: Tuple[int, int]
    metrics: ChartMetrics = field(default_factory=ChartMetrics)
    
    @property
    def candle_count(self) -> int:
        return len(self.candlesticks)
    
    @property
    def bullish_count(self) -> int:
        return sum(1 for c in self.candlesticks if c.is_bullish)
    
    @property
    def bearish_count(self) -> int:
        return len(self.candlesticks) - self.bullish_count


# ============================================================================
# Pattern Detection Results
# ============================================================================

@dataclass
class DetectedPattern:
    """A pattern detected in the chart"""
    name: str
    pattern_type: str  # 'bullish', 'bearish', 'neutral'
    confidence: float  # 0.0 to 1.0
    location: PatternLocation
    candle_indices: List[int]
    reasoning: str
    volume_confirmed: bool = False
    near_support_resistance: bool = False
    trend_aligned: bool = False
    quality_score: float = 0.0


@dataclass
class AnalysisResult:
    """Complete analysis result"""
    patterns: List[DetectedPattern]
    market_bias: MarketBias
    overall_confidence: float
    trend_analysis: str
    reasoning: str
    support_levels: List[SupportResistance] = field(default_factory=list)
    resistance_levels: List[SupportResistance] = field(default_factory=list)
    key_price_levels: List[float] = field(default_factory=list)
    volatility: float = 0.0
    chart_analysis: Optional[ChartAnalysis] = None
