"""
In-House Machine Learning Module for Candle-Light

This module provides a comprehensive candlestick pattern detection
system that works without external AI services.

Components:
- types.py: Centralized type definitions (enums, dataclasses)
- constants.py: All ML thresholds and magic numbers
- patterns.py: Database of 30+ candlestick pattern definitions
- image_processor.py: OpenCV-based chart image processing
- pattern_detector.py: Rule-based pattern detection engine

Usage:
    from app.ml import analyze_chart_image, PatternDetector
    
    result = analyze_chart_image(image_bytes)
    print(result.market_bias)
    print(result.patterns)
"""

# Types and enums (canonical source)
from .types import (
    TrendDirection,
    CandleColor,
    MarketBias,
    PatternLocation,
    ChartMetrics,
)

# Pattern definitions
from .patterns import (
    CandlestickPattern,
    PatternType,
    PatternCategory,
    ALL_PATTERNS,
    get_pattern_by_name,
    get_patterns_by_bias,
    get_patterns_by_category,
)

# Image processing (uses types internally)
from .image_processor import (
    ImageProcessor,
    ChartAnalysis,
    Candlestick,
)

# Pattern detection
from .pattern_detector import (
    PatternDetector,
    DetectedPattern,
    AnalysisResult,
    SupportResistance,
    analyze_chart_image,
    analyze_chart_image_async,
    get_pattern_detector,
    get_pattern_summary,
)


__all__ = [
    # Types and enums
    "TrendDirection",
    "CandleColor",
    "MarketBias",
    "PatternLocation",
    "ChartMetrics",
    # Pattern definitions
    "CandlestickPattern",
    "PatternType", 
    "PatternCategory",
    "ALL_PATTERNS",
    "get_pattern_by_name",
    "get_patterns_by_bias",
    "get_patterns_by_category",
    # Image processing
    "ImageProcessor",
    "ChartAnalysis",
    "Candlestick",
    # Pattern detection
    "PatternDetector",
    "DetectedPattern",
    "AnalysisResult",
    "SupportResistance",
    "analyze_chart_image",
    "analyze_chart_image_async",
    "get_pattern_detector",
    "get_pattern_summary",
]


