"""
In-House Machine Learning Module for Candle-Light

This module provides a comprehensive candlestick pattern detection
system that works without external AI services.

Components:
- patterns.py: Database of 30+ candlestick pattern definitions
- image_processor.py: OpenCV-based chart image processing
- pattern_detector.py: Rule-based pattern detection engine

Usage:
    from app.ml import analyze_chart_image, PatternDetector
    
    result = analyze_chart_image(image_bytes)
    print(result.market_bias)
    print(result.patterns)
"""

from .patterns import (
    CandlestickPattern,
    PatternType,
    PatternCategory,
    ALL_PATTERNS,
    get_pattern_by_name,
    get_patterns_by_bias,
    get_patterns_by_category,
)

from .image_processor import (
    ImageProcessor,
    ChartAnalysis,
    Candlestick,
)

from .pattern_detector import (
    PatternDetector,
    DetectedPattern,
    AnalysisResult,
    analyze_chart_image,
    pattern_detector,
)


__all__ = [
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
    "analyze_chart_image",
    "pattern_detector",
]
