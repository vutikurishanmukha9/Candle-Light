"""
In-House Candlestick Pattern Detector

A comprehensive rule-based pattern detection engine that works without
external AI services. This serves as a reliable fallback and can be
used for quick local analysis.

Features:
- Single, double, and triple candle pattern detection
- Complex chart pattern recognition (Head & Shoulders, Double Top/Bottom)
- Confidence scoring based on pattern clarity
- Market bias determination
- Detailed reasoning generation
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

from .patterns import (
    CandlestickPattern, PatternType, PatternCategory,
    ALL_PATTERNS, get_pattern_by_name,
    DOJI, HAMMER, HANGING_MAN, INVERTED_HAMMER, SHOOTING_STAR,
    MARUBOZU, SPINNING_TOP,
    BULLISH_ENGULFING, BEARISH_ENGULFING, BULLISH_HARAMI, BEARISH_HARAMI,
    TWEEZER_TOP, TWEEZER_BOTTOM, PIERCING_LINE, DARK_CLOUD_COVER,
    MORNING_STAR, EVENING_STAR, THREE_WHITE_SOLDIERS, THREE_BLACK_CROWS,
    DOUBLE_TOP, DOUBLE_BOTTOM, HEAD_AND_SHOULDERS, RISING_WEDGE, FALLING_WEDGE
)
from .image_processor import ImageProcessor, ChartAnalysis, Candlestick


logger = logging.getLogger(__name__)


@dataclass
class DetectedPattern:
    """A pattern detected in the chart"""
    pattern: CandlestickPattern
    confidence: float  # 0.0 to 1.0
    location: str  # 'recent', 'middle', 'historical'
    candle_indices: List[int]
    reasoning: str


@dataclass
class AnalysisResult:
    """Complete analysis result from the in-house model"""
    patterns: List[DetectedPattern]
    market_bias: str  # 'bullish', 'bearish', 'neutral'
    overall_confidence: float
    trend_analysis: str
    reasoning: str
    raw_data: Optional[ChartAnalysis] = None


class PatternDetector:
    """
    Rule-based candlestick pattern detection engine.
    
    This detector analyzes extracted candlestick data to identify
    patterns using predefined rules and heuristics.
    """
    
    # Thresholds for pattern detection
    DOJI_BODY_RATIO = 0.1
    LONG_SHADOW_RATIO = 0.6
    ENGULFING_MIN_RATIO = 1.2
    SMALL_BODY_RATIO = 0.3
    LARGE_BODY_RATIO = 0.6
    
    def __init__(self):
        self.image_processor = ImageProcessor()
    
    def analyze_image(self, image_data: bytes) -> AnalysisResult:
        """
        Perform complete pattern analysis on a chart image.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            AnalysisResult with detected patterns and analysis
        """
        # Process the image
        chart_analysis = self.image_processor.analyze_chart(image_data)
        
        if chart_analysis is None or len(chart_analysis.candlesticks) == 0:
            return self._create_fallback_result(chart_analysis)
        
        # Detect patterns
        patterns = self._detect_all_patterns(chart_analysis)
        
        # Determine market bias
        market_bias, bias_confidence = self._determine_market_bias(
            patterns, chart_analysis
        )
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            patterns, chart_analysis
        )
        
        # Generate trend analysis
        trend_analysis = self._generate_trend_analysis(chart_analysis)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            patterns, market_bias, chart_analysis
        )
        
        return AnalysisResult(
            patterns=patterns,
            market_bias=market_bias,
            overall_confidence=overall_confidence,
            trend_analysis=trend_analysis,
            reasoning=reasoning,
            raw_data=chart_analysis
        )
    
    def _detect_all_patterns(self, analysis: ChartAnalysis) -> List[DetectedPattern]:
        """Detect all patterns in the chart"""
        patterns = []
        candles = analysis.candlesticks
        
        if len(candles) < 1:
            return patterns
        
        # Detect single candle patterns (focus on recent candles)
        for i in range(max(0, len(candles) - 10), len(candles)):
            single_patterns = self._detect_single_patterns(candles[i], i, len(candles))
            patterns.extend(single_patterns)
        
        # Detect double candle patterns
        if len(candles) >= 2:
            for i in range(max(0, len(candles) - 10), len(candles) - 1):
                double_patterns = self._detect_double_patterns(
                    candles[i], candles[i + 1], i, len(candles)
                )
                patterns.extend(double_patterns)
        
        # Detect triple candle patterns
        if len(candles) >= 3:
            for i in range(max(0, len(candles) - 10), len(candles) - 2):
                triple_patterns = self._detect_triple_patterns(
                    candles[i], candles[i + 1], candles[i + 2], i, len(candles)
                )
                patterns.extend(triple_patterns)
        
        # Detect complex patterns (need more candles)
        if len(candles) >= 5:
            complex_patterns = self._detect_complex_patterns(candles, analysis)
            patterns.extend(complex_patterns)
        
        # Sort by confidence and remove duplicates
        patterns.sort(key=lambda p: p.confidence, reverse=True)
        
        # Keep top patterns (avoid overwhelming the user)
        return patterns[:8]
    
    def _detect_single_patterns(
        self, candle: Candlestick, index: int, total: int
    ) -> List[DetectedPattern]:
        """Detect single candle patterns"""
        patterns = []
        location = self._get_location(index, total)
        
        # Doji detection
        if candle.is_doji(self.DOJI_BODY_RATIO):
            confidence = 1.0 - candle.body_ratio / self.DOJI_BODY_RATIO
            patterns.append(DetectedPattern(
                pattern=DOJI,
                confidence=min(confidence * 0.8, 0.9),  # Cap at 0.9
                location=location,
                candle_indices=[index],
                reasoning=f"Candle has very small body ({candle.body_ratio:.1%} of range)"
            ))
        
        # Hammer detection (bullish reversal after downtrend)
        if (candle.has_long_lower_shadow(self.LONG_SHADOW_RATIO) and
            candle.upper_shadow_ratio < 0.1 and
            candle.body_ratio < 0.4):
            confidence = candle.lower_shadow_ratio * 0.9
            patterns.append(DetectedPattern(
                pattern=HAMMER,
                confidence=confidence,
                location=location,
                candle_indices=[index],
                reasoning=f"Long lower shadow ({candle.lower_shadow_ratio:.1%}) with small body"
            ))
        
        # Shooting Star detection (bearish reversal after uptrend)
        if (candle.has_long_upper_shadow(self.LONG_SHADOW_RATIO) and
            candle.lower_shadow_ratio < 0.1 and
            candle.body_ratio < 0.4):
            confidence = candle.upper_shadow_ratio * 0.9
            patterns.append(DetectedPattern(
                pattern=SHOOTING_STAR,
                confidence=confidence,
                location=location,
                candle_indices=[index],
                reasoning=f"Long upper shadow ({candle.upper_shadow_ratio:.1%}) with small body"
            ))
        
        # Marubozu detection (strong momentum)
        if (candle.upper_shadow_ratio < 0.05 and 
            candle.lower_shadow_ratio < 0.05 and
            candle.body_ratio > 0.9):
            confidence = candle.body_ratio * 0.95
            patterns.append(DetectedPattern(
                pattern=MARUBOZU,
                confidence=confidence,
                location=location,
                candle_indices=[index],
                reasoning=f"Full body candle with no shadows, showing strong {'bullish' if candle.is_bullish else 'bearish'} momentum"
            ))
        
        # Spinning Top
        if (candle.body_ratio < self.SMALL_BODY_RATIO and
            candle.upper_shadow_ratio > 0.2 and
            candle.lower_shadow_ratio > 0.2):
            confidence = (1 - candle.body_ratio) * 0.6
            patterns.append(DetectedPattern(
                pattern=SPINNING_TOP,
                confidence=confidence,
                location=location,
                candle_indices=[index],
                reasoning="Small body with shadows on both sides indicating indecision"
            ))
        
        return patterns
    
    def _detect_double_patterns(
        self, c1: Candlestick, c2: Candlestick, index: int, total: int
    ) -> List[DetectedPattern]:
        """Detect two-candle patterns"""
        patterns = []
        location = self._get_location(index, total)
        
        # Bullish Engulfing
        if (not c1.is_bullish and c2.is_bullish and
            c2.body_height > c1.body_height * self.ENGULFING_MIN_RATIO):
            engulf_ratio = c2.body_height / (c1.body_height + 0.01)
            confidence = min(engulf_ratio / 2, 0.9)
            patterns.append(DetectedPattern(
                pattern=BULLISH_ENGULFING,
                confidence=confidence,
                location=location,
                candle_indices=[index, index + 1],
                reasoning=f"Bullish candle engulfs previous bearish candle by {engulf_ratio:.1f}x"
            ))
        
        # Bearish Engulfing
        if (c1.is_bullish and not c2.is_bullish and
            c2.body_height > c1.body_height * self.ENGULFING_MIN_RATIO):
            engulf_ratio = c2.body_height / (c1.body_height + 0.01)
            confidence = min(engulf_ratio / 2, 0.9)
            patterns.append(DetectedPattern(
                pattern=BEARISH_ENGULFING,
                confidence=confidence,
                location=location,
                candle_indices=[index, index + 1],
                reasoning=f"Bearish candle engulfs previous bullish candle by {engulf_ratio:.1f}x"
            ))
        
        # Bullish Harami
        if (not c1.is_bullish and c2.is_bullish and
            c1.body_height > c2.body_height * 1.5 and
            c2.body_height < c1.body_height * 0.5):
            containment = 1 - (c2.body_height / c1.body_height)
            confidence = containment * 0.7
            patterns.append(DetectedPattern(
                pattern=BULLISH_HARAMI,
                confidence=confidence,
                location=location,
                candle_indices=[index, index + 1],
                reasoning="Small bullish candle contained within large bearish candle"
            ))
        
        # Bearish Harami
        if (c1.is_bullish and not c2.is_bullish and
            c1.body_height > c2.body_height * 1.5 and
            c2.body_height < c1.body_height * 0.5):
            containment = 1 - (c2.body_height / c1.body_height)
            confidence = containment * 0.7
            patterns.append(DetectedPattern(
                pattern=BEARISH_HARAMI,
                confidence=confidence,
                location=location,
                candle_indices=[index, index + 1],
                reasoning="Small bearish candle contained within large bullish candle"
            ))
        
        # Tweezer patterns (matching highs/lows)
        high_diff = abs(c1.high_price - c2.high_price)
        low_diff = abs(c1.low_price - c2.low_price)
        
        if high_diff < 1.0 and c1.is_bullish and not c2.is_bullish:
            confidence = (1 - high_diff) * 0.7
            patterns.append(DetectedPattern(
                pattern=TWEEZER_TOP,
                confidence=confidence,
                location=location,
                candle_indices=[index, index + 1],
                reasoning=f"Matching highs within {high_diff:.1f} points"
            ))
        
        if low_diff < 1.0 and not c1.is_bullish and c2.is_bullish:
            confidence = (1 - low_diff) * 0.7
            patterns.append(DetectedPattern(
                pattern=TWEEZER_BOTTOM,
                confidence=confidence,
                location=location,
                candle_indices=[index, index + 1],
                reasoning=f"Matching lows within {low_diff:.1f} points"
            ))
        
        return patterns
    
    def _detect_triple_patterns(
        self, c1: Candlestick, c2: Candlestick, c3: Candlestick,
        index: int, total: int
    ) -> List[DetectedPattern]:
        """Detect three-candle patterns"""
        patterns = []
        location = self._get_location(index, total)
        
        # Morning Star
        if (not c1.is_bullish and 
            c2.is_doji(0.2) and
            c3.is_bullish and
            c3.body_height > c1.body_height * 0.5):
            confidence = 0.75
            patterns.append(DetectedPattern(
                pattern=MORNING_STAR,
                confidence=confidence,
                location=location,
                candle_indices=[index, index + 1, index + 2],
                reasoning="Bearish candle, small doji, then strong bullish reversal"
            ))
        
        # Evening Star
        if (c1.is_bullish and 
            c2.is_doji(0.2) and
            not c3.is_bullish and
            c3.body_height > c1.body_height * 0.5):
            confidence = 0.75
            patterns.append(DetectedPattern(
                pattern=EVENING_STAR,
                confidence=confidence,
                location=location,
                candle_indices=[index, index + 1, index + 2],
                reasoning="Bullish candle, small doji, then strong bearish reversal"
            ))
        
        # Three White Soldiers
        if (c1.is_bullish and c2.is_bullish and c3.is_bullish and
            c1.body_ratio > 0.5 and c2.body_ratio > 0.5 and c3.body_ratio > 0.5 and
            c2.close_price > c1.close_price and c3.close_price > c2.close_price):
            confidence = 0.85
            patterns.append(DetectedPattern(
                pattern=THREE_WHITE_SOLDIERS,
                confidence=confidence,
                location=location,
                candle_indices=[index, index + 1, index + 2],
                reasoning="Three consecutive bullish candles, each closing higher"
            ))
        
        # Three Black Crows
        if (not c1.is_bullish and not c2.is_bullish and not c3.is_bullish and
            c1.body_ratio > 0.5 and c2.body_ratio > 0.5 and c3.body_ratio > 0.5 and
            c2.close_price < c1.close_price and c3.close_price < c2.close_price):
            confidence = 0.85
            patterns.append(DetectedPattern(
                pattern=THREE_BLACK_CROWS,
                confidence=confidence,
                location=location,
                candle_indices=[index, index + 1, index + 2],
                reasoning="Three consecutive bearish candles, each closing lower"
            ))
        
        return patterns
    
    def _detect_complex_patterns(
        self, candles: List[Candlestick], analysis: ChartAnalysis
    ) -> List[DetectedPattern]:
        """Detect complex chart patterns like Double Top, Head & Shoulders"""
        patterns = []
        
        if len(candles) < 10:
            return patterns
        
        # Get price series
        highs = [c.high_price for c in candles]
        lows = [c.low_price for c in candles]
        closes = [c.close_price for c in candles]
        
        # Find local maxima and minima
        local_maxima = self._find_local_extrema(highs, is_max=True)
        local_minima = self._find_local_extrema(lows, is_max=False)
        
        # Double Top detection
        if len(local_maxima) >= 2:
            for i in range(len(local_maxima) - 1):
                idx1, val1 = local_maxima[i]
                idx2, val2 = local_maxima[i + 1]
                
                # Check if peaks are at similar levels
                if abs(val1 - val2) < 2.0 and idx2 - idx1 > 3:
                    # Check for valley between peaks
                    valley = min(lows[idx1:idx2+1])
                    if val1 - valley > 3.0:
                        patterns.append(DetectedPattern(
                            pattern=DOUBLE_TOP,
                            confidence=0.7,
                            location="recent" if idx2 > len(candles) - 5 else "middle",
                            candle_indices=list(range(idx1, idx2 + 1)),
                            reasoning=f"Two peaks at similar levels ({val1:.1f}, {val2:.1f}) with valley between"
                        ))
        
        # Double Bottom detection
        if len(local_minima) >= 2:
            for i in range(len(local_minima) - 1):
                idx1, val1 = local_minima[i]
                idx2, val2 = local_minima[i + 1]
                
                if abs(val1 - val2) < 2.0 and idx2 - idx1 > 3:
                    peak = max(highs[idx1:idx2+1])
                    if peak - val1 > 3.0:
                        patterns.append(DetectedPattern(
                            pattern=DOUBLE_BOTTOM,
                            confidence=0.7,
                            location="recent" if idx2 > len(candles) - 5 else "middle",
                            candle_indices=list(range(idx1, idx2 + 1)),
                            reasoning=f"Two troughs at similar levels ({val1:.1f}, {val2:.1f}) with peak between"
                        ))
        
        # Trend-based patterns (wedges)
        trend = analysis.trend_direction
        if trend == "uptrend" and len(candles) > 15:
            # Check for rising wedge (bearish in uptrend)
            recent_highs = highs[-10:]
            recent_lows = lows[-10:]
            high_slope = (recent_highs[-1] - recent_highs[0]) / len(recent_highs)
            low_slope = (recent_lows[-1] - recent_lows[0]) / len(recent_lows)
            
            if high_slope > 0 and low_slope > 0 and low_slope > high_slope:
                patterns.append(DetectedPattern(
                    pattern=RISING_WEDGE,
                    confidence=0.6,
                    location="recent",
                    candle_indices=list(range(len(candles) - 10, len(candles))),
                    reasoning="Converging upward trendlines forming rising wedge"
                ))
        
        elif trend == "downtrend" and len(candles) > 15:
            # Check for falling wedge (bullish in downtrend)
            recent_highs = highs[-10:]
            recent_lows = lows[-10:]
            high_slope = (recent_highs[-1] - recent_highs[0]) / len(recent_highs)
            low_slope = (recent_lows[-1] - recent_lows[0]) / len(recent_lows)
            
            if high_slope < 0 and low_slope < 0 and high_slope < low_slope:
                patterns.append(DetectedPattern(
                    pattern=FALLING_WEDGE,
                    confidence=0.6,
                    location="recent",
                    candle_indices=list(range(len(candles) - 10, len(candles))),
                    reasoning="Converging downward trendlines forming falling wedge"
                ))
        
        return patterns
    
    def _find_local_extrema(
        self, values: List[float], is_max: bool, window: int = 3
    ) -> List[Tuple[int, float]]:
        """Find local maxima or minima in a price series"""
        extrema = []
        
        for i in range(window, len(values) - window):
            window_vals = values[i - window:i + window + 1]
            current = values[i]
            
            if is_max:
                if current == max(window_vals):
                    extrema.append((i, current))
            else:
                if current == min(window_vals):
                    extrema.append((i, current))
        
        return extrema
    
    def _get_location(self, index: int, total: int) -> str:
        """Determine the location category of a pattern"""
        position = index / total
        if position > 0.8:
            return "recent"
        elif position > 0.4:
            return "middle"
        else:
            return "historical"
    
    def _determine_market_bias(
        self, patterns: List[DetectedPattern], analysis: ChartAnalysis
    ) -> Tuple[str, float]:
        """Determine overall market bias from patterns and trend"""
        bullish_score = 0.0
        bearish_score = 0.0
        
        # Weight recent patterns more heavily
        location_weights = {"recent": 1.5, "middle": 1.0, "historical": 0.5}
        
        for detected in patterns:
            weight = location_weights.get(detected.location, 1.0)
            score = detected.confidence * weight
            
            if detected.pattern.bias == PatternType.BULLISH:
                bullish_score += score
            elif detected.pattern.bias == PatternType.BEARISH:
                bearish_score += score
        
        # Include trend analysis
        trend = analysis.trend_direction
        trend_weight = analysis.trend_strength * 2
        
        if trend == "uptrend":
            bullish_score += trend_weight
        elif trend == "downtrend":
            bearish_score += trend_weight
        
        # Include color analysis
        color = analysis.dominant_color
        if color == "green":
            bullish_score += 0.5
        elif color == "red":
            bearish_score += 0.5
        
        # Determine bias
        total = bullish_score + bearish_score + 0.01
        
        if bullish_score > bearish_score * 1.3:
            return "bullish", bullish_score / total
        elif bearish_score > bullish_score * 1.3:
            return "bearish", bearish_score / total
        else:
            return "neutral", 0.5
    
    def _calculate_overall_confidence(
        self, patterns: List[DetectedPattern], analysis: ChartAnalysis
    ) -> float:
        """Calculate overall confidence score"""
        if not patterns:
            return 0.3 + analysis.chart_quality * 0.2
        
        # Average of top 3 pattern confidences
        top_confidences = sorted([p.confidence for p in patterns], reverse=True)[:3]
        pattern_confidence = sum(top_confidences) / len(top_confidences)
        
        # Factor in chart quality
        quality_factor = analysis.chart_quality
        
        # Combine
        overall = pattern_confidence * 0.7 + quality_factor * 0.3
        
        return round(overall * 100) / 100  # Round to 2 decimals
    
    def _generate_trend_analysis(self, analysis: ChartAnalysis) -> str:
        """Generate human-readable trend analysis"""
        trend = analysis.trend_direction
        strength = analysis.trend_strength
        color = analysis.dominant_color
        
        strength_word = "strong" if strength > 0.7 else "moderate" if strength > 0.4 else "weak"
        
        if trend == "uptrend":
            base = f"The chart shows a {strength_word} uptrend"
        elif trend == "downtrend":
            base = f"The chart shows a {strength_word} downtrend"
        else:
            base = "The chart shows sideways/consolidating price action"
        
        if color == "green":
            base += " with predominantly bullish (green) candles."
        elif color == "red":
            base += " with predominantly bearish (red) candles."
        else:
            base += " with mixed bullish and bearish candles."
        
        return base
    
    def _generate_reasoning(
        self, patterns: List[DetectedPattern], 
        market_bias: str, 
        analysis: ChartAnalysis
    ) -> str:
        """Generate detailed reasoning for the analysis"""
        lines = []
        
        # Start with trend
        lines.append(self._generate_trend_analysis(analysis))
        lines.append("")
        
        # Describe detected patterns
        if patterns:
            lines.append("Key patterns detected:")
            for i, detected in enumerate(patterns[:5], 1):
                reliability = "high" if detected.pattern.reliability > 0.7 else \
                             "moderate" if detected.pattern.reliability > 0.5 else "low"
                lines.append(f"{i}. {detected.pattern.name} ({detected.location}) - "
                           f"{detected.confidence:.0%} confidence, {reliability} reliability")
                lines.append(f"   {detected.reasoning}")
            lines.append("")
        else:
            lines.append("No strong candlestick patterns detected in the visible chart area.")
            lines.append("")
        
        # Trading implications
        if market_bias == "bullish":
            lines.append("Trading Implication: The overall setup is BULLISH. Consider long positions "
                        "with appropriate risk management. Watch for confirmation on the next candle.")
        elif market_bias == "bearish":
            lines.append("Trading Implication: The overall setup is BEARISH. Consider short positions "
                        "or reducing long exposure. Wait for confirmation before acting.")
        else:
            lines.append("Trading Implication: The market shows INDECISION. Best to wait for clearer "
                        "signals before taking new positions.")
        
        return "\n".join(lines)
    
    def _create_fallback_result(self, analysis: Optional[ChartAnalysis]) -> AnalysisResult:
        """Create a result when pattern detection fails"""
        return AnalysisResult(
            patterns=[],
            market_bias="neutral",
            overall_confidence=0.3,
            trend_analysis="Unable to clearly analyze the chart. The image may be unclear or not contain standard candlestick patterns.",
            reasoning="The analysis could not extract clear candlestick patterns from this image. "
                     "This could be due to low image quality, non-standard chart formatting, or "
                     "image content that doesn't contain candlestick charts. Please upload a clear "
                     "candlestick chart image for better analysis.",
            raw_data=analysis
        )


# Create singleton instance
pattern_detector = PatternDetector()


def analyze_chart_image(image_data: bytes) -> AnalysisResult:
    """
    Main entry point for in-house chart analysis.
    
    Args:
        image_data: Raw image bytes
        
    Returns:
        AnalysisResult with detected patterns and analysis
    """
    return pattern_detector.analyze_image(image_data)
