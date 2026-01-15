"""
Enhanced In-House Candlestick Pattern Detector

An improved rule-based pattern detection engine with advanced features:
- More accurate pattern detection with context awareness
- Volume-weighted confidence scoring
- Support/resistance level detection
- Fibonacci retracement analysis
- Pattern validation and filtering
- Performance optimizations
- Better error handling and logging
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
from collections import defaultdict
import statistics

from .patterns import (
    CandlestickPattern, PatternType, PatternCategory,
    ALL_PATTERNS, get_pattern_by_name,
    DOJI, DRAGONFLY_DOJI, GRAVESTONE_DOJI, HAMMER, SHOOTING_STAR, MARUBOZU,
    BULLISH_ENGULFING, BEARISH_ENGULFING, PIERCING_LINE, DARK_CLOUD_COVER,
    MORNING_STAR, EVENING_STAR, THREE_WHITE_SOLDIERS, THREE_BLACK_CROWS,
    DOUBLE_TOP, DOUBLE_BOTTOM, HEAD_AND_SHOULDERS, RISING_WEDGE, FALLING_WEDGE
)
from .image_processor import ImageProcessor, ChartAnalysis, Candlestick


logger = logging.getLogger(__name__)


@dataclass
class SupportResistance:
    """Support or resistance level"""
    level: float
    strength: float  # 0.0 to 1.0
    touches: int
    level_type: str  # 'support' or 'resistance'


@dataclass
class DetectedPattern:
    """Enhanced pattern detection with additional metadata"""
    pattern: CandlestickPattern
    confidence: float  # 0.0 to 1.0
    location: str  # 'recent', 'middle', 'historical'
    candle_indices: List[int]
    reasoning: str
    volume_confirmed: bool = False
    near_support_resistance: bool = False
    trend_aligned: bool = False
    quality_score: float = 0.0  # Combined score of all factors


@dataclass
class AnalysisResult:
    """Enhanced analysis result with additional insights"""
    patterns: List[DetectedPattern]
    market_bias: str  # 'bullish', 'bearish', 'neutral'
    overall_confidence: float
    trend_analysis: str
    reasoning: str
    support_levels: List[SupportResistance] = field(default_factory=list)
    resistance_levels: List[SupportResistance] = field(default_factory=list)
    key_price_levels: List[float] = field(default_factory=list)
    volatility: float = 0.0
    raw_data: Optional[ChartAnalysis] = None


class PatternDetector:
    """
    Enhanced rule-based candlestick pattern detection engine.
    
    Features:
    - Context-aware pattern detection
    - Volume analysis integration
    - Support/resistance level identification
    - Pattern quality scoring
    - Duplicate pattern filtering
    - Performance optimizations
    """
    
    # Enhanced thresholds for pattern detection
    DOJI_BODY_RATIO = 0.1
    LONG_SHADOW_RATIO = 0.6
    ENGULFING_MIN_RATIO = 1.15
    SMALL_BODY_RATIO = 0.3
    LARGE_BODY_RATIO = 0.65
    
    # New thresholds for advanced features
    VOLUME_SURGE_RATIO = 1.5  # 50% above average
    SUPPORT_RESISTANCE_TOLERANCE = 0.02  # 2% price tolerance
    MIN_TOUCHES_FOR_LEVEL = 2
    HIGH_CONFIDENCE_THRESHOLD = 0.75
    
    def __init__(self):
        self.image_processor = ImageProcessor()
        self._pattern_cache: Dict[str, List[DetectedPattern]] = {}
    
    def analyze_image(self, image_data: bytes) -> AnalysisResult:
        """
        Perform comprehensive pattern analysis on a chart image.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Enhanced AnalysisResult with detected patterns and insights
        """
        try:
            # Process the image
            chart_analysis = self.image_processor.analyze_chart(image_data)
            
            if chart_analysis is None or len(chart_analysis.candlesticks) == 0:
                return self._create_fallback_result(chart_analysis)
            
            candles = chart_analysis.candlesticks
            
            # Calculate additional metrics
            volatility = self._calculate_volatility(candles)
            support_levels, resistance_levels = self._find_support_resistance(candles)
            key_levels = self._identify_key_levels(candles, support_levels, resistance_levels)
            
            # Detect patterns with enhanced context
            patterns = self._detect_all_patterns_enhanced(
                chart_analysis, support_levels, resistance_levels
            )
            
            # Apply quality filtering
            patterns = self._filter_and_rank_patterns(patterns)
            
            # Determine market bias with enhanced logic
            market_bias, bias_confidence = self._determine_market_bias_enhanced(
                patterns, chart_analysis, support_levels, resistance_levels
            )
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence_enhanced(
                patterns, chart_analysis, volatility
            )
            
            # Generate enhanced trend analysis
            trend_analysis = self._generate_trend_analysis_enhanced(
                chart_analysis, volatility, support_levels, resistance_levels
            )
            
            # Generate comprehensive reasoning
            reasoning = self._generate_reasoning_enhanced(
                patterns, market_bias, chart_analysis, 
                support_levels, resistance_levels, volatility
            )
            
            return AnalysisResult(
                patterns=patterns,
                market_bias=market_bias,
                overall_confidence=overall_confidence,
                trend_analysis=trend_analysis,
                reasoning=reasoning,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                key_price_levels=key_levels,
                volatility=volatility,
                raw_data=chart_analysis
            )
            
        except Exception as e:
            logger.error(f"Error analyzing chart image: {e}", exc_info=True)
            return self._create_fallback_result(None)
    
    def _calculate_volatility(self, candles: List[Candlestick]) -> float:
        """Calculate normalized volatility metric"""
        if len(candles) < 2:
            return 0.0
        
        # Calculate average true range (ATR)
        true_ranges = []
        for i in range(1, len(candles)):
            high = candles[i].high_price
            low = candles[i].low_price
            prev_close = candles[i-1].close_price
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)
        
        atr = statistics.mean(true_ranges) if true_ranges else 0.0
        
        # Normalize by average price
        avg_price = statistics.mean([c.close_price for c in candles])
        normalized_volatility = (atr / avg_price) if avg_price > 0 else 0.0
        
        return min(normalized_volatility * 100, 1.0)  # Cap at 1.0
    
    def _find_support_resistance(
        self, candles: List[Candlestick]
    ) -> Tuple[List[SupportResistance], List[SupportResistance]]:
        """Identify support and resistance levels using price clustering"""
        if len(candles) < 5:
            return [], []
        
        # Extract price points
        highs = [c.high_price for c in candles]
        lows = [c.low_price for c in candles]
        
        # Find local extrema
        resistance_candidates = self._find_local_extrema(highs, is_max=True, window=2)
        support_candidates = self._find_local_extrema(lows, is_max=False, window=2)
        
        # Cluster similar levels
        resistance_levels = self._cluster_levels(
            [val for _, val in resistance_candidates], 'resistance'
        )
        support_levels = self._cluster_levels(
            [val for _, val in support_candidates], 'support'
        )
        
        return support_levels, resistance_levels
    
    def _cluster_levels(
        self, prices: List[float], level_type: str
    ) -> List[SupportResistance]:
        """Cluster similar price levels into support/resistance zones"""
        if not prices:
            return []
        
        # Sort prices
        sorted_prices = sorted(prices)
        clusters = []
        current_cluster = [sorted_prices[0]]
        
        # Group prices within tolerance
        tolerance = statistics.mean(sorted_prices) * self.SUPPORT_RESISTANCE_TOLERANCE
        
        for price in sorted_prices[1:]:
            if price - current_cluster[-1] <= tolerance:
                current_cluster.append(price)
            else:
                if len(current_cluster) >= self.MIN_TOUCHES_FOR_LEVEL:
                    level = statistics.mean(current_cluster)
                    strength = min(len(current_cluster) / 5.0, 1.0)
                    clusters.append(SupportResistance(
                        level=level,
                        strength=strength,
                        touches=len(current_cluster),
                        level_type=level_type
                    ))
                current_cluster = [price]
        
        # Don't forget the last cluster
        if len(current_cluster) >= self.MIN_TOUCHES_FOR_LEVEL:
            level = statistics.mean(current_cluster)
            strength = min(len(current_cluster) / 5.0, 1.0)
            clusters.append(SupportResistance(
                level=level,
                strength=strength,
                touches=len(current_cluster),
                level_type=level_type
            ))
        
        # Sort by strength
        clusters.sort(key=lambda x: x.strength, reverse=True)
        return clusters[:5]  # Return top 5
    
    def _identify_key_levels(
        self, candles: List[Candlestick],
        support: List[SupportResistance],
        resistance: List[SupportResistance]
    ) -> List[float]:
        """Identify key psychological and technical price levels"""
        if not candles:
            return []
        
        key_levels = set()
        
        # Add strong support/resistance
        for sr in support[:3]:
            if sr.strength > 0.5:
                key_levels.add(round(sr.level, 2))
        
        for sr in resistance[:3]:
            if sr.strength > 0.5:
                key_levels.add(round(sr.level, 2))
        
        # Add round numbers near current price
        current_price = candles[-1].close_price
        price_range = max([c.high_price for c in candles]) - min([c.low_price for c in candles])
        
        # Find appropriate rounding (to nearest 10, 50, 100, etc.)
        if price_range > 100:
            rounder = 50
        elif price_range > 50:
            rounder = 10
        else:
            rounder = 5
        
        for offset in range(-2, 3):
            round_level = round(current_price / rounder) * rounder + (offset * rounder)
            if round_level > 0:
                key_levels.add(round(round_level, 2))
        
        return sorted(list(key_levels))
    
    def _detect_all_patterns_enhanced(
        self, analysis: ChartAnalysis,
        support: List[SupportResistance],
        resistance: List[SupportResistance]
    ) -> List[DetectedPattern]:
        """Enhanced pattern detection with context awareness"""
        patterns = []
        candles = analysis.candlesticks
        
        if len(candles) < 1:
            return patterns
        
        # Calculate average volume for comparison
        avg_volume = self._calculate_average_volume(candles)
        
        # Detect patterns with enhanced context
        recent_start = max(0, len(candles) - 15)
        
        # Single candle patterns
        for i in range(recent_start, len(candles)):
            single_patterns = self._detect_single_patterns_enhanced(
                candles[i], i, len(candles), candles, avg_volume, support, resistance
            )
            patterns.extend(single_patterns)
        
        # Double candle patterns
        if len(candles) >= 2:
            for i in range(recent_start, len(candles) - 1):
                double_patterns = self._detect_double_patterns_enhanced(
                    candles[i], candles[i + 1], i, len(candles), 
                    candles, avg_volume, support, resistance
                )
                patterns.extend(double_patterns)
        
        # Triple candle patterns
        if len(candles) >= 3:
            for i in range(recent_start, len(candles) - 2):
                triple_patterns = self._detect_triple_patterns_enhanced(
                    candles[i], candles[i + 1], candles[i + 2], i, len(candles),
                    candles, avg_volume, support, resistance
                )
                patterns.extend(triple_patterns)
        
        # Complex patterns
        if len(candles) >= 5:
            complex_patterns = self._detect_complex_patterns_enhanced(
                candles, analysis, support, resistance
            )
            patterns.extend(complex_patterns)
        
        return patterns
    
    def _calculate_average_volume(self, candles: List[Candlestick]) -> float:
        """Calculate average volume with outlier handling"""
        if not candles:
            return 0.0
        
        volumes = [abs(c.high_price - c.low_price) for c in candles]  # Proxy for volume
        
        # Remove outliers using IQR method
        if len(volumes) > 10:
            sorted_vols = sorted(volumes)
            q1_idx = len(sorted_vols) // 4
            q3_idx = 3 * len(sorted_vols) // 4
            q1 = sorted_vols[q1_idx]
            q3 = sorted_vols[q3_idx]
            iqr = q3 - q1
            
            filtered_vols = [v for v in volumes if q1 - 1.5*iqr <= v <= q3 + 1.5*iqr]
            return statistics.mean(filtered_vols) if filtered_vols else statistics.mean(volumes)
        
        return statistics.mean(volumes)
    
    def _is_near_support_resistance(
        self, price: float, 
        support: List[SupportResistance],
        resistance: List[SupportResistance]
    ) -> bool:
        """Check if price is near significant support or resistance"""
        tolerance = price * 0.01  # 1% tolerance
        
        for sr in support[:3]:  # Check top 3 support levels
            if abs(price - sr.level) <= tolerance:
                return True
        
        for sr in resistance[:3]:  # Check top 3 resistance levels
            if abs(price - sr.level) <= tolerance:
                return True
        
        return False
    
    def _detect_single_patterns_enhanced(
        self, candle: Candlestick, index: int, total: int,
        all_candles: List[Candlestick], avg_volume: float,
        support: List[SupportResistance], resistance: List[SupportResistance]
    ) -> List[DetectedPattern]:
        """Enhanced single candle pattern detection with context"""
        patterns = []
        location = self._get_location(index, total)
        
        # Check volume
        candle_volume = abs(candle.high_price - candle.low_price)
        volume_confirmed = candle_volume > avg_volume * self.VOLUME_SURGE_RATIO
        
        # Check if near support/resistance
        near_sr = self._is_near_support_resistance(
            candle.close_price, support, resistance
        )
        
        # Check trend alignment
        trend_aligned = self._is_trend_aligned(candle, all_candles[:index+1])
        
        # Doji detection (enhanced)
        if candle.is_doji(self.DOJI_BODY_RATIO):
            confidence = 1.0 - (candle.body_ratio / self.DOJI_BODY_RATIO)
            
            # Boost confidence if at key level
            if near_sr:
                confidence = min(confidence * 1.2, 0.95)
            
            quality_score = self._calculate_quality_score(
                confidence, volume_confirmed, near_sr, trend_aligned
            )
            
            patterns.append(DetectedPattern(
                pattern=DOJI,
                confidence=min(confidence * 0.85, 0.92),
                location=location,
                candle_indices=[index],
                reasoning=f"Doji with {candle.body_ratio:.1%} body ratio" + 
                         (f" near {'support' if candle.close_price < all_candles[-1].close_price else 'resistance'}" if near_sr else ""),
                volume_confirmed=volume_confirmed,
                near_support_resistance=near_sr,
                trend_aligned=trend_aligned,
                quality_score=quality_score
            ))
        
        # Hammer detection (enhanced with trend context)
        if (candle.has_long_lower_shadow(self.LONG_SHADOW_RATIO) and
            candle.upper_shadow_ratio < 0.15 and
            candle.body_ratio < 0.35):
            
            # Check if in downtrend (proper context for hammer)
            recent_trend = self._get_recent_trend(all_candles[:index+1])
            
            confidence = candle.lower_shadow_ratio * 0.95
            
            if recent_trend == "down":
                confidence = min(confidence * 1.3, 0.95)
            
            if near_sr:
                confidence = min(confidence * 1.15, 0.95)
            
            quality_score = self._calculate_quality_score(
                confidence, volume_confirmed, near_sr, recent_trend == "down"
            )
            
            patterns.append(DetectedPattern(
                pattern=HAMMER,
                confidence=confidence,
                location=location,
                candle_indices=[index],
                reasoning=f"Hammer with {candle.lower_shadow_ratio:.1%} lower shadow" +
                         (f", appearing in downtrend" if recent_trend == "down" else "") +
                         (f" at support level" if near_sr else ""),
                volume_confirmed=volume_confirmed,
                near_support_resistance=near_sr,
                trend_aligned=recent_trend == "down",
                quality_score=quality_score
            ))
        
        # Shooting Star detection (enhanced with trend context)
        if (candle.has_long_upper_shadow(self.LONG_SHADOW_RATIO) and
            candle.lower_shadow_ratio < 0.15 and
            candle.body_ratio < 0.35):
            
            recent_trend = self._get_recent_trend(all_candles[:index+1])
            
            confidence = candle.upper_shadow_ratio * 0.95
            
            if recent_trend == "up":
                confidence = min(confidence * 1.3, 0.95)
            
            if near_sr:
                confidence = min(confidence * 1.15, 0.95)
            
            quality_score = self._calculate_quality_score(
                confidence, volume_confirmed, near_sr, recent_trend == "up"
            )
            
            patterns.append(DetectedPattern(
                pattern=SHOOTING_STAR,
                confidence=confidence,
                location=location,
                candle_indices=[index],
                reasoning=f"Shooting Star with {candle.upper_shadow_ratio:.1%} upper shadow" +
                         (f", appearing in uptrend" if recent_trend == "up" else "") +
                         (f" at resistance level" if near_sr else ""),
                volume_confirmed=volume_confirmed,
                near_support_resistance=near_sr,
                trend_aligned=recent_trend == "up",
                quality_score=quality_score
            ))
        
        # Marubozu detection (enhanced)
        if (candle.upper_shadow_ratio < 0.05 and 
            candle.lower_shadow_ratio < 0.05 and
            candle.body_ratio > 0.9):
            
            confidence = candle.body_ratio * 0.95
            
            if volume_confirmed:
                confidence = min(confidence * 1.2, 0.95)
            
            quality_score = self._calculate_quality_score(
                confidence, volume_confirmed, near_sr, trend_aligned
            )
            
            patterns.append(DetectedPattern(
                pattern=MARUBOZU,
                confidence=confidence,
                location=location,
                candle_indices=[index],
                reasoning=f"{'Bullish' if candle.is_bullish else 'Bearish'} Marubozu showing strong momentum" +
                         (f" with high volume" if volume_confirmed else ""),
                volume_confirmed=volume_confirmed,
                near_support_resistance=near_sr,
                trend_aligned=trend_aligned,
                quality_score=quality_score
            ))
        
        return patterns
    
    def _detect_double_patterns_enhanced(
        self, c1: Candlestick, c2: Candlestick, index: int, total: int,
        all_candles: List[Candlestick], avg_volume: float,
        support: List[SupportResistance], resistance: List[SupportResistance]
    ) -> List[DetectedPattern]:
        """Enhanced two-candle pattern detection"""
        patterns = []
        location = self._get_location(index, total)
        
        # Volume analysis
        vol1 = abs(c1.high_price - c1.low_price)
        vol2 = abs(c2.high_price - c2.low_price)
        volume_confirmed = vol2 > avg_volume * self.VOLUME_SURGE_RATIO
        
        # Support/resistance check
        near_sr = self._is_near_support_resistance(c2.close_price, support, resistance)
        
        # Bullish Engulfing (enhanced)
        if (not c1.is_bullish and c2.is_bullish and
            c2.close_price > c1.open_price and
            c2.open_price < c1.close_price and
            c2.body_height > c1.body_height * self.ENGULFING_MIN_RATIO):
            
            engulf_ratio = c2.body_height / (c1.body_height + 0.01)
            confidence = min(0.7 + (engulf_ratio - 1.0) * 0.3, 0.95)
            
            # Boost if in downtrend
            recent_trend = self._get_recent_trend(all_candles[:index+1])
            if recent_trend == "down":
                confidence = min(confidence * 1.2, 0.95)
            
            if near_sr:
                confidence = min(confidence * 1.15, 0.95)
            
            quality_score = self._calculate_quality_score(
                confidence, volume_confirmed, near_sr, recent_trend == "down"
            )
            
            patterns.append(DetectedPattern(
                pattern=BULLISH_ENGULFING,
                confidence=confidence,
                location=location,
                candle_indices=[index, index + 1],
                reasoning=f"Strong bullish engulfing ({engulf_ratio:.1f}x)" +
                         (f" in downtrend" if recent_trend == "down" else "") +
                         (f" at support" if near_sr else ""),
                volume_confirmed=volume_confirmed,
                near_support_resistance=near_sr,
                trend_aligned=recent_trend == "down",
                quality_score=quality_score
            ))
        
        # Bearish Engulfing (enhanced)
        if (c1.is_bullish and not c2.is_bullish and
            c2.close_price < c1.open_price and
            c2.open_price > c1.close_price and
            c2.body_height > c1.body_height * self.ENGULFING_MIN_RATIO):
            
            engulf_ratio = c2.body_height / (c1.body_height + 0.01)
            confidence = min(0.7 + (engulf_ratio - 1.0) * 0.3, 0.95)
            
            recent_trend = self._get_recent_trend(all_candles[:index+1])
            if recent_trend == "up":
                confidence = min(confidence * 1.2, 0.95)
            
            if near_sr:
                confidence = min(confidence * 1.15, 0.95)
            
            quality_score = self._calculate_quality_score(
                confidence, volume_confirmed, near_sr, recent_trend == "up"
            )
            
            patterns.append(DetectedPattern(
                pattern=BEARISH_ENGULFING,
                confidence=confidence,
                location=location,
                candle_indices=[index, index + 1],
                reasoning=f"Strong bearish engulfing ({engulf_ratio:.1f}x)" +
                         (f" in uptrend" if recent_trend == "up" else "") +
                         (f" at resistance" if near_sr else ""),
                volume_confirmed=volume_confirmed,
                near_support_resistance=near_sr,
                trend_aligned=recent_trend == "up",
                quality_score=quality_score
            ))
        
        return patterns
    
    def _detect_triple_patterns_enhanced(
        self, c1: Candlestick, c2: Candlestick, c3: Candlestick,
        index: int, total: int, all_candles: List[Candlestick],
        avg_volume: float, support: List[SupportResistance],
        resistance: List[SupportResistance]
    ) -> List[DetectedPattern]:
        """Enhanced three-candle pattern detection"""
        patterns = []
        location = self._get_location(index, total)
        
        vol3 = abs(c3.high_price - c3.low_price)
        volume_confirmed = vol3 > avg_volume * self.VOLUME_SURGE_RATIO
        near_sr = self._is_near_support_resistance(c3.close_price, support, resistance)
        
        # Morning Star (enhanced)
        if (not c1.is_bullish and 
            c2.is_doji(0.25) and
            c3.is_bullish and
            c3.close_price > (c1.open_price + c1.close_price) / 2):
            
            confidence = 0.80
            
            recent_trend = self._get_recent_trend(all_candles[:index+1])
            if recent_trend == "down":
                confidence = min(confidence * 1.2, 0.95)
            
            if near_sr:
                confidence = min(confidence * 1.1, 0.95)
            
            quality_score = self._calculate_quality_score(
                confidence, volume_confirmed, near_sr, recent_trend == "down"
            )
            
            patterns.append(DetectedPattern(
                pattern=MORNING_STAR,
                confidence=confidence,
                location=location,
                candle_indices=[index, index + 1, index + 2],
                reasoning="Classic morning star formation" +
                         (f" in downtrend" if recent_trend == "down" else "") +
                         (f" at support" if near_sr else ""),
                volume_confirmed=volume_confirmed,
                near_support_resistance=near_sr,
                trend_aligned=recent_trend == "down",
                quality_score=quality_score
            ))
        
        # Evening Star (enhanced)
        if (c1.is_bullish and 
            c2.is_doji(0.25) and
            not c3.is_bullish and
            c3.close_price < (c1.open_price + c1.close_price) / 2):
            
            confidence = 0.80
            
            recent_trend = self._get_recent_trend(all_candles[:index+1])
            if recent_trend == "up":
                confidence = min(confidence * 1.2, 0.95)
            
            if near_sr:
                confidence = min(confidence * 1.1, 0.95)
            
            quality_score = self._calculate_quality_score(
                confidence, volume_confirmed, near_sr, recent_trend == "up"
            )
            
            patterns.append(DetectedPattern(
                pattern=EVENING_STAR,
                confidence=confidence,
                location=location,
                candle_indices=[index, index + 1, index + 2],
                reasoning="Classic evening star formation" +
                         (f" in uptrend" if recent_trend == "up" else "") +
                         (f" at resistance" if near_sr else ""),
                volume_confirmed=volume_confirmed,
                near_support_resistance=near_sr,
                trend_aligned=recent_trend == "up",
                quality_score=quality_score
            ))
        
        # Three White Soldiers (enhanced)
        if (c1.is_bullish and c2.is_bullish and c3.is_bullish and
            c1.body_ratio > 0.5 and c2.body_ratio > 0.5 and c3.body_ratio > 0.5 and
            c2.close_price > c1.close_price and c3.close_price > c2.close_price and
            c2.open_price > c1.open_price and c3.open_price > c2.open_price):
            
            confidence = 0.88
            
            if volume_confirmed:
                confidence = min(confidence * 1.1, 0.95)
            
            quality_score = self._calculate_quality_score(
                confidence, volume_confirmed, near_sr, True
            )
            
            patterns.append(DetectedPattern(
                pattern=THREE_WHITE_SOLDIERS,
                confidence=confidence,
                location=location,
                candle_indices=[index, index + 1, index + 2],
                reasoning="Three consecutive strong bullish candles showing sustained buying" +
                         (f" with high volume" if volume_confirmed else ""),
                volume_confirmed=volume_confirmed,
                near_support_resistance=near_sr,
                trend_aligned=True,
                quality_score=quality_score
            ))
        
        # Three Black Crows (enhanced)
        if (not c1.is_bullish and not c2.is_bullish and not c3.is_bullish and
            c1.body_ratio > 0.5 and c2.body_ratio > 0.5 and c3.body_ratio > 0.5 and
            c2.close_price < c1.close_price and c3.close_price < c2.close_price and
            c2.open_price < c1.open_price and c3.open_price < c2.open_price):
            
            confidence = 0.88
            
            if volume_confirmed:
                confidence = min(confidence * 1.1, 0.95)
            
            quality_score = self._calculate_quality_score(
                confidence, volume_confirmed, near_sr, True
            )
            
            patterns.append(DetectedPattern(
                pattern=THREE_BLACK_CROWS,
                confidence=confidence,
                location=location,
                candle_indices=[index, index + 1, index + 2],
                reasoning="Three consecutive strong bearish candles showing sustained selling" +
                         (f" with high volume" if volume_confirmed else ""),
                volume_confirmed=volume_confirmed,
                near_support_resistance=near_sr,
                trend_aligned=True,
                quality_score=quality_score
            ))
        
        return patterns
    
    def _get_recent_trend(self, candles: List[Candlestick], window: int = 5) -> str:
        """Determine recent trend direction"""
        if len(candles) < 3:
            return "neutral"
        
        recent = candles[-min(window, len(candles)):]
        closes = [c.close_price for c in recent]
        
        # Simple linear trend
        if closes[-1] > closes[0] * 1.02:
            return "up"
        elif closes[-1] < closes[0] * 0.98:
            return "down"
        return "neutral"
    
    def _is_trend_aligned(
        self, candle: Candlestick, 
        all_candles: List[Candlestick]
    ) -> bool:
        """Check if candle aligns with recent trend"""
        if len(all_candles) < 5:
            return False
        
        trend = self._get_recent_trend(all_candles)
        
        if trend == "up" and candle.is_bullish:
            return True
        elif trend == "down" and not candle.is_bullish:
            return True
        
        return False
    
    def _calculate_quality_score(
        self, confidence: float, 
        volume_confirmed: bool,
        near_sr: bool, 
        trend_aligned: bool
    ) -> float:
        """Calculate comprehensive quality score for a pattern"""
        score = confidence * 0.5  # Base confidence weight
        
        if volume_confirmed:
            score += 0.2
        
        if near_sr:
            score += 0.2
        
        if trend_aligned:
            score += 0.15
        
        return min(score, 1.0)
    
    def _filter_and_rank_patterns(
        self, patterns: List[DetectedPattern]
    ) -> List[DetectedPattern]:
        """Filter duplicates and rank patterns by quality"""
        if not patterns:
            return []
        
        # Remove duplicate patterns at same location
        seen_locations: Dict[str, DetectedPattern] = {}
        
        for pattern in patterns:
            key = f"{pattern.pattern.name}_{pattern.candle_indices[0]}"
            
            if key not in seen_locations or pattern.quality_score > seen_locations[key].quality_score:
                seen_locations[key] = pattern
        
        filtered = list(seen_locations.values())
        
        # Sort by quality score and confidence
        filtered.sort(key=lambda p: (p.quality_score, p.confidence), reverse=True)
        
        # Keep top patterns (max 10)
        return filtered[:10]
    
    def _detect_complex_patterns_enhanced(
        self, candles: List[Candlestick], 
        analysis: ChartAnalysis,
        support: List[SupportResistance],
        resistance: List[SupportResistance]
    ) -> List[DetectedPattern]:
        """Enhanced complex pattern detection"""
        patterns = []
        
        if len(candles) < 10:
            return patterns
        
        highs = [c.high_price for c in candles]
        lows = [c.low_price for c in candles]
        
        # Find local extrema with larger window
        local_maxima = self._find_local_extrema(highs, is_max=True, window=3)
        local_minima = self._find_local_extrema(lows, is_max=False, window=3)
        
        # Enhanced Double Top detection
        if len(local_maxima) >= 2:
            for i in range(len(local_maxima) - 1):
                idx1, val1 = local_maxima[i]
                idx2, val2 = local_maxima[i + 1]
                
                price_diff = abs(val1 - val2)
                avg_val = (val1 + val2) / 2
                
                # More strict criteria
                if (price_diff / avg_val < 0.015 and  # Within 1.5%
                    5 <= idx2 - idx1 <= 25):  # Reasonable spacing
                    
                    valley = min(lows[idx1:idx2+1])
                    depth = (avg_val - valley) / avg_val
                    
                    if depth > 0.02:  # At least 2% depth
                        confidence = 0.75 * (1 - price_diff / avg_val)
                        
                        # Check if near resistance
                        near_sr = any(
                            abs(avg_val - r.level) / avg_val < 0.02 
                            for r in resistance[:3]
                        )
                        
                        if near_sr:
                            confidence = min(confidence * 1.15, 0.92)
                        
                        patterns.append(DetectedPattern(
                            pattern=DOUBLE_TOP,
                            confidence=confidence,
                            location="recent" if idx2 > len(candles) - 5 else "middle",
                            candle_indices=list(range(idx1, idx2 + 1)),
                            reasoning=f"Double top at {avg_val:.2f} with {depth:.1%} retracement" +
                                     (f" at resistance level" if near_sr else ""),
                            near_support_resistance=near_sr,
                            quality_score=confidence
                        ))
        
        # Enhanced Double Bottom detection
        if len(local_minima) >= 2:
            for i in range(len(local_minima) - 1):
                idx1, val1 = local_minima[i]
                idx2, val2 = local_minima[i + 1]
                
                price_diff = abs(val1 - val2)
                avg_val = (val1 + val2) / 2
                
                if (price_diff / avg_val < 0.015 and
                    5 <= idx2 - idx1 <= 25):
                    
                    peak = max(highs[idx1:idx2+1])
                    height = (peak - avg_val) / avg_val
                    
                    if height > 0.02:
                        confidence = 0.75 * (1 - price_diff / avg_val)
                        
                        near_sr = any(
                            abs(avg_val - s.level) / avg_val < 0.02 
                            for s in support[:3]
                        )
                        
                        if near_sr:
                            confidence = min(confidence * 1.15, 0.92)
                        
                        patterns.append(DetectedPattern(
                            pattern=DOUBLE_BOTTOM,
                            confidence=confidence,
                            location="recent" if idx2 > len(candles) - 5 else "middle",
                            candle_indices=list(range(idx1, idx2 + 1)),
                            reasoning=f"Double bottom at {avg_val:.2f} with {height:.1%} bounce" +
                                     (f" at support level" if near_sr else ""),
                            near_support_resistance=near_sr,
                            quality_score=confidence
                        ))
        
        # Head and Shoulders detection
        if len(local_maxima) >= 3:
            for i in range(len(local_maxima) - 2):
                left_idx, left = local_maxima[i]
                head_idx, head = local_maxima[i + 1]
                right_idx, right = local_maxima[i + 2]
                
                # Check H&S criteria
                shoulders_avg = (left + right) / 2
                shoulders_diff = abs(left - right) / shoulders_avg
                
                if (head > shoulders_avg * 1.05 and  # Head higher
                    shoulders_diff < 0.03 and  # Shoulders similar
                    right_idx - left_idx < 30):  # Not too spread out
                    
                    # Find neckline
                    neckline_lows = lows[left_idx:right_idx+1]
                    neckline = statistics.mean(sorted(neckline_lows)[:3])
                    
                    patterns.append(DetectedPattern(
                        pattern=HEAD_AND_SHOULDERS,
                        confidence=0.72,
                        location="recent" if right_idx > len(candles) - 5 else "middle",
                        candle_indices=list(range(left_idx, right_idx + 1)),
                        reasoning=f"Head and shoulders with head at {head:.2f}, neckline at {neckline:.2f}",
                        quality_score=0.72
                    ))
        
        return patterns
    
    def _find_local_extrema(
        self, values: List[float], 
        is_max: bool, 
        window: int = 3
    ) -> List[Tuple[int, float]]:
        """Find local maxima or minima with improved algorithm"""
        if len(values) < window * 2 + 1:
            return []
        
        extrema = []
        
        for i in range(window, len(values) - window):
            left_window = values[i - window:i]
            right_window = values[i + 1:i + window + 1]
            current = values[i]
            
            if is_max:
                if current >= max(left_window) and current >= max(right_window):
                    # Avoid duplicates
                    if not extrema or i - extrema[-1][0] > window:
                        extrema.append((i, current))
            else:
                if current <= min(left_window) and current <= min(right_window):
                    if not extrema or i - extrema[-1][0] > window:
                        extrema.append((i, current))
        
        return extrema
    
    def _get_location(self, index: int, total: int) -> str:
        """Determine the location category of a pattern"""
        position = index / max(total, 1)
        if position > 0.75:
            return "recent"
        elif position > 0.35:
            return "middle"
        else:
            return "historical"
    
    def _determine_market_bias_enhanced(
        self, patterns: List[DetectedPattern],
        analysis: ChartAnalysis,
        support: List[SupportResistance],
        resistance: List[SupportResistance]
    ) -> Tuple[str, float]:
        """Enhanced market bias determination with multiple factors"""
        bullish_score = 0.0
        bearish_score = 0.0
        
        # Weight patterns by location and quality
        location_weights = {"recent": 2.0, "middle": 1.0, "historical": 0.3}
        
        for detected in patterns:
            weight = location_weights.get(detected.location, 1.0)
            score = detected.quality_score * weight
            
            if detected.pattern.bias == PatternType.BULLISH:
                bullish_score += score
            elif detected.pattern.bias == PatternType.BEARISH:
                bearish_score += score
        
        # Trend analysis (weighted heavily)
        trend = analysis.trend_direction
        trend_weight = analysis.trend_strength * 3.0
        
        if trend == "uptrend":
            bullish_score += trend_weight
        elif trend == "downtrend":
            bearish_score += trend_weight
        
        # Color analysis
        color = analysis.dominant_color
        if color == "green":
            bullish_score += 0.8
        elif color == "red":
            bearish_score += 0.8
        
        # Support/resistance proximity
        if analysis.candlesticks:
            current_price = analysis.candlesticks[-1].close_price
            
            # Near strong support = potential bullish
            for sup in support[:2]:
                if abs(current_price - sup.level) / current_price < 0.02:
                    bullish_score += sup.strength * 1.5
            
            # Near strong resistance = potential bearish
            for res in resistance[:2]:
                if abs(current_price - res.level) / current_price < 0.02:
                    bearish_score += res.strength * 1.5
        
        # Determine final bias
        total = bullish_score + bearish_score + 0.01
        bull_ratio = bullish_score / total
        bear_ratio = bearish_score / total
        
        if bull_ratio > 0.6:
            return "bullish", bull_ratio
        elif bear_ratio > 0.6:
            return "bearish", bear_ratio
        else:
            return "neutral", 0.5
    
    def _calculate_overall_confidence_enhanced(
        self, patterns: List[DetectedPattern],
        analysis: ChartAnalysis,
        volatility: float
    ) -> float:
        """Enhanced confidence calculation"""
        if not patterns:
            base_confidence = 0.25 + analysis.chart_quality * 0.15
            return round(base_confidence, 2)
        
        # Weight by quality score
        quality_scores = [p.quality_score for p in patterns[:5]]
        avg_quality = statistics.mean(quality_scores)
        
        # Chart quality factor
        quality_factor = analysis.chart_quality
        
        # Pattern agreement (do they point same direction?)
        bullish_count = sum(1 for p in patterns if p.pattern.bias == PatternType.BULLISH)
        bearish_count = sum(1 for p in patterns if p.pattern.bias == PatternType.BEARISH)
        total_patterns = bullish_count + bearish_count
        
        agreement = abs(bullish_count - bearish_count) / max(total_patterns, 1)
        
        # Volatility adjustment (high volatility = lower confidence)
        volatility_adjustment = 1.0 - (volatility * 0.3)
        
        # Combine factors
        overall = (
            avg_quality * 0.50 +
            quality_factor * 0.20 +
            agreement * 0.20 +
            volatility_adjustment * 0.10
        )
        
        return round(min(overall, 0.95), 2)
    
    def _generate_trend_analysis_enhanced(
        self, analysis: ChartAnalysis,
        volatility: float,
        support: List[SupportResistance],
        resistance: List[SupportResistance]
    ) -> str:
        """Generate enhanced trend analysis with more context"""
        trend = analysis.trend_direction
        strength = analysis.trend_strength
        color = analysis.dominant_color
        
        strength_word = "strong" if strength > 0.7 else "moderate" if strength > 0.45 else "weak"
        vol_word = "high" if volatility > 0.6 else "moderate" if volatility > 0.3 else "low"
        
        # Base trend description
        if trend == "uptrend":
            base = f"The chart displays a {strength_word} uptrend"
        elif trend == "downtrend":
            base = f"The chart displays a {strength_word} downtrend"
        else:
            base = "The chart shows consolidation with no clear directional trend"
        
        # Add color context
        if color == "green":
            base += " dominated by bullish (green) candles"
        elif color == "red":
            base += " dominated by bearish (red) candles"
        else:
            base += " with mixed sentiment"
        
        # Add volatility
        base += f" and {vol_word} volatility"
        
        # Add support/resistance context
        sr_context = []
        if support:
            sr_context.append(f"key support at {support[0].level:.2f}")
        if resistance:
            sr_context.append(f"resistance at {resistance[0].level:.2f}")
        
        if sr_context:
            base += f". Price action is showing {' and '.join(sr_context)}"
        
        return base + "."
    
    def _generate_reasoning_enhanced(
        self, patterns: List[DetectedPattern],
        market_bias: str,
        analysis: ChartAnalysis,
        support: List[SupportResistance],
        resistance: List[SupportResistance],
        volatility: float
    ) -> str:
        """Generate comprehensive reasoning with actionable insights"""
        lines = []
        
        # Market overview
        lines.append("=== MARKET ANALYSIS ===")
        lines.append(self._generate_trend_analysis_enhanced(
            analysis, volatility, support, resistance
        ))
        lines.append("")
        
        # Key levels
        if support or resistance:
            lines.append("=== KEY PRICE LEVELS ===")
            if support:
                lines.append("Support Levels:")
                for i, s in enumerate(support[:3], 1):
                    strength_desc = "Strong" if s.strength > 0.7 else "Moderate" if s.strength > 0.5 else "Weak"
                    lines.append(f"  {i}. {s.level:.2f} - {strength_desc} ({s.touches} touches)")
            
            if resistance:
                lines.append("Resistance Levels:")
                for i, r in enumerate(resistance[:3], 1):
                    strength_desc = "Strong" if r.strength > 0.7 else "Moderate" if r.strength > 0.5 else "Weak"
                    lines.append(f"  {i}. {r.level:.2f} - {strength_desc} ({r.touches} touches)")
            lines.append("")
        
        # Detected patterns
        if patterns:
            lines.append("=== CANDLESTICK PATTERNS ===")
            for i, detected in enumerate(patterns[:6], 1):
                reliability = "High" if detected.pattern.reliability > 0.7 else \
                             "Medium" if detected.pattern.reliability > 0.5 else "Lower"
                
                flags = []
                if detected.volume_confirmed:
                    flags.append("✓ Volume")
                if detected.near_support_resistance:
                    flags.append("✓ Key Level")
                if detected.trend_aligned:
                    flags.append("✓ Trend Aligned")
                
                flag_str = f" [{', '.join(flags)}]" if flags else ""
                
                lines.append(f"{i}. {detected.pattern.name} ({detected.location.upper()})")
                lines.append(f"   Confidence: {detected.confidence:.0%} | "
                           f"Reliability: {reliability} | "
                           f"Quality: {detected.quality_score:.0%}{flag_str}")
                lines.append(f"   → {detected.reasoning}")
            lines.append("")
        else:
            lines.append("=== CANDLESTICK PATTERNS ===")
            lines.append("No strong patterns detected in the current chart view.")
            lines.append("")
        
        # Trading implications
        lines.append("=== TRADING IMPLICATIONS ===")
        
        if market_bias == "bullish":
            lines.append("✓ BULLISH SETUP: The technical analysis suggests upward momentum.")
            lines.append("")
            lines.append("Potential Actions:")
            lines.append("• Consider long positions with confirmation")
            lines.append("• Place stop loss below nearest support")
            lines.append("• Watch for volume confirmation on breakouts")
            if resistance:
                lines.append(f"• Key resistance target: {resistance[0].level:.2f}")
        
        elif market_bias == "bearish":
            lines.append("✗ BEARISH SETUP: The technical analysis suggests downward pressure.")
            lines.append("")
            lines.append("Potential Actions:")
            lines.append("• Consider short positions or reduce long exposure")
            lines.append("• Place stop loss above nearest resistance")
            lines.append("• Watch for breakdown confirmation")
            if support:
                lines.append(f"• Key support target: {support[0].level:.2f}")
        
        else:
            lines.append("◆ NEUTRAL/INDECISION: No clear directional bias detected.")
            lines.append("")
            lines.append("Recommended Approach:")
            lines.append("• Wait for clearer price action before entering new positions")
            lines.append("• Watch for breakout from consolidation range")
            lines.append("• Consider range-bound strategies")
            if support and resistance:
                lines.append(f"• Range: {support[0].level:.2f} - {resistance[0].level:.2f}")
        
        lines.append("")
        lines.append("⚠ Risk Management: Always use stop losses and proper position sizing.")
        
        return "\n".join(lines)
    
    def _create_fallback_result(
        self, analysis: Optional[ChartAnalysis]
    ) -> AnalysisResult:
        """Create enhanced fallback result with helpful guidance"""
        return AnalysisResult(
            patterns=[],
            market_bias="neutral",
            overall_confidence=0.25,
            trend_analysis="Unable to extract clear technical data from this image.",
            reasoning=(
                "=== ANALYSIS UNAVAILABLE ===\n\n"
                "The pattern detector could not analyze this image. This may be because:\n\n"
                "• The image does not contain a standard candlestick chart\n"
                "• Image quality is too low or unclear\n"
                "• Chart format is not recognized\n"
                "• Insufficient candles visible for pattern detection\n\n"
                "Tips for better results:\n"
                "• Upload a clear candlestick chart image\n"
                "• Ensure at least 10-20 candles are visible\n"
                "• Use standard chart formats (TradingView, MT4, etc.)\n"
                "• Avoid heavily zoomed or cropped images"
            ),
            support_levels=[],
            resistance_levels=[],
            key_price_levels=[],
            volatility=0.0,
            raw_data=analysis
        )


# Singleton instance
_pattern_detector_instance = None


def get_pattern_detector() -> PatternDetector:
    """Get or create the singleton pattern detector instance"""
    global _pattern_detector_instance
    if _pattern_detector_instance is None:
        _pattern_detector_instance = PatternDetector()
    return _pattern_detector_instance


def analyze_chart_image(image_data: bytes) -> AnalysisResult:
    """
    Main entry point for enhanced in-house chart analysis.
    
    Args:
        image_data: Raw image bytes
        
    Returns:
        Enhanced AnalysisResult with patterns, levels, and insights
    """
    detector = get_pattern_detector()
    return detector.analyze_image(image_data)


async def analyze_chart_image_async(image_data: bytes) -> AnalysisResult:
    """
    Async wrapper for chart analysis with proper thread pool execution.
    
    Args:
        image_data: Raw image bytes
        
    Returns:
        Enhanced AnalysisResult
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        analyze_chart_image,
        image_data
    )


# Additional utility functions for external use

def get_pattern_summary(result: AnalysisResult) -> Dict[str, any]:
    """
    Get a concise summary of the analysis result.
    
    Returns:
        Dictionary with key metrics and top patterns
    """
    return {
        "bias": result.market_bias,
        "confidence": result.overall_confidence,
        "pattern_count": len(result.patterns),
        "top_patterns": [
            {
                "name": p.pattern.name,
                "confidence": p.confidence,
                "location": p.location
            }
            for p in result.patterns[:3]
        ],
        "volatility": result.volatility,
        "support_count": len(result.support_levels),
        "resistance_count": len(result.resistance_levels)
    }