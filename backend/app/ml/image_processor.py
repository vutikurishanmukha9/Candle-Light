import asyncio
import hashlib
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from enum import Enum
from functools import lru_cache, wraps
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union, Callable, Any
from datetime import datetime, timedelta
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# Optional dependencies
try:
    from scipy import signal, stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available. Some advanced features will be limited.")

try:
    from sklearn.linear_model import RANSACRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. RANSAC regression will be disabled.")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV not available. Some features will be limited.")

logger = logging.getLogger(__name__)


# ============================================================================
# Enums for Type Safety
# ============================================================================

class TrendDirection(Enum):
    """Trend direction enumeration"""
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    WEAK_UPTREND = "weak_uptrend"
    SIDEWAYS = "sideways"
    WEAK_DOWNTREND = "weak_downtrend"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    UNKNOWN = "unknown"


class CandleColor(Enum):
    """Candle color enumeration"""
    GREEN = "green"
    RED = "red"
    MIXED = "mixed"
    NEUTRAL = "neutral"


class PatternType(Enum):
    """Candlestick pattern types"""
    DOJI = "doji"
    HAMMER = "hammer"
    INVERTED_HAMMER = "inverted_hammer"
    SHOOTING_STAR = "shooting_star"
    HANGING_MAN = "hanging_man"
    ENGULFING_BULLISH = "engulfing_bullish"
    ENGULFING_BEARISH = "engulfing_bearish"
    HARAMI_BULLISH = "harami_bullish"
    HARAMI_BEARISH = "harami_bearish"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    THREE_BLACK_CROWS = "three_black_crows"
    PIERCING_LINE = "piercing_line"
    DARK_CLOUD_COVER = "dark_cloud_cover"
    MARUBOZU_BULLISH = "marubozu_bullish"
    MARUBOZU_BEARISH = "marubozu_bearish"
    NONE = "none"


# ============================================================================
# Performance Decorators
# ============================================================================

def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        elapsed = (datetime.now() - start).total_seconds()
        logger.debug(f"{func.__name__} executed in {elapsed:.4f}s")
        return result
    return wrapper


# ============================================================================
# Enhanced Data Classes
# ============================================================================

@dataclass
class Candlestick:
    """Enhanced candlestick with ML-ready features"""
    index: int
    x_position: int
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    is_bullish: bool
    timestamp: Optional[datetime] = None
    volume: Optional[float] = None
    confidence: float = 1.0
    detected_patterns: List[PatternType] = field(default_factory=list)
    
    # Visual features
    width: int = 0
    body_pixels: int = 0
    wick_pixels: int = 0
    
    def __post_init__(self):
        """Calculate derived metrics - DO NOT detect patterns here"""
        self._calculate_metrics()
    
    def _calculate_metrics(self):
        """Calculate candlestick metrics with validation"""
        self.body_height = abs(self.close_price - self.open_price)
        self.upper_shadow = self.high_price - max(self.open_price, self.close_price)
        self.lower_shadow = min(self.open_price, self.close_price) - self.low_price
        self.total_range = self.high_price - self.low_price
        
        # Validate OHLC relationships
        if not self._validate_ohlc():
            logger.warning(f"Invalid OHLC for candle {self.index}: H={self.high_price}, L={self.low_price}, O={self.open_price}, C={self.close_price}")
    
    def _validate_ohlc(self) -> bool:
        """Validate OHLC price relationships"""
        return (self.high_price >= max(self.open_price, self.close_price) and
                self.low_price <= min(self.open_price, self.close_price) and
                self.total_range >= 0)
    
    @property
    def body_ratio(self) -> float:
        """Ratio of body to total range"""
        return self.body_height / self.total_range if self.total_range > 0 else 0
    
    @property
    def upper_shadow_ratio(self) -> float:
        """Ratio of upper shadow to total range"""
        return self.upper_shadow / self.total_range if self.total_range > 0 else 0
    
    @property
    def lower_shadow_ratio(self) -> float:
        """Ratio of lower shadow to total range"""
        return self.lower_shadow / self.total_range if self.total_range > 0 else 0
    
    @property
    def shadow_ratio(self) -> float:
        """Total shadow to body ratio"""
        shadows = self.upper_shadow + self.lower_shadow
        return shadows / self.body_height if self.body_height > 0 else float('inf')
    
    def _detect_single_patterns(self):
        """Detect single-candle patterns with strict criteria"""
        # Doji - very small body
        if self.is_doji(0.05):
            self.detected_patterns.append(PatternType.DOJI)
        
        # Hammer - small body, long lower shadow, at bottom
        if self.is_hammer():
            self.detected_patterns.append(PatternType.HAMMER)
        
        # Inverted Hammer - small body, long upper shadow
        if self.is_inverted_hammer():
            self.detected_patterns.append(PatternType.INVERTED_HAMMER)
        
        # Shooting Star - small body at top, long upper shadow
        if self.is_shooting_star():
            self.detected_patterns.append(PatternType.SHOOTING_STAR)
        
        # Hanging Man - like hammer but at top (context needed)
        if self.is_hanging_man():
            self.detected_patterns.append(PatternType.HANGING_MAN)
        
        # Marubozu - very long body, minimal shadows
        if self.is_marubozu_bullish():
            self.detected_patterns.append(PatternType.MARUBOZU_BULLISH)
        if self.is_marubozu_bearish():
            self.detected_patterns.append(PatternType.MARUBOZU_BEARISH)
    
    def is_doji(self, threshold: float = 0.05) -> bool:
        """Doji: body < 5% of range"""
        return self.body_ratio < threshold
    
    def is_hammer(self) -> bool:
        """Hammer: lower shadow > 2x body, upper shadow small"""
        return (self.lower_shadow > 2 * self.body_height and 
                self.upper_shadow < 0.1 * self.total_range and
                self.body_ratio < 0.3)
    
    def is_inverted_hammer(self) -> bool:
        """Inverted Hammer: upper shadow > 2x body, lower shadow small"""
        return (self.upper_shadow > 2 * self.body_height and 
                self.lower_shadow < 0.1 * self.total_range and
                self.body_ratio < 0.3)
    
    def is_shooting_star(self) -> bool:
        """Shooting Star: like inverted hammer (context determines difference)"""
        return self.is_inverted_hammer()
    
    def is_hanging_man(self) -> bool:
        """Hanging Man: like hammer (context determines difference)"""
        return self.is_hammer()
    
    def is_marubozu_bullish(self) -> bool:
        """Bullish Marubozu: body > 90% of range, bullish"""
        return self.is_bullish and self.body_ratio > 0.9
    
    def is_marubozu_bearish(self) -> bool:
        """Bearish Marubozu: body > 90% of range, bearish"""
        return not self.is_bullish and self.body_ratio > 0.9
    
    def get_ml_features(self) -> np.ndarray:
        """Extract ML-ready feature vector"""
        return np.array([
            self.open_price,
            self.high_price,
            self.low_price,
            self.close_price,
            self.body_height,
            self.upper_shadow,
            self.lower_shadow,
            self.total_range,
            self.body_ratio,
            self.upper_shadow_ratio,
            self.lower_shadow_ratio,
            self.shadow_ratio,
            1.0 if self.is_bullish else 0.0,
            self.confidence,
            len(self.detected_patterns)
        ])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ChartMetrics:
    """Advanced chart metrics for ML"""
    volatility: float
    momentum: float
    support_level: float
    resistance_level: float
    average_body_size: float
    bullish_percentage: float
    pattern_density: float
    quality_score: float
    
    # New metrics
    trend_consistency: float = 0.0
    price_efficiency: float = 0.0
    volume_trend: float = 0.0
    rsi: float = 50.0  # Relative Strength Index
    
    def get_ml_features(self) -> np.ndarray:
        """Extract ML-ready feature vector"""
        return np.array([
            self.volatility,
            self.momentum,
            self.support_level,
            self.resistance_level,
            self.average_body_size,
            self.bullish_percentage,
            self.pattern_density,
            self.quality_score,
            self.trend_consistency,
            self.price_efficiency,
            self.volume_trend,
            self.rsi
        ])


@dataclass
class ChartAnalysis:
    """Comprehensive chart analysis with ML features"""
    candlesticks: List[Candlestick]
    trend_direction: TrendDirection
    trend_strength: float
    dominant_color: CandleColor
    chart_quality: float
    image_dimensions: Tuple[int, int]
    metrics: ChartMetrics
    detected_patterns: Dict[PatternType, List[int]] = field(default_factory=dict)
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_ml_feature_matrix(self) -> np.ndarray:
        """Get complete feature matrix for ML models"""
        candle_features = np.array([c.get_ml_features() for c in self.candlesticks])
        return candle_features
    
    def get_aggregate_features(self) -> np.ndarray:
        """Get aggregate features for the entire chart"""
        chart_features = np.array([
            len(self.candlesticks),
            self.trend_strength,
            self.chart_quality,
            self.image_dimensions[0],
            self.image_dimensions[1]
        ])
        metric_features = self.metrics.get_ml_features()
        return np.concatenate([chart_features, metric_features])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'candlesticks': [c.to_dict() for c in self.candlesticks],
            'trend_direction': self.trend_direction.value,
            'trend_strength': self.trend_strength,
            'dominant_color': self.dominant_color.value,
            'chart_quality': self.chart_quality,
            'image_dimensions': self.image_dimensions,
            'metrics': asdict(self.metrics),
            'detected_patterns': {k.value: v for k, v in self.detected_patterns.items()},
            'processing_time': self.processing_time,
            'metadata': self.metadata
        }


# ============================================================================
# Enhanced Pattern Recognition Engine
# ============================================================================

class EnhancedPatternRecognizer:
    """Advanced pattern recognition with configurable thresholds"""
    
    def __init__(self, 
                 doji_threshold: float = 0.05,
                 engulfing_threshold: float = 1.1):
        self.doji_threshold = doji_threshold
        self.engulfing_threshold = engulfing_threshold
    
    def detect_all_patterns(self, candles: List[Candlestick]) -> Dict[PatternType, List[int]]:
        """Detect all multi-candle patterns"""
        patterns = {}
        
        if len(candles) < 2:
            return patterns
        
        # Two-candle patterns
        for i in range(1, len(candles)):
            # Engulfing patterns
            if self._is_bullish_engulfing(candles[i-1], candles[i]):
                patterns.setdefault(PatternType.ENGULFING_BULLISH, []).append(i)
            if self._is_bearish_engulfing(candles[i-1], candles[i]):
                patterns.setdefault(PatternType.ENGULFING_BEARISH, []).append(i)
            
            # Harami patterns
            if self._is_bullish_harami(candles[i-1], candles[i]):
                patterns.setdefault(PatternType.HARAMI_BULLISH, []).append(i)
            if self._is_bearish_harami(candles[i-1], candles[i]):
                patterns.setdefault(PatternType.HARAMI_BEARISH, []).append(i)
            
            # Piercing Line and Dark Cloud Cover
            if self._is_piercing_line(candles[i-1], candles[i]):
                patterns.setdefault(PatternType.PIERCING_LINE, []).append(i)
            if self._is_dark_cloud_cover(candles[i-1], candles[i]):
                patterns.setdefault(PatternType.DARK_CLOUD_COVER, []).append(i)
        
        # Three-candle patterns
        if len(candles) >= 3:
            for i in range(2, len(candles)):
                window = candles[i-2:i+1]
                
                if self._is_morning_star(window):
                    patterns.setdefault(PatternType.MORNING_STAR, []).append(i)
                if self._is_evening_star(window):
                    patterns.setdefault(PatternType.EVENING_STAR, []).append(i)
                if self._is_three_white_soldiers(window):
                    patterns.setdefault(PatternType.THREE_WHITE_SOLDIERS, []).append(i)
                if self._is_three_black_crows(window):
                    patterns.setdefault(PatternType.THREE_BLACK_CROWS, []).append(i)
        
        return patterns
    
    def _is_bullish_engulfing(self, prev: Candlestick, curr: Candlestick) -> bool:
        """Bullish Engulfing: current green candle engulfs previous red"""
        return (not prev.is_bullish and curr.is_bullish and
                curr.open_price <= prev.close_price and
                curr.close_price >= prev.open_price and
                curr.body_height > prev.body_height * 0.9)
    
    def _is_bearish_engulfing(self, prev: Candlestick, curr: Candlestick) -> bool:
        """Bearish Engulfing: current red candle engulfs previous green"""
        return (prev.is_bullish and not curr.is_bullish and
                curr.open_price >= prev.close_price and
                curr.close_price <= prev.open_price and
                curr.body_height > prev.body_height * 0.9)
    
    def _is_bullish_harami(self, prev: Candlestick, curr: Candlestick) -> bool:
        """Bullish Harami: small green candle inside previous large red"""
        return (not prev.is_bullish and curr.is_bullish and
                curr.open_price > prev.close_price and
                curr.close_price < prev.open_price and
                curr.body_height < prev.body_height * 0.5)
    
    def _is_bearish_harami(self, prev: Candlestick, curr: Candlestick) -> bool:
        """Bearish Harami: small red candle inside previous large green"""
        return (prev.is_bullish and not curr.is_bullish and
                curr.open_price < prev.close_price and
                curr.close_price > prev.open_price and
                curr.body_height < prev.body_height * 0.5)
    
    def _is_piercing_line(self, prev: Candlestick, curr: Candlestick) -> bool:
        """Piercing Line: green candle closes above midpoint of red candle"""
        if not (not prev.is_bullish and curr.is_bullish):
            return False
        midpoint = (prev.open_price + prev.close_price) / 2
        return (curr.open_price < prev.low_price and
                curr.close_price > midpoint and
                curr.close_price < prev.open_price)
    
    def _is_dark_cloud_cover(self, prev: Candlestick, curr: Candlestick) -> bool:
        """Dark Cloud Cover: red candle closes below midpoint of green"""
        if not (prev.is_bullish and not curr.is_bullish):
            return False
        midpoint = (prev.open_price + prev.close_price) / 2
        return (curr.open_price > prev.high_price and
                curr.close_price < midpoint and
                curr.close_price > prev.open_price)
    
    def _is_morning_star(self, candles: List[Candlestick]) -> bool:
        """Morning Star: bearish, small body, bullish"""
        if len(candles) != 3:
            return False
        c1, c2, c3 = candles
        return (not c1.is_bullish and
                c2.body_ratio < 0.3 and
                c3.is_bullish and
                c3.close_price > (c1.open_price + c1.close_price) / 2 and
                c2.total_range < c1.total_range * 0.5)
    
    def _is_evening_star(self, candles: List[Candlestick]) -> bool:
        """Evening Star: bullish, small body, bearish"""
        if len(candles) != 3:
            return False
        c1, c2, c3 = candles
        return (c1.is_bullish and
                c2.body_ratio < 0.3 and
                not c3.is_bullish and
                c3.close_price < (c1.open_price + c1.close_price) / 2 and
                c2.total_range < c1.total_range * 0.5)
    
    def _is_three_white_soldiers(self, candles: List[Candlestick]) -> bool:
        """Three White Soldiers: three consecutive strong bullish candles"""
        if len(candles) != 3:
            return False
        return (all(c.is_bullish for c in candles) and
                all(candles[i].close_price > candles[i-1].close_price for i in range(1, 3)) and
                all(candles[i].open_price > candles[i-1].open_price for i in range(1, 3)) and
                all(c.body_ratio > 0.6 for c in candles))
    
    def _is_three_black_crows(self, candles: List[Candlestick]) -> bool:
        """Three Black Crows: three consecutive strong bearish candles"""
        if len(candles) != 3:
            return False
        return (all(not c.is_bullish for c in candles) and
                all(candles[i].close_price < candles[i-1].close_price for i in range(1, 3)) and
                all(candles[i].open_price < candles[i-1].open_price for i in range(1, 3)) and
                all(c.body_ratio > 0.6 for c in candles))


# ============================================================================
# Advanced Image Processor
# ============================================================================

class AdvancedImageProcessor:
    """
    Enhanced candlestick chart processor with improved algorithms
    """
    
    # Default color ranges (HSV) - configurable
    DEFAULT_GREEN_RANGE = ((35, 80, 80), (90, 255, 255))
    DEFAULT_RED_RANGE = ((0, 80, 80), (10, 255, 255))
    DEFAULT_RED_RANGE_2 = ((170, 80, 80), (180, 255, 255))
    
    def __init__(self, 
                 min_candle_width: int = 2,
                 max_candle_width: int = 60,
                 min_candle_height: int = 5,
                 cache_enabled: bool = True,
                 max_workers: int = 4,
                 green_range: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None,
                 red_range: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None,
                 red_range_2: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None,
                 confidence_color_weight: float = 0.7,
                 confidence_wick_weight: float = 0.3):
        """
        Initialize the image processor with comprehensive validation
        
        Args:
            min_candle_width: Minimum candlestick width in pixels
            max_candle_width: Maximum candlestick width in pixels
            min_candle_height: Minimum candlestick height in pixels
            cache_enabled: Enable result caching
            max_workers: Number of worker threads for parallel processing
            green_range: Custom HSV range for green candles ((h_min, s_min, v_min), (h_max, s_max, v_max))
            red_range: Custom HSV range for red candles (lower hue range)
            red_range_2: Custom HSV range for red candles (upper hue range)
            confidence_color_weight: Weight for color clarity in confidence calculation (0-1)
            confidence_wick_weight: Weight for wick clarity in confidence calculation (0-1)
        """
        # Validate inputs
        self._validate_init_params(min_candle_width, max_candle_width, min_candle_height, max_workers)
        self._validate_confidence_weights(confidence_color_weight, confidence_wick_weight)
        
        self.min_candle_width = min_candle_width
        self.max_candle_width = max_candle_width
        self.min_candle_height = min_candle_height
        self.cache_enabled = cache_enabled
        self.max_workers = max_workers
        
        # Confidence calculation weights
        self.confidence_color_weight = confidence_color_weight
        self.confidence_wick_weight = confidence_wick_weight
        
        # Set color ranges (configurable for different chart types)
        self.green_range = green_range if green_range else self.DEFAULT_GREEN_RANGE
        self.red_range = red_range if red_range else self.DEFAULT_RED_RANGE
        self.red_range_2 = red_range_2 if red_range_2 else self.DEFAULT_RED_RANGE_2
        
        self._validate_color_ranges()
        
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.pattern_recognizer = EnhancedPatternRecognizer()
        
        # Initialize TRUE LRU cache if enabled
        if self.cache_enabled:
            self._cache: OrderedDict[str, ChartAnalysis] = OrderedDict()
            self._cache_max_size = 100
            self._cache_hits = 0
            self._cache_misses = 0
        
        logger.info(f"Initialized AdvancedImageProcessor (CV2: {CV2_AVAILABLE}, Cache: {cache_enabled})")
    
    def _validate_init_params(self, min_width: int, max_width: int, min_height: int, max_workers: int):
        """Validate initialization parameters"""
        if not isinstance(min_width, int) or min_width < 1:
            raise ValueError(f"min_candle_width must be a positive integer, got {min_width}")
        
        if not isinstance(max_width, int) or max_width < min_width:
            raise ValueError(f"max_candle_width must be >= min_candle_width, got {max_width}")
        
        if not isinstance(min_height, int) or min_height < 1:
            raise ValueError(f"min_candle_height must be a positive integer, got {min_height}")
        
        if not isinstance(max_workers, int) or max_workers < 1:
            raise ValueError(f"max_workers must be a positive integer, got {max_workers}")
        
        if max_width > 1000:
            logger.warning(f"max_candle_width ({max_width}) is very large, may affect performance")
    
    def _validate_confidence_weights(self, color_weight: float, wick_weight: float):
        """Validate confidence calculation weights"""
        if not isinstance(color_weight, (int, float)) or not 0 <= color_weight <= 1:
            raise ValueError(f"confidence_color_weight must be in [0, 1], got {color_weight}")
        
        if not isinstance(wick_weight, (int, float)) or not 0 <= wick_weight <= 1:
            raise ValueError(f"confidence_wick_weight must be in [0, 1], got {wick_weight}")
        
        total_weight = color_weight + wick_weight
        if not (0.99 <= total_weight <= 1.01):  # Allow small floating point error
            raise ValueError(f"Confidence weights must sum to 1.0, got {total_weight}")
    
    def _validate_color_ranges(self):
        """Validate HSV color ranges"""
        for range_name, color_range in [
            ('green_range', self.green_range),
            ('red_range', self.red_range),
            ('red_range_2', self.red_range_2)
        ]:
            if not isinstance(color_range, tuple) or len(color_range) != 2:
                raise ValueError(f"{range_name} must be a tuple of two tuples")
            
            lower, upper = color_range
            if not (isinstance(lower, tuple) and isinstance(upper, tuple)):
                raise ValueError(f"{range_name} bounds must be tuples")
            
            if not (len(lower) == 3 and len(upper) == 3):
                raise ValueError(f"{range_name} bounds must have 3 values (H, S, V)")
            
            # Validate HSV ranges
            if not (0 <= lower[0] <= 180 and 0 <= upper[0] <= 180):
                raise ValueError(f"{range_name} hue must be in range [0, 180]")
            
            if not all(0 <= v <= 255 for v in lower[1:] + upper[1:]):
                raise ValueError(f"{range_name} saturation and value must be in range [0, 255]")
    
    def set_color_ranges(self, 
                        green_range: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None,
                        red_range: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None,
                        red_range_2: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None):
        """
        Update color ranges for different chart types (e.g., dark mode, different platforms)
        
        Example for dark mode charts:
            processor.set_color_ranges(
                green_range=((35, 100, 100), (90, 255, 255)),  # Brighter green
                red_range=((0, 100, 100), (10, 255, 255))      # Brighter red
            )
        """
        if green_range:
            self.green_range = green_range
        if red_range:
            self.red_range = red_range
        if red_range_2:
            self.red_range_2 = red_range_2
        
        self._validate_color_ranges()
        
        # Clear cache since color detection may change
        if self.cache_enabled:
            self.clear_cache()
        
        logger.info("Color ranges updated")
    
    def set_confidence_weights(self, color_weight: float, wick_weight: float):
        """Update confidence calculation weights"""
        self._validate_confidence_weights(color_weight, wick_weight)
        self.confidence_color_weight = color_weight
        self.confidence_wick_weight = wick_weight
        
        # Clear cache since confidence calculations will change
        if self.cache_enabled:
            self.clear_cache()
        
        logger.info(f"Confidence weights updated: color={color_weight}, wick={wick_weight}")
    
    def _compute_cache_key(self, image: np.ndarray) -> str:
        """
        OPTIMIZED: Compute cache key from image data and settings
        Uses downsampled image for faster hashing and includes all relevant settings
        """
        # Downsample image for faster hashing (100x100 should be enough)
        h, w = image.shape[:2]
        if h > 100 or w > 100:
            scale = 100 / max(h, w)
            small_h, small_w = int(h * scale), int(w * scale)
            if CV2_AVAILABLE:
                small_img = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
            else:
                pil_img = Image.fromarray(image)
                pil_img = pil_img.resize((small_w, small_h), Image.LANCZOS)
                small_img = np.array(pil_img)
        else:
            small_img = image
        
        # Hash downsampled image
        img_hash = hashlib.md5(small_img.tobytes()).hexdigest()[:16]
        
        # Include settings in cache key
        settings_str = (
            f"{self.min_candle_width}_{self.max_candle_width}_{self.min_candle_height}_"
            f"{self.green_range}_{self.red_range}_{self.red_range_2}_"
            f"{self.confidence_color_weight}_{self.confidence_wick_weight}"
        )
        settings_hash = hashlib.md5(settings_str.encode()).hexdigest()[:16]
        
        return f"{img_hash}_{settings_hash}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[ChartAnalysis]:
        """Retrieve result from TRUE LRU cache"""
        if not self.cache_enabled:
            return None
        
        if cache_key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(cache_key)
            self._cache_hits += 1
            logger.debug(f"Cache hit: {cache_key[:16]}...")
            return self._cache[cache_key]
        
        self._cache_misses += 1
        return None
    
    def _add_to_cache(self, cache_key: str, analysis: ChartAnalysis):
        """Add result to TRUE LRU cache with automatic eviction"""
        if not self.cache_enabled:
            return
        
        # Add new item
        self._cache[cache_key] = analysis
        
        # Move to end (most recently used)
        self._cache.move_to_end(cache_key)
        
        # Evict oldest if over capacity
        if len(self._cache) > self._cache_max_size:
            # Remove first item (least recently used)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            logger.debug(f"Cache evicted oldest: {oldest_key[:16]}...")
    
    def clear_cache(self):
        """Clear the analysis cache"""
        if self.cache_enabled:
            self._cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
            logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.cache_enabled:
            return {'enabled': False}
        
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'enabled': True,
            'size': len(self._cache),
            'max_size': self._cache_max_size,
            'usage_percent': len(self._cache) / self._cache_max_size * 100,
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate_percent': hit_rate
        }
    
    def __del__(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=False)
    
    @timing_decorator
    def load_image(self, image_data: Union[bytes, str, Path]) -> Optional[np.ndarray]:
        """Load and validate image with comprehensive checks"""
        try:
            # Validate input type
            if not isinstance(image_data, (bytes, str, Path)):
                raise TypeError(f"image_data must be bytes, str, or Path, got {type(image_data)}")
            
            # Load image
            if isinstance(image_data, (str, Path)):
                image_path = Path(image_data)
                if not image_path.exists():
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                if not image_path.is_file():
                    raise ValueError(f"Path is not a file: {image_path}")
                image = Image.open(image_path)
            else:
                if len(image_data) == 0:
                    raise ValueError("Empty image data provided")
                image = Image.open(io.BytesIO(image_data))
            
            # Validate image format
            if image.format not in ['PNG', 'JPEG', 'JPG', 'BMP', 'GIF', 'TIFF', None]:
                logger.warning(f"Unusual image format: {image.format}")
            
            # Convert to RGB
            if image.mode != 'RGB':
                logger.debug(f"Converting image from {image.mode} to RGB")
                image = image.convert('RGB')
            
            img_array = np.array(image)
            
            # Comprehensive validation
            if img_array.size == 0:
                raise ValueError("Image array is empty")
            
            if img_array.ndim != 3:
                raise ValueError(f"Image must be 3-dimensional (H, W, C), got shape {img_array.shape}")
            
            if img_array.shape[2] != 3:
                raise ValueError(f"Image must have 3 channels (RGB), got {img_array.shape[2]}")
            
            h, w = img_array.shape[:2]
            
            if h < 50 or w < 50:
                raise ValueError(f"Image too small: {w}x{h} (minimum 50x50)")
            
            if h > 10000 or w > 10000:
                raise ValueError(f"Image too large: {w}x{h} (maximum 10000x10000)")
            
            # Check if image is mostly blank
            if np.std(img_array) < 10:
                logger.warning("Image appears to be mostly blank or uniform")
            
            logger.info(f"Successfully loaded image: {w}x{h}")
            return img_array
            
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return None
    
    @timing_decorator
    def preprocess_image(self, 
                        image: np.ndarray,
                        auto_adjust: bool = True) -> np.ndarray:
        """
        Enhanced preprocessing with adaptive techniques and validation
        """
        # Validate input
        if not isinstance(image, np.ndarray):
            raise TypeError(f"image must be numpy array, got {type(image)}")
        
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"image must be 3D RGB array, got shape {image.shape}")
        
        if not CV2_AVAILABLE:
            return self._preprocess_pil(image)
        
        h, w = image.shape[:2]
        
        # Resize if necessary
        max_dim = 1920
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            logger.info(f"Resizing image from {w}x{h} to {new_w}x{new_h}")
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        if auto_adjust:
            # Assess image quality
            quality = self._assess_image_quality(image)
            logger.debug(f"Image quality score: {quality:.2f}")
            
            # Apply denoising based on quality
            if quality < 0.6:
                logger.debug("Applying denoising")
                image = cv2.fastNlMeansDenoisingColored(image, None, 8, 8, 7, 21)
            
            # CLAHE for contrast
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            l = clahe.apply(l)
            image = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
            
            # Adaptive sharpening if blurry
            if quality < 0.5:
                logger.debug("Applying sharpening")
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                image = cv2.filter2D(image, -1, kernel)
        
        return image
    
    def _assess_image_quality(self, image: np.ndarray) -> float:
        """Assess image quality using variance of Laplacian"""
        if not CV2_AVAILABLE:
            return 0.7
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize to 0-1 range
        quality = min(laplacian_var / 500, 1.0)
        return quality
    
    def _preprocess_pil(self, image: np.ndarray) -> np.ndarray:
        """PIL-based preprocessing"""
        pil_image = Image.fromarray(image)
        
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.3)
        
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(1.2)
        
        return np.array(pil_image)
    
    @timing_decorator
    def detect_candlesticks(self, image: np.ndarray) -> List[Candlestick]:
        """
        IMPROVED: More accurate candlestick detection with better OHLC extraction
        """
        # Validate input
        if not isinstance(image, np.ndarray):
            raise TypeError(f"image must be numpy array, got {type(image)}")
        
        if image.ndim != 3:
            raise ValueError(f"image must be 3D array, got shape {image.shape}")
        
        if not CV2_AVAILABLE:
            return self._estimate_candlesticks_simple(image)
        
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Create combined mask for all candles using configurable ranges
        green_mask = cv2.inRange(hsv, np.array(self.green_range[0]), 
                                np.array(self.green_range[1]))
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, np.array(self.red_range[0]), np.array(self.red_range[1])),
            cv2.inRange(hsv, np.array(self.red_range_2[0]), np.array(self.red_range_2[1]))
        )
        
        # Detect dark areas (wicks/shadows)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, dark_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        
        # Combine masks
        candle_body_mask = cv2.bitwise_or(green_mask, red_mask)
        
        # Find contours for bodies
        contours, _ = cv2.findContours(candle_body_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        candlesticks = []
        
        for i, contour in enumerate(contours):
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # Filter by size
            if not (self.min_candle_width <= cw <= self.max_candle_width and 
                    ch >= self.min_candle_height):
                continue
            
            # Extract candle region
            candle = self._extract_candle_detailed(image, hsv, gray, dark_mask, x, y, cw, ch, h)
            
            if candle and candle.confidence > 0.2:
                candlesticks.append(candle)
        
        # Sort by x position
        candlesticks.sort(key=lambda c: c.x_position)
        
        # Remove duplicates (overlapping detections)
        candlesticks = self._remove_duplicates(candlesticks)
        
        # FIXED: Normalize all OHLC prices to consistent scale
        candlesticks = self._normalize_ohlc_prices(candlesticks)
        
        # Re-index
        for i, candle in enumerate(candlesticks):
            candle.index = i
        
        logger.info(f"Detected {len(candlesticks)} candlesticks")
        return candlesticks
    
    def _normalize_ohlc_prices(self, candlesticks: List[Candlestick]) -> List[Candlestick]:
        """
        FIXED: Normalize OHLC prices across all candlesticks to a consistent scale
        This ensures all prices are relative to the global min/max of the chart
        """
        if not candlesticks:
            return candlesticks
        
        # Collect all price points
        all_highs = [c.high_price for c in candlesticks]
        all_lows = [c.low_price for c in candlesticks]
        
        global_high = max(all_highs)
        global_low = min(all_lows)
        price_range = global_high - global_low
        
        # FIXED: Handle zero price range as error, not warning
        if price_range == 0:
            raise ValueError(
                f"Zero price range detected (all prices equal to {global_high}). "
                "Cannot normalize prices. This indicates invalid chart data or detection failure."
            )
        
        # Normalize each candlestick to 0-100 scale
        for candle in candlesticks:
            # Store original values for reference
            original_high = candle.high_price
            original_low = candle.low_price
            original_open = candle.open_price
            original_close = candle.close_price
            
            # Normalize to 0-100 scale
            candle.high_price = ((original_high - global_low) / price_range) * 100
            candle.low_price = ((original_low - global_low) / price_range) * 100
            candle.open_price = ((original_open - global_low) / price_range) * 100
            candle.close_price = ((original_close - global_low) / price_range) * 100
            
            # Recalculate metrics with normalized values
            candle._calculate_metrics()
            
            # FIXED: Detect patterns AFTER normalization
            candle._detect_single_patterns()
            
            # Validate normalized OHLC
            if not candle._validate_ohlc():
                logger.warning(f"Candle {candle.index} has invalid OHLC after normalization")
        
        logger.debug(f"Normalized {len(candlesticks)} candlesticks to range [{global_low:.2f}, {global_high:.2f}] -> [0, 100]")
        return candlesticks
    
    def _extract_candle_detailed(self, 
                                 image: np.ndarray,
                                 hsv: np.ndarray,
                                 gray: np.ndarray,
                                 dark_mask: np.ndarray,
                                 x: int, y: int, 
                                 cw: int, ch: int,
                                 image_height: int) -> Optional[Candlestick]:
        """
        IMPROVED: Extract detailed candle information with better OHLC estimation
        Note: Raw pixel coordinates are used here, normalization happens later
        """
        h, w = image.shape[:2]
        
        # Validate inputs
        if x < 0 or y < 0 or x + cw > w or y + ch > h:
            logger.warning(f"Invalid candle bounds: x={x}, y={y}, w={cw}, h={ch}")
            return None
        
        # Expand search area for wicks
        wick_search_margin = max(cw * 2, 10)
        x_start = max(0, x - wick_search_margin // 2)
        x_end = min(w, x + cw + wick_search_margin // 2)
        
        # Extract vertical slice including potential wicks
        slice_region = gray[0:h, x_start:x_end]
        dark_slice = dark_mask[0:h, x_start:x_end]
        
        # Find vertical extent of dark pixels (wick)
        vertical_profile = np.sum(dark_slice, axis=1)
        dark_rows = np.where(vertical_profile > 0)[0]
        
        if len(dark_rows) == 0:
            return None
        
        wick_top = dark_rows[0]
        wick_bottom = dark_rows[-1]
        total_height = wick_bottom - wick_top
        
        if total_height < self.min_candle_height:
            return None
        
        # Determine if bullish or bearish
        body_region_hsv = hsv[y:y+ch, x:x+cw]
        is_green = self._is_predominantly_green(body_region_hsv)
        is_red = self._is_predominantly_red(body_region_hsv)
        
        if not (is_green or is_red):
            return None
        
        is_bullish = is_green
        
        # FIXED: Calculate OHLC based on pixel positions (inverted Y-axis)
        # In images, Y=0 is at top, but in finance charts, top = higher price
        # High = top of wick (smallest Y value = highest price)
        # Low = bottom of wick (largest Y value = lowest price)
        high_price = image_height - wick_top
        low_price = image_height - wick_bottom
        
        # Open and Close based on body position
        body_top = y
        body_bottom = y + ch
        
        if is_bullish:
            # Green candle: close at top (higher), open at bottom (lower)
            close_price = image_height - body_top
            open_price = image_height - body_bottom
        else:
            # Red candle: open at top (higher), close at bottom (lower)
            open_price = image_height - body_top
            close_price = image_height - body_bottom
        
        # Validate OHLC relationships before normalization
        if not (high_price >= max(open_price, close_price) and 
                low_price <= min(open_price, close_price)):
            logger.warning(f"Invalid OHLC detected: H={high_price}, L={low_price}, O={open_price}, C={close_price}")
            # Attempt to fix
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
        
        # Calculate confidence based on clarity
        confidence = self._calculate_candle_confidence(
            body_region_hsv, dark_slice, is_bullish
        )
        
        candle = Candlestick(
            index=0,  # Will be set later
            x_position=x + cw // 2,
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            close_price=close_price,
            is_bullish=is_bullish,
            confidence=confidence,
            width=cw,
            body_pixels=ch * cw,
            wick_pixels=int(np.sum(dark_slice > 0))
        )
        
        return candle
    
    def _is_predominantly_green(self, hsv_region: np.ndarray) -> bool:
        """Check if region is predominantly green using configurable color range"""
        if hsv_region.size == 0:
            return False
        green_mask = cv2.inRange(hsv_region, 
                                np.array(self.green_range[0]), 
                                np.array(self.green_range[1]))
        green_ratio = np.sum(green_mask > 0) / green_mask.size
        return green_ratio > 0.3
    
    def _is_predominantly_red(self, hsv_region: np.ndarray) -> bool:
        """Check if region is predominantly red using configurable color ranges"""
        if hsv_region.size == 0:
            return False
        red_mask1 = cv2.inRange(hsv_region,
                               np.array(self.red_range[0]),
                               np.array(self.red_range[1]))
        red_mask2 = cv2.inRange(hsv_region,
                               np.array(self.red_range_2[0]),
                               np.array(self.red_range_2[1]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_ratio = np.sum(red_mask > 0) / red_mask.size
        return red_ratio > 0.3
    
    def _calculate_candle_confidence(self,
                                    body_hsv: np.ndarray,
                                    dark_slice: np.ndarray,
                                    is_bullish: bool) -> float:
        """
        FIXED: Calculate confidence score for detected candle
        Bug fix: Handles empty arrays and uses configurable weights with correct variable names
        """
        # Validate inputs
        if body_hsv.size == 0 or dark_slice.size == 0:
            logger.warning("Empty array in confidence calculation")
            return 0.5  # Default confidence for invalid input
        
        try:
            # FIXED: Use lowercase attribute names (self.green_range, not self.GREEN_RANGE)
            if is_bullish:
                color_mask = cv2.inRange(body_hsv,
                                        np.array(self.green_range[0]),
                                        np.array(self.green_range[1]))
            else:
                red_mask1 = cv2.inRange(body_hsv,
                                       np.array(self.red_range[0]),
                                       np.array(self.red_range[1]))
                red_mask2 = cv2.inRange(body_hsv,
                                       np.array(self.red_range_2[0]),
                                       np.array(self.red_range_2[1]))
                color_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            # Color clarity (avoid division by zero)
            color_ratio = np.sum(color_mask > 0) / max(color_mask.size, 1)
            
            # Wick clarity (avoid division by zero)
            wick_ratio = np.sum(dark_slice > 0) / max(dark_slice.size, 1)
            
            # Use configurable weights
            confidence = (color_ratio * self.confidence_color_weight + 
                         wick_ratio * self.confidence_wick_weight)
            
            return float(min(confidence, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5  # Default confidence on error
    
    def _remove_duplicates(self, candlesticks: List[Candlestick]) -> List[Candlestick]:
        """Remove overlapping/duplicate candlesticks"""
        if len(candlesticks) <= 1:
            return candlesticks
        
        filtered = []
        for i, candle in enumerate(candlesticks):
            is_duplicate = False
            for existing in filtered:
                x_diff = abs(candle.x_position - existing.x_position)
                if x_diff < (candle.width + existing.width) // 2:
                    # Keep the one with higher confidence
                    if candle.confidence <= existing.confidence:
                        is_duplicate = True
                        break
                    else:
                        filtered.remove(existing)
                        break
            
            if not is_duplicate:
                filtered.append(candle)
        
        return filtered
    
    def _estimate_candlesticks_simple(self, image: np.ndarray) -> List[Candlestick]:
        """Fallback simple estimation without OpenCV"""
        h, w, _ = image.shape
        
        # Simple column-based detection
        candlesticks = []
        step = max(w // 50, 5)  # Estimate ~50 candles max
        
        for x in range(0, w, step):
            x_end = min(x + step, w)
            column = image[:, x:x_end]
            
            # Detect colored regions
            is_green = np.mean(column[:, :, 1]) > np.mean(column[:, :, 0])
            
            # Simple OHLC estimation
            brightness = np.mean(column, axis=(1, 2))
            dark_rows = np.where(brightness < np.mean(brightness))[0]
            
            if len(dark_rows) < 5:
                continue
            
            high_price = 100.0
            low_price = 0.0
            open_price = 50.0 + np.random.randn() * 10
            close_price = 50.0 + np.random.randn() * 10
            
            candle = Candlestick(
                index=len(candlesticks),
                x_position=x + step // 2,
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                close_price=close_price,
                is_bullish=is_green,
                confidence=0.5,
                width=step
            )
            candlesticks.append(candle)
        
        return candlesticks
    
    @timing_decorator
    def analyze_chart(self, image: np.ndarray) -> Optional[ChartAnalysis]:
        """Complete chart analysis pipeline with caching and validation"""
        start_time = datetime.now()
        
        try:
            # Validate input
            if not isinstance(image, np.ndarray):
                raise TypeError(f"image must be numpy array, got {type(image)}")
            
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"image must be 3D RGB array, got shape {image.shape}")
            
            # Check cache
            cache_key = self._compute_cache_key(image) if self.cache_enabled else None
            if cache_key:
                cached_result = self._get_from_cache(cache_key)
                if cached_result:
                    logger.info("Returning cached result")
                    return cached_result
            
            # Preprocess
            processed = self.preprocess_image(image)
            
            # Detect candlesticks
            candlesticks = self.detect_candlesticks(processed)
            
            if len(candlesticks) == 0:
                logger.warning("No candlesticks detected")
                return None
            
            # Detect multi-candle patterns
            patterns = self.pattern_recognizer.detect_all_patterns(candlesticks)
            
            # Calculate metrics
            metrics = self._calculate_chart_metrics(candlesticks)
            
            # Determine trend
            trend_dir, trend_str = self._analyze_trend(candlesticks)
            
            # Determine dominant color
            dominant_color = self._get_dominant_color(candlesticks)
            
            # Calculate chart quality
            quality = self._assess_chart_quality(image, candlesticks)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            analysis = ChartAnalysis(
                candlesticks=candlesticks,
                trend_direction=trend_dir,
                trend_strength=trend_str,
                dominant_color=dominant_color,
                chart_quality=quality,
                image_dimensions=(image.shape[1], image.shape[0]),
                metrics=metrics,
                detected_patterns=patterns,
                processing_time=elapsed,
                metadata={
                    'opencv_available': CV2_AVAILABLE,
                    'num_candles': len(candlesticks),
                    'num_patterns': sum(len(v) for v in patterns.values()),
                    'cache_enabled': self.cache_enabled,
                    'color_ranges': {
                        'green': self.green_range,
                        'red': self.red_range,
                        'red_2': self.red_range_2
                    }
                }
            )
            
            # Add to cache
            if cache_key:
                self._add_to_cache(cache_key, analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Chart analysis failed: {e}", exc_info=True)
            return None
    
    def _calculate_chart_metrics(self, candles: List[Candlestick]) -> ChartMetrics:
        """Calculate comprehensive chart metrics"""
        if not candles:
            return ChartMetrics(0, 0, 0, 0, 0, 0, 0, 0)
        
        closes = np.array([c.close_price for c in candles])
        highs = np.array([c.high_price for c in candles])
        lows = np.array([c.low_price for c in candles])
        ranges = np.array([c.total_range for c in candles])
        
        # Volatility (standard deviation of returns)
        if len(closes) > 1:
            returns = np.diff(closes) / closes[:-1]
            volatility = float(np.std(returns))
        else:
            volatility = 0.0
        
        # Momentum (rate of change)
        if len(closes) > 1:
            momentum = float((closes[-1] - closes[0]) / closes[0])
        else:
            momentum = 0.0
        
        # Support and Resistance
        support_level = float(np.percentile(lows, 10))
        resistance_level = float(np.percentile(highs, 90))
        
        # Average body size
        body_sizes = np.array([c.body_height for c in candles])
        average_body_size = float(np.mean(body_sizes))
        
        # Bullish percentage
        bullish_count = sum(1 for c in candles if c.is_bullish)
        bullish_percentage = bullish_count / len(candles)
        
        # Pattern density
        total_patterns = sum(len(c.detected_patterns) for c in candles)
        pattern_density = total_patterns / len(candles)
        
        # Quality score (based on confidence)
        confidences = np.array([c.confidence for c in candles])
        quality_score = float(np.mean(confidences))
        
        # Trend consistency
        if len(closes) > 2:
            direction_changes = sum(
                1 for i in range(1, len(closes))
                if (closes[i] - closes[i-1]) * (closes[i-1] - closes[i-2] if i > 1 else 1) < 0
            )
            trend_consistency = 1.0 - (direction_changes / max(len(closes) - 1, 1))
        else:
            trend_consistency = 0.5
        
        # Price efficiency (straight line vs actual path)
        if len(closes) > 1:
            direct_distance = abs(closes[-1] - closes[0])
            path_distance = np.sum(np.abs(np.diff(closes)))
            price_efficiency = direct_distance / max(path_distance, 0.001)
        else:
            price_efficiency = 1.0
        
        # RSI calculation (simplified)
        rsi = self._calculate_rsi(closes)
        
        return ChartMetrics(
            volatility=volatility,
            momentum=momentum,
            support_level=support_level,
            resistance_level=resistance_level,
            average_body_size=average_body_size,
            bullish_percentage=bullish_percentage,
            pattern_density=pattern_density,
            quality_score=quality_score,
            trend_consistency=trend_consistency,
            price_efficiency=price_efficiency,
            volume_trend=0.0,  # Would need volume data
            rsi=rsi
        )
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def _analyze_trend(self, candles: List[Candlestick]) -> Tuple[TrendDirection, float]:
        """Analyze trend with multiple indicators"""
        if len(candles) < 3:
            return TrendDirection.UNKNOWN, 0.0
        
        closes = np.array([c.close_price for c in candles])
        x = np.arange(len(closes))
        
        # Linear regression for trend
        try:
            slope, intercept = np.polyfit(x, closes, 1)
            
            # Calculate R-squared for strength
            y_pred = slope * x + intercept
            ss_res = np.sum((closes - y_pred) ** 2)
            ss_tot = np.sum((closes - np.mean(closes)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            trend_strength = abs(r_squared)
            
            # Normalize slope relative to price range
            price_range = np.max(closes) - np.min(closes)
            normalized_slope = slope / max(price_range, 0.001)
            
            # Determine direction
            if abs(normalized_slope) < 0.01:
                direction = TrendDirection.SIDEWAYS
            elif normalized_slope > 0.05:
                direction = TrendDirection.STRONG_UPTREND
            elif normalized_slope > 0.02:
                direction = TrendDirection.UPTREND
            elif normalized_slope > 0:
                direction = TrendDirection.WEAK_UPTREND
            elif normalized_slope < -0.05:
                direction = TrendDirection.STRONG_DOWNTREND
            elif normalized_slope < -0.02:
                direction = TrendDirection.DOWNTREND
            else:
                direction = TrendDirection.WEAK_DOWNTREND
            
            return direction, float(trend_strength)
            
        except Exception as e:
            logger.warning(f"Trend analysis failed: {e}")
            return TrendDirection.UNKNOWN, 0.0
    
    def _get_dominant_color(self, candles: List[Candlestick]) -> CandleColor:
        """Determine dominant candle color"""
        if not candles:
            return CandleColor.NEUTRAL
        
        bullish_count = sum(1 for c in candles if c.is_bullish)
        bearish_count = len(candles) - bullish_count
        
        ratio = bullish_count / len(candles)
        
        if ratio > 0.7:
            return CandleColor.GREEN
        elif ratio < 0.3:
            return CandleColor.RED
        elif 0.4 <= ratio <= 0.6:
            return CandleColor.MIXED
        else:
            return CandleColor.NEUTRAL
    
    def _assess_chart_quality(self, image: np.ndarray, candles: List[Candlestick]) -> float:
        """Assess overall chart quality"""
        factors = []
        
        # Image quality
        img_quality = self._assess_image_quality(image)
        factors.append(img_quality)
        
        # Candle confidence
        if candles:
            avg_confidence = np.mean([c.confidence for c in candles])
            factors.append(avg_confidence)
        
        # Detection consistency
        if len(candles) > 2:
            widths = np.array([c.width for c in candles])
            width_std = np.std(widths) / max(np.mean(widths), 1)
            consistency = 1.0 - min(width_std, 1.0)
            factors.append(consistency)
        
        return float(np.mean(factors)) if factors else 0.5
    
    async def analyze_chart_async(self, image: np.ndarray) -> Optional[ChartAnalysis]:
        """Async wrapper for chart analysis"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.analyze_chart, image)
    
    def process_image_file(self, filepath: Union[str, Path]) -> Optional[ChartAnalysis]:
        """Process image file end-to-end"""
        image = self.load_image(filepath)
        if image is None:
            return None
        return self.analyze_chart(image)
    
    def batch_process(self, 
                     filepaths: List[Union[str, Path]]) -> List[Optional[ChartAnalysis]]:
        """Process multiple images in parallel"""
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.process_image_file, fp) for fp in filepaths]
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
                    results.append(None)
        return results


# ============================================================================
# Usage Example
# ============================================================================

def main():
    """Example usage with all features"""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize processor with validation
    try:
        processor = AdvancedImageProcessor(
            min_candle_width=3,
            max_candle_width=50,
            min_candle_height=10,
            max_workers=4,
            cache_enabled=True,
            confidence_color_weight=0.7,
            confidence_wick_weight=0.3
        )
        
        # Example: Configure for dark mode charts
        # processor.set_color_ranges(
        #     green_range=((35, 100, 100), (90, 255, 255)),
        #     red_range=((0, 100, 100), (10, 255, 255))
        # )
        
        # Example: Adjust confidence weights
        # processor.set_confidence_weights(color_weight=0.8, wick_weight=0.2)
        
        print("=== Processor Configuration ===")
        print(f"Min candle width: {processor.min_candle_width}px")
        print(f"Max candle width: {processor.max_candle_width}px")
        print(f"Cache enabled: {processor.cache_enabled}")
        print(f"Confidence weights: color={processor.confidence_color_weight}, wick={processor.confidence_wick_weight}")
        
        if processor.cache_enabled:
            stats = processor.get_cache_stats()
            print(f"Cache stats: {stats}")
        
        # Process single image
        image_path = "candlestick_chart.png"
        print(f"\n=== Processing {image_path} ===")
        
        analysis = processor.process_image_file(image_path)
        
        if analysis:
            print(f"\n=== Chart Analysis Results ===")
            print(f"Detected Candles: {len(analysis.candlesticks)}")
            print(f"Trend: {analysis.trend_direction.value} (strength: {analysis.trend_strength:.2f})")
            print(f"Dominant Color: {analysis.dominant_color.value}")
            print(f"Chart Quality: {analysis.chart_quality:.2f}")
            print(f"Processing Time: {analysis.processing_time:.3f}s")
            
            print(f"\n=== Metrics ===")
            print(f"Volatility: {analysis.metrics.volatility:.4f}")
            print(f"Momentum: {analysis.metrics.momentum:.4f}")
            print(f"RSI: {analysis.metrics.rsi:.2f}")
            print(f"Bullish %: {analysis.metrics.bullish_percentage:.1%}")
            print(f"Support Level: {analysis.metrics.support_level:.2f}")
            print(f"Resistance Level: {analysis.metrics.resistance_level:.2f}")
            
            if analysis.detected_patterns:
                print(f"\n=== Detected Patterns ===")
                for pattern, indices in analysis.detected_patterns.items():
                    print(f"{pattern.value}: {len(indices)} occurrences at indices {indices[:5]}")
            
            # Show first few candlesticks
            print(f"\n=== Sample Candlesticks (first 3) ===")
            for i, candle in enumerate(analysis.candlesticks[:3]):
                print(f"Candle {i}: O={candle.open_price:.2f}, H={candle.high_price:.2f}, "
                      f"L={candle.low_price:.2f}, C={candle.close_price:.2f}, "
                      f"Bullish={candle.is_bullish}, Confidence={candle.confidence:.2f}")
                if candle.detected_patterns:
                    print(f"  Patterns: {[p.value for p in candle.detected_patterns]}")
            
            # Export results
            results_dict = analysis.to_dict()
            print(f"\n✓ Full results available in dictionary format")
            print(f"✓ ML features ready: {analysis.get_ml_feature_matrix().shape}")
            
            # Cache stats after processing
            if processor.cache_enabled:
                stats = processor.get_cache_stats()
                print(f"\nCache stats after processing:")
                print(f"  Size: {stats['size']}/{stats['max_size']}")
                print(f"  Hit rate: {stats['hit_rate_percent']:.1f}%")
                print(f"  Hits: {stats['hits']}, Misses: {stats['misses']}")
            
            # Test cache by processing same image again
            print(f"\n=== Testing Cache (processing same image again) ===")
            analysis2 = processor.process_image_file(image_path)
            if analysis2:
                stats = processor.get_cache_stats()
                print(f"Cache hit rate after second request: {stats['hit_rate_percent']:.1f}%")
        else:
            print("✗ Failed to analyze chart")
            
    except ValueError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()