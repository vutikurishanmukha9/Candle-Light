"""
ML Constants and Thresholds

Centralized configuration for all machine learning detection thresholds
and magic numbers used throughout the pattern detection system.
"""

from typing import Tuple


# ============================================================================
# Candlestick Pattern Thresholds
# ============================================================================

# Body ratio thresholds (body_height / total_range)
DOJI_BODY_RATIO = 0.1          # Maximum body ratio for doji detection
SMALL_BODY_RATIO = 0.25        # Maximum for considered "small body"
LARGE_BODY_RATIO = 0.7         # Minimum for considered "large body"
MARUBOZU_BODY_RATIO = 0.9      # Minimum for marubozu (no shadows)

# Shadow ratio thresholds (shadow / total_range)
LONG_SHADOW_RATIO = 0.6        # Minimum for "long shadow"
SHORT_SHADOW_RATIO = 0.1       # Maximum for "short/no shadow"
HAMMER_LOWER_SHADOW = 0.6      # Minimum lower shadow for hammer
HAMMER_UPPER_SHADOW = 0.15     # Maximum upper shadow for hammer

# ============================================================================
# Pattern Detection Thresholds
# ============================================================================

# Engulfing patterns
ENGULFING_MIN_RATIO = 1.1      # Minimum ratio for engulfing (body2 / body1)

# Gap detection
GAP_THRESHOLD_PERCENT = 0.5    # Minimum gap size as percentage of candle range

# Star patterns (morning/evening star)
STAR_BODY_MAX_RATIO = 0.25     # Maximum body ratio for the "star" candle
STAR_GAP_PERCENT = 0.003       # Minimum gap percentage for star patterns

# Three soldiers/crows
THREE_CANDLE_MIN_BODY = 0.5    # Minimum body ratio for each candle

# ============================================================================
# Volume Analysis
# ============================================================================

VOLUME_SURGE_RATIO = 1.5       # Volume considered "high" if > avg * this
VOLUME_LOW_RATIO = 0.5         # Volume considered "low" if < avg * this
VOLUME_OUTLIER_FACTOR = 1.5    # IQR multiplier for outlier detection

# ============================================================================
# Support/Resistance Detection
# ============================================================================

SUPPORT_RESISTANCE_TOLERANCE = 0.02   # Price tolerance for level clustering (2%)
MIN_TOUCHES_FOR_LEVEL = 2             # Minimum touches to confirm S/R level
LEVEL_STRENGTH_DIVISOR = 5.0          # Divisor for calculating level strength

# ============================================================================
# Trend Analysis
# ============================================================================

UPTREND_THRESHOLD = 0.02       # Price change percentage for uptrend
DOWNTREND_THRESHOLD = 0.02     # Price change percentage for downtrend
TREND_WINDOW_SIZE = 5          # Number of candles for trend calculation

# ============================================================================
# Quality and Confidence
# ============================================================================

HIGH_CONFIDENCE_THRESHOLD = 0.8      # Threshold for "high confidence" patterns
MIN_CONFIDENCE_THRESHOLD = 0.5       # Minimum confidence to report pattern
MAX_CONFIDENCE_CAP = 0.95            # Maximum confidence (never 100%)

# Confidence boost/reduction factors
VOLUME_CONFIRMATION_BOOST = 0.2      # Added when volume confirms
SUPPORT_RESISTANCE_BOOST = 0.15      # Added when at S/R level
TREND_ALIGNMENT_BOOST = 0.1          # Added when aligned with trend

# ============================================================================
# Image Processing
# ============================================================================

# Candle detection dimensions
MIN_CANDLE_WIDTH = 2           # Minimum pixel width
MAX_CANDLE_WIDTH = 80          # Maximum pixel width
MIN_CANDLE_HEIGHT = 5          # Minimum pixel height
MAX_CANDLES = 200              # Maximum candles to detect

# Image preprocessing
MAX_IMAGE_DIMENSION = 2000     # Maximum width/height before resize
DENOISE_STRENGTH = 10          # OpenCV denoising strength
CLAHE_CLIP_LIMIT = 3.0         # CLAHE contrast limit

# HSV Color ranges for candlestick detection
GREEN_HSV_RANGE: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = (
    (35, 80, 80), (85, 255, 255)
)
RED_HSV_RANGE: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = (
    (0, 80, 80), (10, 255, 255)
)
RED_HSV_RANGE_2: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = (
    (170, 80, 80), (180, 255, 255)  # Red wraps around in HSV
)

# ============================================================================
# Complex Pattern Detection
# ============================================================================

# Double top/bottom
DOUBLE_PATTERN_TOLERANCE = 0.015     # Price tolerance (1.5%)
DOUBLE_PATTERN_MIN_SPACING = 5       # Minimum candles between peaks
DOUBLE_PATTERN_MAX_SPACING = 25      # Maximum candles between peaks
DOUBLE_PATTERN_MIN_DEPTH = 0.02      # Minimum retracement (2%)

# Head and shoulders
HS_SHOULDER_TOLERANCE = 0.03         # Shoulder price tolerance (3%)
HS_HEAD_MIN_HEIGHT = 0.05            # Head must be 5% higher than shoulders
HS_MAX_PATTERN_WIDTH = 30            # Maximum candles for pattern

# Wedge patterns
WEDGE_MIN_CANDLES = 10               # Minimum candles for wedge
WEDGE_CONVERGENCE_RATE = 0.7         # Rate of trendline convergence

# ============================================================================
# Analysis Limits
# ============================================================================

MAX_PATTERNS_RETURNED = 10           # Maximum patterns in result
MAX_SUPPORT_LEVELS = 5               # Maximum support levels to return
MAX_RESISTANCE_LEVELS = 5            # Maximum resistance levels to return


# ============================================================================
# Asset Class Profiles (Dynamic Thresholds)
# ============================================================================
# Different assets have different volatility profiles. These profiles
# adjust detection thresholds based on asset characteristics.

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class AssetClass(Enum):
    """Asset class enumeration for dynamic threshold selection"""
    STOCKS = "stocks"          # Equities, indices
    FOREX = "forex"            # Currency pairs (low volatility)
    CRYPTO = "crypto"          # Cryptocurrencies (high volatility)
    COMMODITIES = "commodities"  # Gold, Oil, etc.
    DEFAULT = "default"        # Fallback


@dataclass(frozen=True)
class AssetProfile:
    """
    Profile containing adjusted thresholds for an asset class.
    
    Higher volatility assets need larger thresholds to filter noise.
    Lower volatility assets need tighter thresholds for sensitivity.
    """
    # Body ratio multiplier (applied to DOJI_BODY_RATIO, etc.)
    body_ratio_multiplier: float
    
    # Shadow ratio multiplier
    shadow_ratio_multiplier: float
    
    # Price tolerance multiplier (for S/R, double patterns)
    price_tolerance_multiplier: float
    
    # Volume surge multiplier
    volume_surge_multiplier: float
    
    # Minimum confidence adjustment (added to base)
    confidence_adjustment: float
    
    # Description
    description: str


# Asset class profiles with calibrated multipliers
ASSET_PROFILES: Dict[AssetClass, AssetProfile] = {
    AssetClass.DEFAULT: AssetProfile(
        body_ratio_multiplier=1.0,
        shadow_ratio_multiplier=1.0,
        price_tolerance_multiplier=1.0,
        volume_surge_multiplier=1.0,
        confidence_adjustment=0.0,
        description="Default profile for general use"
    ),
    
    AssetClass.FOREX: AssetProfile(
        body_ratio_multiplier=0.8,      # Tighter thresholds (less volatile)
        shadow_ratio_multiplier=0.85,
        price_tolerance_multiplier=0.5,  # Much tighter (e.g., 0.75% vs 1.5%)
        volume_surge_multiplier=0.9,
        confidence_adjustment=0.05,      # Slightly boost confidence
        description="Forex pairs - low volatility, tight ranges"
    ),
    
    AssetClass.CRYPTO: AssetProfile(
        body_ratio_multiplier=1.5,       # Much looser (high volatility)
        shadow_ratio_multiplier=1.3,
        price_tolerance_multiplier=2.0,  # Double tolerance (e.g., 3% vs 1.5%)
        volume_surge_multiplier=1.5,
        confidence_adjustment=-0.1,      # More conservative (volatile = less certain)
        description="Cryptocurrencies - high volatility, wide ranges"
    ),
    
    AssetClass.STOCKS: AssetProfile(
        body_ratio_multiplier=1.0,
        shadow_ratio_multiplier=1.0,
        price_tolerance_multiplier=1.2,  # Slightly looser
        volume_surge_multiplier=1.2,
        confidence_adjustment=0.0,
        description="Stocks and indices - moderate volatility"
    ),
    
    AssetClass.COMMODITIES: AssetProfile(
        body_ratio_multiplier=1.2,
        shadow_ratio_multiplier=1.1,
        price_tolerance_multiplier=1.3,
        volume_surge_multiplier=1.3,
        confidence_adjustment=-0.05,
        description="Commodities - can be volatile with trends"
    ),
}


def get_adjusted_threshold(
    base_value: float,
    multiplier_type: str,
    asset_class: AssetClass = AssetClass.DEFAULT
) -> float:
    """
    Get threshold adjusted for asset class.
    
    Args:
        base_value: The base threshold value (e.g., DOJI_BODY_RATIO)
        multiplier_type: One of 'body', 'shadow', 'price', 'volume'
        asset_class: The asset class to adjust for
        
    Returns:
        Adjusted threshold value
        
    Example:
        # For crypto, doji body ratio becomes 0.1 * 1.5 = 0.15
        adjusted = get_adjusted_threshold(DOJI_BODY_RATIO, 'body', AssetClass.CRYPTO)
    """
    profile = ASSET_PROFILES.get(asset_class, ASSET_PROFILES[AssetClass.DEFAULT])
    
    multipliers = {
        'body': profile.body_ratio_multiplier,
        'shadow': profile.shadow_ratio_multiplier,
        'price': profile.price_tolerance_multiplier,
        'volume': profile.volume_surge_multiplier,
    }
    
    multiplier = multipliers.get(multiplier_type, 1.0)
    return base_value * multiplier


def get_confidence_adjustment(asset_class: AssetClass = AssetClass.DEFAULT) -> float:
    """Get confidence adjustment for an asset class."""
    profile = ASSET_PROFILES.get(asset_class, ASSET_PROFILES[AssetClass.DEFAULT])
    return profile.confidence_adjustment

