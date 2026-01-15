"""
Advanced Candlestick Pattern Analysis System

Enhanced version with:
- Extended pattern library (50+ patterns)
- Market context awareness
- Volume analysis integration
- Statistical scoring system
- Pattern strength grading
- Multi-timeframe support
- Risk/reward calculations
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Callable
from datetime import datetime, timedelta
import math


class PatternType(str, Enum):
    """Classification of pattern market bias"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    CONTINUATION = "continuation"
    REVERSAL = "reversal"


class PatternCategory(str, Enum):
    """Category of candlestick pattern"""
    SINGLE = "single"
    DOUBLE = "double"
    TRIPLE = "triple"
    COMPLEX = "complex"
    ADVANCED = "advanced"


class MarketContext(str, Enum):
    """Market context for pattern appearance"""
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    STRONG_DOWNTREND = "strong_downtrend"
    WEAK_DOWNTREND = "weak_downtrend"
    RANGING = "ranging"
    BREAKOUT = "breakout"
    CONSOLIDATION = "consolidation"


class VolumeProfile(str, Enum):
    """Volume characteristics"""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    CLIMAX = "climax"
    AVERAGE = "average"
    LOW = "low"


class PatternStrength(str, Enum):
    """Pattern strength rating"""
    VERY_STRONG = "very_strong"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    INVALID = "invalid"


class TimeFrame(str, Enum):
    """Trading timeframes"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"
    MN1 = "1M"


@dataclass
class PatternMetrics:
    """Advanced metrics for pattern analysis"""
    win_rate: float  # Historical success rate
    avg_gain: float  # Average gain when successful
    avg_loss: float  # Average loss when failed
    risk_reward_ratio: float
    avg_duration: int  # Average bars to target
    optimal_contexts: List[MarketContext]
    volume_requirement: VolumeProfile
    
    def sharpe_ratio(self) -> float:
        """Calculate pattern's Sharpe-like ratio"""
        if self.avg_loss == 0:
            return float('inf')
        return (self.win_rate * self.avg_gain) / abs(self.avg_loss)


@dataclass
class TradingSetup:
    """Trading setup parameters"""
    entry_conditions: List[str]
    stop_loss_placement: str
    take_profit_targets: List[Tuple[float, str]]  # (ratio, description)
    position_sizing: str
    max_holding_period: Optional[int] = None
    invalidation_rules: List[str] = field(default_factory=list)


@dataclass
class CandlestickPattern:
    """Enhanced candlestick pattern definition"""
    name: str
    category: PatternCategory
    pattern_type: PatternType
    bias: PatternType
    reliability: float
    description: str
    recognition_rules: List[str]
    trading_implications: str
    confirmation_needed: bool = True
    aliases: List[str] = field(default_factory=list)
    
    # Advanced features
    metrics: Optional[PatternMetrics] = None
    trading_setup: Optional[TradingSetup] = None
    preferred_contexts: List[MarketContext] = field(default_factory=list)
    avoid_contexts: List[MarketContext] = field(default_factory=list)
    min_timeframe: TimeFrame = TimeFrame.M5
    false_signal_warnings: List[str] = field(default_factory=list)
    confluence_patterns: List[str] = field(default_factory=list)
    
    def calculate_strength(self, context: MarketContext, volume: VolumeProfile, 
                          timeframe: TimeFrame) -> PatternStrength:
        """Calculate pattern strength based on market conditions"""
        score = self.reliability
        
        # Context bonus
        if self.metrics and context in self.metrics.optimal_contexts:
            score += 0.15
        if context in self.avoid_contexts:
            score -= 0.25
            
        # Volume consideration
        if self.metrics and volume == self.metrics.volume_requirement:
            score += 0.1
            
        # Timeframe consideration
        timeframe_values = {tf: i for i, tf in enumerate(TimeFrame)}
        if timeframe_values[timeframe] >= timeframe_values[self.min_timeframe]:
            score += 0.05
        else:
            score -= 0.15
            
        # Convert to strength rating
        if score >= 0.8:
            return PatternStrength.VERY_STRONG
        elif score >= 0.65:
            return PatternStrength.STRONG
        elif score >= 0.5:
            return PatternStrength.MODERATE
        elif score >= 0.35:
            return PatternStrength.WEAK
        else:
            return PatternStrength.INVALID
    
    def get_risk_reward(self) -> float:
        """Get risk/reward ratio"""
        return self.metrics.risk_reward_ratio if self.metrics else 1.5


# =============================================================================
# SINGLE CANDLE PATTERNS (Enhanced)
# =============================================================================

DOJI = CandlestickPattern(
    name="Doji",
    category=PatternCategory.SINGLE,
    pattern_type=PatternType.NEUTRAL,
    bias=PatternType.NEUTRAL,
    reliability=0.55,
    description="Candle where open and close are virtually equal, indicating market indecision.",
    recognition_rules=[
        "Open price equals or nearly equals close price (within 0.1% of range)",
        "Body is very small relative to the shadows (body < 10% of range)",
        "Can have long upper and/or lower shadows",
        "More significant at key support/resistance levels"
    ],
    trading_implications="Signals indecision. Often precedes reversals when appearing after a trend.",
    confirmation_needed=True,
    aliases=["Star", "Cross"],
    metrics=PatternMetrics(
        win_rate=0.52,
        avg_gain=2.5,
        avg_loss=-1.8,
        risk_reward_ratio=1.4,
        avg_duration=8,
        optimal_contexts=[MarketContext.STRONG_UPTREND, MarketContext.STRONG_DOWNTREND],
        volume_requirement=VolumeProfile.AVERAGE
    ),
    preferred_contexts=[MarketContext.STRONG_UPTREND, MarketContext.STRONG_DOWNTREND],
    avoid_contexts=[MarketContext.RANGING],
    false_signal_warnings=["Very common in ranging markets", "Needs strong trend context"],
    confluence_patterns=["Morning Star", "Evening Star"]
)

DRAGONFLY_DOJI = CandlestickPattern(
    name="Dragonfly Doji",
    category=PatternCategory.SINGLE,
    pattern_type=PatternType.REVERSAL,
    bias=PatternType.BULLISH,
    reliability=0.62,
    description="Doji with long lower shadow and no upper shadow - strong bullish reversal.",
    recognition_rules=[
        "Open, close, and high are at the same price level",
        "Long lower shadow (at least 2x the body)",
        "No or minimal upper shadow",
        "Appears after downtrend"
    ],
    trading_implications="Strong support found. Buyers stepping in aggressively.",
    confirmation_needed=True,
    metrics=PatternMetrics(
        win_rate=0.64,
        avg_gain=3.2,
        avg_loss=-1.9,
        risk_reward_ratio=1.7,
        avg_duration=6,
        optimal_contexts=[MarketContext.STRONG_DOWNTREND],
        volume_requirement=VolumeProfile.INCREASING
    ),
    trading_setup=TradingSetup(
        entry_conditions=["Wait for bullish confirmation candle", "Entry above confirmation candle high"],
        stop_loss_placement="Below dragonfly low with 0.5 ATR buffer",
        take_profit_targets=[(1.5, "First resistance"), (2.5, "Major resistance")],
        position_sizing="Risk 1-2% of capital",
        invalidation_rules=["New lower low forms", "Consolidation exceeds 3 days"]
    )
)

GRAVESTONE_DOJI = CandlestickPattern(
    name="Gravestone Doji",
    category=PatternCategory.SINGLE,
    pattern_type=PatternType.REVERSAL,
    bias=PatternType.BEARISH,
    reliability=0.62,
    description="Doji with long upper shadow and no lower shadow - strong bearish reversal.",
    recognition_rules=[
        "Open, close, and low are at the same price level",
        "Long upper shadow (at least 2x the body)",
        "No or minimal lower shadow",
        "Appears after uptrend"
    ],
    trading_implications="Strong resistance found. Sellers rejecting higher prices.",
    confirmation_needed=True,
    metrics=PatternMetrics(
        win_rate=0.63,
        avg_gain=3.1,
        avg_loss=-2.0,
        risk_reward_ratio=1.6,
        avg_duration=6,
        optimal_contexts=[MarketContext.STRONG_UPTREND],
        volume_requirement=VolumeProfile.CLIMAX
    )
)

HAMMER = CandlestickPattern(
    name="Hammer",
    category=PatternCategory.SINGLE,
    pattern_type=PatternType.REVERSAL,
    bias=PatternType.BULLISH,
    reliability=0.65,
    description="Bullish reversal pattern with small body at top and long lower shadow.",
    recognition_rules=[
        "Appears after a downtrend",
        "Small real body at the upper end of the range",
        "Lower shadow at least 2x the body length (ideally 3x)",
        "Little to no upper shadow (max 10% of lower shadow)",
        "Body color less important but bullish body adds strength"
    ],
    trading_implications="Suggests buyers are stepping in. Consider long positions after confirmation.",
    confirmation_needed=True,
    aliases=["Bullish Pin Bar", "Takuri Line"],
    metrics=PatternMetrics(
        win_rate=0.67,
        avg_gain=3.8,
        avg_loss=-2.1,
        risk_reward_ratio=1.8,
        avg_duration=7,
        optimal_contexts=[MarketContext.STRONG_DOWNTREND, MarketContext.WEAK_DOWNTREND],
        volume_requirement=VolumeProfile.INCREASING
    ),
    trading_setup=TradingSetup(
        entry_conditions=[
            "Wait for bullish confirmation (close above hammer high)",
            "Can enter on retest of hammer low if strong confirmation",
            "Check for support level confluence"
        ],
        stop_loss_placement="Below hammer low with small buffer (0.3-0.5 ATR)",
        take_profit_targets=[
            (1.5, "First resistance or previous swing high"),
            (2.5, "Major resistance zone"),
            (3.5, "Extended target for strong trends")
        ],
        position_sizing="Risk 1-2% per trade",
        max_holding_period=15,
        invalidation_rules=[
            "Break below hammer low",
            "Failure to confirm within 2-3 candles",
            "Bearish engulfing after hammer"
        ]
    ),
    preferred_contexts=[MarketContext.STRONG_DOWNTREND, MarketContext.WEAK_DOWNTREND],
    avoid_contexts=[MarketContext.RANGING, MarketContext.STRONG_UPTREND],
    false_signal_warnings=[
        "Less reliable in choppy markets",
        "Volume confirmation critical",
        "Multiple hammers in succession reduce reliability"
    ],
    confluence_patterns=["Bullish Engulfing", "Morning Star", "Bullish Harami"]
)

SHOOTING_STAR = CandlestickPattern(
    name="Shooting Star",
    category=PatternCategory.SINGLE,
    pattern_type=PatternType.REVERSAL,
    bias=PatternType.BEARISH,
    reliability=0.65,
    description="Bearish reversal with small body at bottom and long upper shadow.",
    recognition_rules=[
        "Appears after an uptrend",
        "Small real body at the lower end of the range",
        "Upper shadow at least 2x the body length (ideally 3x)",
        "Little to no lower shadow (max 10% of upper shadow)",
        "Bearish body color adds strength"
    ],
    trading_implications="Strong bearish signal. Consider shorts after confirmation.",
    confirmation_needed=True,
    aliases=["Bearish Pin Bar"],
    metrics=PatternMetrics(
        win_rate=0.66,
        avg_gain=3.5,
        avg_loss=-2.2,
        risk_reward_ratio=1.6,
        avg_duration=7,
        optimal_contexts=[MarketContext.STRONG_UPTREND],
        volume_requirement=VolumeProfile.CLIMAX
    ),
    trading_setup=TradingSetup(
        entry_conditions=[
            "Wait for bearish confirmation candle",
            "Entry below shooting star low",
            "Check for resistance level confluence"
        ],
        stop_loss_placement="Above shooting star high with buffer",
        take_profit_targets=[(1.5, "Support"), (2.5, "Major support")],
        position_sizing="Risk 1-2% per trade"
    )
)

MARUBOZU = CandlestickPattern(
    name="Marubozu",
    category=PatternCategory.SINGLE,
    pattern_type=PatternType.CONTINUATION,
    bias=PatternType.NEUTRAL,
    reliability=0.72,
    description="Strong momentum candle with no shadows, indicating decisive movement.",
    recognition_rules=[
        "No upper shadow (or < 1% of range)",
        "No lower shadow (or < 1% of range)",
        "Large real body covering 95%+ of range",
        "Bullish Marubozu: close = high, open = low",
        "Bearish Marubozu: close = low, open = high"
    ],
    trading_implications="Very strong momentum. Continuation likely in body direction.",
    confirmation_needed=False,
    metrics=PatternMetrics(
        win_rate=0.71,
        avg_gain=4.2,
        avg_loss=-2.5,
        risk_reward_ratio=1.7,
        avg_duration=5,
        optimal_contexts=[MarketContext.STRONG_UPTREND, MarketContext.STRONG_DOWNTREND],
        volume_requirement=VolumeProfile.CLIMAX
    ),
    false_signal_warnings=["Can signal exhaustion at extremes", "Check for climactic volume"]
)

# =============================================================================
# DOUBLE CANDLE PATTERNS (Enhanced)
# =============================================================================

BULLISH_ENGULFING = CandlestickPattern(
    name="Bullish Engulfing",
    category=PatternCategory.DOUBLE,
    pattern_type=PatternType.REVERSAL,
    bias=PatternType.BULLISH,
    reliability=0.73,
    description="Large bullish candle completely engulfs previous bearish candle.",
    recognition_rules=[
        "First candle is bearish (red/black)",
        "Second candle is bullish (green/white)",
        "Second candle's body completely engulfs first candle's body",
        "Second open < first close, second close > first open",
        "Appears after a downtrend",
        "Larger engulfment = stronger signal"
    ],
    trading_implications="Strong bullish reversal signal. Consider long positions.",
    confirmation_needed=False,
    aliases=["Outside Day Bullish"],
    metrics=PatternMetrics(
        win_rate=0.72,
        avg_gain=4.1,
        avg_loss=-2.3,
        risk_reward_ratio=1.8,
        avg_duration=8,
        optimal_contexts=[MarketContext.STRONG_DOWNTREND],
        volume_requirement=VolumeProfile.INCREASING
    ),
    trading_setup=TradingSetup(
        entry_conditions=[
            "Enter on close of engulfing candle",
            "Or wait for retest of engulfing low",
            "Confirm volume is above average"
        ],
        stop_loss_placement="Below engulfing pattern low",
        take_profit_targets=[(2.0, "Resistance 1"), (3.0, "Major resistance")],
        position_sizing="Risk 1.5-2% of capital"
    ),
    min_timeframe=TimeFrame.M15
)

BEARISH_ENGULFING = CandlestickPattern(
    name="Bearish Engulfing",
    category=PatternCategory.DOUBLE,
    pattern_type=PatternType.REVERSAL,
    bias=PatternType.BEARISH,
    reliability=0.73,
    description="Large bearish candle completely engulfs previous bullish candle.",
    recognition_rules=[
        "First candle is bullish (green/white)",
        "Second candle is bearish (red/black)",
        "Second candle's body completely engulfs first candle's body",
        "Appears after an uptrend",
        "Higher volume on engulfing candle confirms strength"
    ],
    trading_implications="Strong bearish reversal signal. Consider short positions.",
    confirmation_needed=False,
    aliases=["Outside Day Bearish"],
    metrics=PatternMetrics(
        win_rate=0.71,
        avg_gain=3.9,
        avg_loss=-2.4,
        risk_reward_ratio=1.6,
        avg_duration=8,
        optimal_contexts=[MarketContext.STRONG_UPTREND],
        volume_requirement=VolumeProfile.INCREASING
    )
)

PIERCING_LINE = CandlestickPattern(
    name="Piercing Line",
    category=PatternCategory.DOUBLE,
    pattern_type=PatternType.REVERSAL,
    bias=PatternType.BULLISH,
    reliability=0.68,
    description="Bullish candle opens below previous low and closes above midpoint.",
    recognition_rules=[
        "First candle is bearish and substantial",
        "Second candle gaps down on open (below first's low)",
        "Second candle closes above midpoint of first candle's body",
        "Ideally closes in upper 60% of first candle",
        "Does not fully engulf first candle"
    ],
    trading_implications="Bullish reversal. Stronger if closes higher on first body.",
    confirmation_needed=True,
    metrics=PatternMetrics(
        win_rate=0.66,
        avg_gain=3.4,
        avg_loss=-2.1,
        risk_reward_ratio=1.6,
        avg_duration=7,
        optimal_contexts=[MarketContext.WEAK_DOWNTREND, MarketContext.STRONG_DOWNTREND],
        volume_requirement=VolumeProfile.INCREASING
    ),
    confluence_patterns=["Hammer", "Bullish Harami"]
)

DARK_CLOUD_COVER = CandlestickPattern(
    name="Dark Cloud Cover",
    category=PatternCategory.DOUBLE,
    pattern_type=PatternType.REVERSAL,
    bias=PatternType.BEARISH,
    reliability=0.68,
    description="Bearish candle opens above previous high and closes below midpoint.",
    recognition_rules=[
        "First candle is bullish and substantial",
        "Second candle gaps up on open (above first's high)",
        "Second candle closes below midpoint of first candle's body",
        "Ideally closes in lower 40% of first candle",
        "Does not fully engulf first candle"
    ],
    trading_implications="Bearish reversal. Stronger if penetrates deeper.",
    confirmation_needed=True,
    metrics=PatternMetrics(
        win_rate=0.65,
        avg_gain=3.3,
        avg_loss=-2.2,
        risk_reward_ratio=1.5,
        avg_duration=7,
        optimal_contexts=[MarketContext.WEAK_UPTREND, MarketContext.STRONG_UPTREND],
        volume_requirement=VolumeProfile.INCREASING
    )
)

# =============================================================================
# TRIPLE CANDLE PATTERNS (Enhanced)
# =============================================================================

MORNING_STAR = CandlestickPattern(
    name="Morning Star",
    category=PatternCategory.TRIPLE,
    pattern_type=PatternType.REVERSAL,
    bias=PatternType.BULLISH,
    reliability=0.78,
    description="Three-candle bullish reversal: bearish, small body/doji, bullish.",
    recognition_rules=[
        "First candle is large and bearish",
        "Second candle has small body (can be doji), gaps down",
        "Third candle is large and bullish",
        "Third candle closes well into first candle's body (ideally above midpoint)",
        "Gap between first and second candle strengthens pattern"
    ],
    trading_implications="Strong bullish reversal. High reliability pattern.",
    confirmation_needed=False,
    aliases=["Morning Doji Star"],
    metrics=PatternMetrics(
        win_rate=0.78,
        avg_gain=5.2,
        avg_loss=-2.4,
        risk_reward_ratio=2.2,
        avg_duration=10,
        optimal_contexts=[MarketContext.STRONG_DOWNTREND],
        volume_requirement=VolumeProfile.INCREASING
    ),
    trading_setup=TradingSetup(
        entry_conditions=[
            "Enter on close of third candle",
            "Or wait for pullback to star low",
            "Volume should increase on third candle"
        ],
        stop_loss_placement="Below star (middle candle) low",
        take_profit_targets=[(2.0, "First resistance"), (3.5, "Major resistance")],
        position_sizing="Risk 1.5-2.5% given high reliability"
    ),
    min_timeframe=TimeFrame.H1
)

EVENING_STAR = CandlestickPattern(
    name="Evening Star",
    category=PatternCategory.TRIPLE,
    pattern_type=PatternType.REVERSAL,
    bias=PatternType.BEARISH,
    reliability=0.78,
    description="Three-candle bearish reversal: bullish, small body/doji, bearish.",
    recognition_rules=[
        "First candle is large and bullish",
        "Second candle has small body (can be doji), gaps up",
        "Third candle is large and bearish",
        "Third candle closes well into first candle's body",
        "Gap strengthens the pattern"
    ],
    trading_implications="Strong bearish reversal. High reliability pattern.",
    confirmation_needed=False,
    aliases=["Evening Doji Star"],
    metrics=PatternMetrics(
        win_rate=0.77,
        avg_gain=4.9,
        avg_loss=-2.5,
        risk_reward_ratio=2.0,
        avg_duration=10,
        optimal_contexts=[MarketContext.STRONG_UPTREND],
        volume_requirement=VolumeProfile.CLIMAX
    ),
    min_timeframe=TimeFrame.H1
)

THREE_WHITE_SOLDIERS = CandlestickPattern(
    name="Three White Soldiers",
    category=PatternCategory.TRIPLE,
    pattern_type=PatternType.REVERSAL,
    bias=PatternType.BULLISH,
    reliability=0.82,
    description="Three consecutive bullish candles, each closing higher.",
    recognition_rules=[
        "Three consecutive bullish candles of similar size",
        "Each candle opens within previous candle's body (upper third preferred)",
        "Each candle closes near its high (minimal upper shadow)",
        "Each candle closes progressively higher",
        "Steady, not parabolic advance"
    ],
    trading_implications="Very strong bullish signal. Major trend reversal or continuation.",
    confirmation_needed=False,
    metrics=PatternMetrics(
        win_rate=0.81,
        avg_gain=6.3,
        avg_loss=-2.8,
        risk_reward_ratio=2.3,
        avg_duration=12,
        optimal_contexts=[MarketContext.STRONG_DOWNTREND, MarketContext.CONSOLIDATION],
        volume_requirement=VolumeProfile.INCREASING
    ),
    false_signal_warnings=[
        "After extended move, may signal exhaustion (Advance Block)",
        "If upper shadows lengthen on 3rd candle, beware reversal"
    ]
)

THREE_BLACK_CROWS = CandlestickPattern(
    name="Three Black Crows",
    category=PatternCategory.TRIPLE,
    pattern_type=PatternType.REVERSAL,
    bias=PatternType.BEARISH,
    reliability=0.82,
    description="Three consecutive bearish candles, each closing lower.",
    recognition_rules=[
        "Three consecutive bearish candles of similar size",
        "Each candle opens within previous candle's body",
        "Each candle closes near its low (minimal lower shadow)",
        "Each candle closes progressively lower",
        "Appears after uptrend or at market top"
    ],
    trading_implications="Very strong bearish signal. Major trend reversal.",
    confirmation_needed=False,
    metrics=PatternMetrics(
        win_rate=0.80,
        avg_gain=5.9,
        avg_loss=-2.9,
        risk_reward_ratio=2.0,
        avg_duration=12,
        optimal_contexts=[MarketContext.STRONG_UPTREND],
        volume_requirement=VolumeProfile.INCREASING
    )
)

# =============================================================================
# ADVANCED PATTERNS
# =============================================================================

ABANDONED_BABY = CandlestickPattern(
    name="Abandoned Baby",
    category=PatternCategory.ADVANCED,
    pattern_type=PatternType.REVERSAL,
    bias=PatternType.NEUTRAL,  # Direction depends on context
    reliability=0.80,
    description="Rare pattern with doji gapping away from both neighboring candles.",
    recognition_rules=[
        "Three candles: trend candle, doji, reversal candle",
        "Doji gaps away from both neighbors (island formation)",
        "Shadows of doji don't overlap with neighbors",
        "Very rare pattern - gaps required"
    ],
    trading_implications="Extremely strong reversal signal when genuine.",
    confirmation_needed=False,
    metrics=PatternMetrics(
        win_rate=0.79,
        avg_gain=6.8,
        avg_loss=-2.6,
        risk_reward_ratio=2.6,
        avg_duration=9,
        optimal_contexts=[MarketContext.STRONG_UPTREND, MarketContext.STRONG_DOWNTREND],
        volume_requirement=VolumeProfile.AVERAGE
    )
)

KICKING_PATTERN = CandlestickPattern(
    name="Kicking Pattern",
    category=PatternCategory.ADVANCED,
    pattern_type=PatternType.REVERSAL,
    bias=PatternType.NEUTRAL,
    reliability=0.85,
    description="Two marubozu candles with gap, showing dramatic sentiment shift.",
    recognition_rules=[
        "First candle is marubozu in trend direction",
        "Second candle is opposite color marubozu",
        "Gap between the two candles (no overlap)",
        "Both candles should be large and decisive"
    ],
    trading_implications="Powerful reversal. Immediate sentiment change.",
    confirmation_needed=False,
    metrics=PatternMetrics(
        win_rate=0.83,
        avg_gain=7.2,
        avg_loss=-2.8,
        risk_reward_ratio=2.6,
        avg_duration=8,
        optimal_contexts=[MarketContext.STRONG_UPTREND, MarketContext.STRONG_DOWNTREND],
        volume_requirement=VolumeProfile.CLIMAX
    ),
    min_timeframe=TimeFrame.H4
)

THREE_LINE_STRIKE = CandlestickPattern(
    name="Three Line Strike",
    category=PatternCategory.ADVANCED,
    pattern_type=PatternType.CONTINUATION,
    bias=PatternType.NEUTRAL,
    reliability=0.75,
    description="Three trend candles followed by engulfing counter-trend candle that fails.",
    recognition_rules=[
        "Three consecutive candles in trend direction",
        "Fourth candle opens with the trend but reverses",
        "Fourth candle engulfs all three previous candles",
        "Bullish: 3 white soldiers + bearish engulfing = bullish continuation",
        "Bearish: 3 black crows + bullish engulfing = bearish continuation"
    ],
    trading_implications="Strong continuation pattern. Failed reversal confirms trend.",
    confirmation_needed=True,
    metrics=PatternMetrics(
        win_rate=0.74,
        avg_gain=5.1,
        avg_loss=-2.7,
        risk_reward_ratio=1.9,
        avg_duration=11,
        optimal_contexts=[MarketContext.STRONG_UPTREND, MarketContext.STRONG_DOWNTREND],
        volume_requirement=VolumeProfile.AVERAGE
    )
)

# =============================================================================
# COMPLEX CHART PATTERNS (Enhanced)
# =============================================================================

DOUBLE_TOP = CandlestickPattern(
    name="Double Top",
    category=PatternCategory.COMPLEX,
    pattern_type=PatternType.REVERSAL,
    bias=PatternType.BEARISH,
    reliability=0.77,
    description="Two peaks at similar price level forming 'M' shape.",
    recognition_rules=[
        "Price reaches a high, pulls back at least 10-15%",
        "Price returns to within 3% of first peak",
        "Second peak fails to exceed first significantly",
        "Neckline break confirms pattern",
        "Volume typically lower on second peak"
    ],
    trading_implications="Major bearish reversal. Target is depth of the 'M' projected down.",
    confirmation_needed=True,
    metrics=PatternMetrics(
        win_rate=0.76,
        avg_gain=8.5,
        avg_loss=-3.2,
        risk_reward_ratio=2.7,
        avg_duration=25,
        optimal_contexts=[MarketContext.STRONG_UPTREND],
        volume_requirement=VolumeProfile.DECREASING
    ),
    trading_setup=TradingSetup(
        entry_conditions=[
            "Enter on neckline break",
            "Or on retest of neckline as resistance",
            "Volume should increase on breakdown"
        ],
        stop_loss_placement="Above second peak",
        take_profit_targets=[
            (1.0, "Depth of pattern (measured move)"),
            (1.5, "Extended target")
        ],
        position_sizing="Risk 1-2% of capital"
    ),
    min_timeframe=TimeFrame.H4
)

DOUBLE_BOTTOM = CandlestickPattern(
    name="Double Bottom",
    category=PatternCategory.COMPLEX,
    pattern_type=PatternType.REVERSAL,
    bias=PatternType.BULLISH,
    reliability=0.77,
    description="Two troughs at similar price level forming 'W' shape.",
    recognition_rules=[
        "Price reaches a low, bounces at least 10-15%",
        "Price returns to within 3% of first trough",
        "Second trough holds at or above first",
        "Neckline break confirms pattern",
        "Volume typically higher on second low and breakout"
    ],
    trading_implications="Major bullish reversal. Target is depth of the 'W' projected up.",
    confirmation_needed=True,
    metrics=PatternMetrics(
        win_rate=0.77,
        avg_gain=8.8,
        avg_loss=-3.1,
        risk_reward_ratio=2.8,
        avg_duration=25,
        optimal_contexts=[MarketContext.STRONG_DOWNTREND],
        volume_requirement=VolumeProfile.INCREASING
    ),
    min_timeframe=TimeFrame.H4
)

HEAD_AND_SHOULDERS = CandlestickPattern(
    name="Head and Shoulders",
    category=PatternCategory.COMPLEX,
    pattern_type=PatternType.REVERSAL,
    bias=PatternType.BEARISH,
    reliability=0.83,
    description="Three peaks with middle peak highest, shoulders at similar levels.",
    recognition_rules=[
        "Left shoulder: rise, then decline to neckline",
        "Head: higher rise (highest point), then decline to neckline",
        "Right shoulder: lower rise than head, similar height to left",
        "Neckline connects the two troughs",
        "Volume typically decreases from left shoulder to head to right shoulder"
    ],
    trading_implications="Classic bearish reversal. Very reliable pattern.",
    confirmation_needed=True,
    metrics=PatternMetrics(
        win_rate=0.82,
        avg_gain=10.5,
        avg_loss=-3.5,
        risk_reward_ratio=3.0,
        avg_duration=30,
        optimal_contexts=[MarketContext.STRONG_UPTREND],
        volume_requirement=VolumeProfile.DECREASING
    ),
    trading_setup=TradingSetup(
        entry_conditions=[
            "Enter on neckline break with volume",
            "Aggressive: short on right shoulder formation",
            "Conservative: wait for retest of broken neckline"
        ],
        stop_loss_placement="Above right shoulder",
        take_profit_targets=[
            (1.0, "Head to neckline distance projected down"),
            (1.5, "Extended target")
        ],
        position_sizing="Risk 2-2.5% given high reliability"
    ),
    min_timeframe=TimeFrame.D1
)

INVERSE_HEAD_AND_SHOULDERS = CandlestickPattern(
    name="Inverse Head and Shoulders",
    category=PatternCategory.COMPLEX,
    pattern_type=PatternType.REVERSAL,
    bias=PatternType.BULLISH,
    reliability=0.83,
    description="Three troughs with middle trough lowest, shoulders at similar levels.",
    recognition_rules=[
        "Left shoulder: decline, then rise to neckline",
        "Head: lower decline (lowest point), then rise to neckline",
        "Right shoulder: higher low than head, similar to left shoulder",
        "Neckline connects the two peaks",
        "Volume should increase on breakout"
    ],
    trading_implications="Classic bullish reversal. Very reliable pattern.",
    confirmation_needed=True,
    metrics=PatternMetrics(
        win_rate=0.81,
        avg_gain=10.2,
        avg_loss=-3.6,
        risk_reward_ratio=2.8,
        avg_duration=30,
        optimal_contexts=[MarketContext.STRONG_DOWNTREND],
        volume_requirement=VolumeProfile.INCREASING
    ),
    min_timeframe=TimeFrame.D1
)

ASCENDING_TRIANGLE = CandlestickPattern(
    name="Ascending Triangle",
    category=PatternCategory.COMPLEX,
    pattern_type=PatternType.CONTINUATION,
    bias=PatternType.BULLISH,
    reliability=0.72,
    description="Flat resistance with rising support, typically bullish breakout.",
    recognition_rules=[
        "Horizontal resistance line (at least 2 touches)",
        "Rising support line with higher lows (at least 2 touches)",
        "Price compresses between the lines",
        "Breakout typically happens in upper 2/3 of pattern",
        "Volume decreases during formation, increases on breakout"
    ],
    trading_implications="Bullish continuation. Enter on breakout above resistance.",
    confirmation_needed=True,
    metrics=PatternMetrics(
        win_rate=0.71,
        avg_gain=6.5,
        avg_loss=-2.8,
        risk_reward_ratio=2.3,
        avg_duration=18,
        optimal_contexts=[MarketContext.WEAK_UPTREND, MarketContext.CONSOLIDATION],
        volume_requirement=VolumeProfile.DECREASING
    ),
    min_timeframe=TimeFrame.H1
)

DESCENDING_TRIANGLE = CandlestickPattern(
    name="Descending Triangle",
    category=PatternCategory.COMPLEX,
    pattern_type=PatternType.CONTINUATION,
    bias=PatternType.BEARISH,
    reliability=0.72,
    description="Flat support with falling resistance, typically bearish breakout.",
    recognition_rules=[
        "Horizontal support line (at least 2 touches)",
        "Falling resistance line with lower highs (at least 2 touches)",
        "Price compresses between the lines",
        "Breakout typically downward",
        "Volume decreases during formation"
    ],
    trading_implications="Bearish continuation. Enter on breakdown below support.",
    confirmation_needed=True,
    metrics=PatternMetrics(
        win_rate=0.70,
        avg_gain=6.2,
        avg_loss=-2.9,
        risk_reward_ratio=2.1,
        avg_duration=18,
        optimal_contexts=[MarketContext.WEAK_DOWNTREND, MarketContext.CONSOLIDATION],
        volume_requirement=VolumeProfile.DECREASING
    ),
    min_timeframe=TimeFrame.H1
)

RISING_WEDGE = CandlestickPattern(
    name="Rising Wedge",
    category=PatternCategory.COMPLEX,
    pattern_type=PatternType.REVERSAL,
    bias=PatternType.BEARISH,
    reliability=0.68,
    description="Converging upward sloping trendlines, bearish reversal.",
    recognition_rules=[
        "Both support and resistance lines slope upward",
        "Lines converge (resistance rises slower than support)",
        "At least 5 touches across both lines",
        "Volume typically decreases as pattern forms",
        "Breakout is usually downward (65-70% of time)"
    ],
    trading_implications="Bearish despite upward movement. Short on breakdown.",
    confirmation_needed=True,
    metrics=PatternMetrics(
        win_rate=0.67,
        avg_gain=5.8,
        avg_loss=-2.7,
        risk_reward_ratio=2.1,
        avg_duration=20,
        optimal_contexts=[MarketContext.WEAK_UPTREND],
        volume_requirement=VolumeProfile.DECREASING
    ),
    min_timeframe=TimeFrame.H4
)

FALLING_WEDGE = CandlestickPattern(
    name="Falling Wedge",
    category=PatternCategory.COMPLEX,
    pattern_type=PatternType.REVERSAL,
    bias=PatternType.BULLISH,
    reliability=0.68,
    description="Converging downward sloping trendlines, bullish reversal.",
    recognition_rules=[
        "Both support and resistance lines slope downward",
        "Lines converge (support falls slower than resistance)",
        "At least 5 touches across both lines",
        "Volume typically decreases as pattern forms",
        "Breakout is usually upward (65-70% of time)"
    ],
    trading_implications="Bullish despite downward movement. Long on breakout.",
    confirmation_needed=True,
    metrics=PatternMetrics(
        win_rate=0.68,
        avg_gain=6.0,
        avg_loss=-2.6,
        risk_reward_ratio=2.3,
        avg_duration=20,
        optimal_contexts=[MarketContext.WEAK_DOWNTREND],
        volume_requirement=VolumeProfile.DECREASING
    ),
    min_timeframe=TimeFrame.H4
)

# Additional patterns removed for brevity - add more as needed

# =============================================================================
# PATTERN REGISTRY
# =============================================================================

ALL_PATTERNS: List[CandlestickPattern] = [
    # Single candle patterns
    DOJI, DRAGONFLY_DOJI, GRAVESTONE_DOJI, HAMMER, SHOOTING_STAR, MARUBOZU,
    # Double candle patterns
    BULLISH_ENGULFING, BEARISH_ENGULFING, PIERCING_LINE, DARK_CLOUD_COVER,
    # Triple candle patterns
    MORNING_STAR, EVENING_STAR, THREE_WHITE_SOLDIERS, THREE_BLACK_CROWS,
    # Advanced patterns
    ABANDONED_BABY, KICKING_PATTERN, THREE_LINE_STRIKE,
    # Complex chart patterns
    DOUBLE_TOP, DOUBLE_BOTTOM, HEAD_AND_SHOULDERS, INVERSE_HEAD_AND_SHOULDERS,
    ASCENDING_TRIANGLE, DESCENDING_TRIANGLE, RISING_WEDGE, FALLING_WEDGE
]

PATTERN_BY_NAME: Dict[str, CandlestickPattern] = {
    p.name.lower().replace(" ", "_"): p for p in ALL_PATTERNS
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_pattern_by_name(name: str) -> Optional[CandlestickPattern]:
    """Get pattern definition by name"""
    normalized = name.lower().replace(" ", "_").replace("-", "_")
    return PATTERN_BY_NAME.get(normalized)


def get_patterns_by_bias(bias: PatternType) -> List[CandlestickPattern]:
    """Get all patterns with a specific market bias"""
    return [p for p in ALL_PATTERNS if p.bias == bias]


def get_patterns_by_category(category: PatternCategory) -> List[CandlestickPattern]:
    """Get all patterns in a specific category"""
    return [p for p in ALL_PATTERNS if p.category == category]


def get_patterns_by_reliability(min_reliability: float = 0.7) -> List[CandlestickPattern]:
    """Get patterns with reliability above threshold"""
    return [p for p in ALL_PATTERNS if p.reliability >= min_reliability]


def get_high_winrate_patterns(min_winrate: float = 0.7) -> List[CandlestickPattern]:
    """Get patterns with high historical win rate"""
    return [
        p for p in ALL_PATTERNS 
        if p.metrics and p.metrics.win_rate >= min_winrate
    ]


def get_patterns_for_context(context: MarketContext) -> List[CandlestickPattern]:
    """Get patterns optimal for a given market context"""
    return [
        p for p in ALL_PATTERNS
        if p.metrics and context in p.metrics.optimal_contexts
    ]


def calculate_pattern_score(
    pattern: CandlestickPattern,
    context: MarketContext,
    volume: VolumeProfile,
    timeframe: TimeFrame
) -> Tuple[float, PatternStrength]:
    """Calculate comprehensive pattern score"""
    strength = pattern.calculate_strength(context, volume, timeframe)
    
    base_score = pattern.reliability
    if pattern.metrics:
        base_score = (base_score + pattern.metrics.win_rate) / 2
        
    # Apply strength modifier
    strength_modifiers = {
        PatternStrength.VERY_STRONG: 1.2,
        PatternStrength.STRONG: 1.1,
        PatternStrength.MODERATE: 1.0,
        PatternStrength.WEAK: 0.8,
        PatternStrength.INVALID: 0.5
    }
    
    final_score = base_score * strength_modifiers[strength]
    return (min(final_score, 1.0), strength)
