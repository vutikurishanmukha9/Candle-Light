"""
Tests for ML Pattern Detection Module

Unit tests for candlestick pattern detection engine.
"""

import pytest
from unittest.mock import MagicMock, patch
import io

from app.ml.patterns import (
    PatternType,
    PatternCategory,
    MarketContext,
    VolumeProfile,
    PatternStrength,
    TimeFrame,
    CandlestickPattern,
    ALL_PATTERNS,
    get_pattern_by_name,
    get_patterns_by_bias,
    get_patterns_by_category,
    get_patterns_by_reliability,
    get_high_winrate_patterns,
    calculate_pattern_score,
)
from app.ml.pattern_detector import PatternDetector


class TestPatternRegistry:
    """Tests for pattern registry and helper functions."""
    
    def test_all_patterns_not_empty(self):
        """Verify pattern registry is populated."""
        assert len(ALL_PATTERNS) > 0
        assert len(ALL_PATTERNS) >= 20  # Expect at least 20 patterns
    
    def test_all_patterns_have_required_fields(self):
        """Verify all patterns have required attributes."""
        for pattern in ALL_PATTERNS:
            assert pattern.name, "Pattern must have a name"
            assert pattern.category in PatternCategory
            assert pattern.pattern_type in PatternType
            assert pattern.bias in PatternType
            assert 0 <= pattern.reliability <= 1
            assert pattern.description
            assert len(pattern.recognition_rules) > 0
    
    def test_get_pattern_by_name(self):
        """Test pattern lookup by name."""
        # Test exact match
        hammer = get_pattern_by_name("hammer")
        assert hammer is not None
        assert hammer.name == "Hammer"
        
        # Test with spaces
        morning_star = get_pattern_by_name("morning star")
        assert morning_star is not None
        assert morning_star.name == "Morning Star"
        
        # Test with dashes
        three_soldiers = get_pattern_by_name("three-white-soldiers")
        assert three_soldiers is not None
        
        # Test non-existent pattern
        invalid = get_pattern_by_name("nonexistent_pattern")
        assert invalid is None
    
    def test_get_patterns_by_bias(self):
        """Test filtering patterns by market bias."""
        bullish = get_patterns_by_bias(PatternType.BULLISH)
        assert len(bullish) > 0
        assert all(p.bias == PatternType.BULLISH for p in bullish)
        
        bearish = get_patterns_by_bias(PatternType.BEARISH)
        assert len(bearish) > 0
        assert all(p.bias == PatternType.BEARISH for p in bearish)
    
    def test_get_patterns_by_category(self):
        """Test filtering patterns by category."""
        single = get_patterns_by_category(PatternCategory.SINGLE)
        assert len(single) > 0
        assert all(p.category == PatternCategory.SINGLE for p in single)
        
        double = get_patterns_by_category(PatternCategory.DOUBLE)
        assert len(double) > 0
    
    def test_get_patterns_by_reliability(self):
        """Test filtering by reliability threshold."""
        reliable = get_patterns_by_reliability(0.7)
        assert len(reliable) > 0
        assert all(p.reliability >= 0.7 for p in reliable)
    
    def test_get_high_winrate_patterns(self):
        """Test filtering by win rate."""
        high_wr = get_high_winrate_patterns(0.7)
        for p in high_wr:
            assert p.metrics is not None
            assert p.metrics.win_rate >= 0.7


class TestPatternStrengthCalculation:
    """Tests for pattern strength calculations."""
    
    def test_calculate_pattern_score(self):
        """Test pattern score calculation."""
        hammer = get_pattern_by_name("hammer")
        assert hammer is not None
        
        score, strength = calculate_pattern_score(
            hammer,
            MarketContext.STRONG_DOWNTREND,
            VolumeProfile.INCREASING,
            TimeFrame.H1
        )
        
        assert 0 <= score <= 1
        assert strength in PatternStrength
    
    def test_strength_varies_with_context(self):
        """Test that strength changes based on market context."""
        hammer = get_pattern_by_name("hammer")
        assert hammer is not None
        
        # Optimal context
        score_good, _ = calculate_pattern_score(
            hammer,
            MarketContext.STRONG_DOWNTREND,
            VolumeProfile.INCREASING,
            TimeFrame.H1
        )
        
        # Poor context
        score_bad, _ = calculate_pattern_score(
            hammer,
            MarketContext.RANGING,
            VolumeProfile.LOW,
            TimeFrame.M1
        )
        
        assert score_good > score_bad


class TestPatternDetector:
    """Tests for pattern detection engine."""
    
    @pytest.fixture
    def detector(self):
        """Create pattern detector instance."""
        return PatternDetector()
    
    def test_detector_initialization(self, detector):
        """Test detector initializes correctly."""
        assert detector is not None
        assert hasattr(detector, 'analyze')
    
    def test_fallback_result(self, detector):
        """Test fallback result when analysis fails."""
        result = detector._get_fallback_result()
        
        assert result is not None
        assert result.confidence == 0
        assert result.market_bias == "neutral"
        assert any("fallback" in r.lower() or "unable" in r.lower() 
                   for r in result.reasoning.split())


class TestPatternMetrics:
    """Tests for pattern metrics."""
    
    def test_all_patterns_have_metrics(self):
        """Verify patterns have metrics defined."""
        patterns_with_metrics = [p for p in ALL_PATTERNS if p.metrics is not None]
        # Most patterns should have metrics
        assert len(patterns_with_metrics) >= len(ALL_PATTERNS) * 0.5
    
    def test_metrics_values_are_valid(self):
        """Verify metric values are within valid ranges."""
        for pattern in ALL_PATTERNS:
            if pattern.metrics:
                assert 0 <= pattern.metrics.win_rate <= 1
                assert pattern.metrics.avg_gain >= 0
                assert pattern.metrics.avg_loss <= 0
                assert pattern.metrics.risk_reward_ratio > 0
                assert pattern.metrics.avg_duration > 0
