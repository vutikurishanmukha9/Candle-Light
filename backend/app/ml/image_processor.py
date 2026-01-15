"""
Image Processor for Candlestick Charts

Handles image preprocessing, candlestick extraction, and visual analysis
using OpenCV and Pillow for robust chart processing.
"""

import io
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV not available. Some features will be limited.")


logger = logging.getLogger(__name__)


@dataclass
class Candlestick:
    """Represents a single extracted candlestick"""
    index: int
    x_position: int
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    is_bullish: bool
    body_height: float
    upper_shadow: float
    lower_shadow: float
    total_range: float
    
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


@dataclass
class ChartAnalysis:
    """Results of chart image analysis"""
    candlesticks: List[Candlestick]
    trend_direction: str  # 'uptrend', 'downtrend', 'sideways'
    trend_strength: float  # 0.0 to 1.0
    dominant_color: str  # 'green', 'red', 'mixed'
    chart_quality: float  # 0.0 to 1.0
    image_dimensions: Tuple[int, int]


class ImageProcessor:
    """
    Processes candlestick chart images for pattern analysis.
    
    This processor uses computer vision techniques to:
    1. Detect candlestick regions in the image
    2. Extract color information (bullish/bearish)
    3. Estimate price levels from visual positions
    4. Determine overall trend direction
    """
    
    # Common candlestick colors
    GREEN_RANGE = ((35, 100, 100), (85, 255, 255))  # HSV range for green
    RED_RANGE = ((0, 100, 100), (10, 255, 255))     # HSV range for red
    RED_RANGE_2 = ((170, 100, 100), (180, 255, 255))  # Red wraps around in HSV
    
    def __init__(self):
        self.min_candle_width = 3
        self.max_candle_width = 50
    
    def load_image(self, image_data: bytes) -> Optional[np.ndarray]:
        """Load image from bytes into numpy array"""
        try:
            image = Image.open(io.BytesIO(image_data))
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return np.array(image)
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return None
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better candlestick detection.
        - Resize if too large
        - Enhance contrast
        - Reduce noise
        """
        if not CV2_AVAILABLE:
            return image
        
        # Resize if too large (keep aspect ratio)
        max_dimension = 1920
        h, w = image.shape[:2]
        if max(h, w) > max_dimension:
            scale = max_dimension / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        # Apply slight Gaussian blur to reduce noise
        image = cv2.GaussianBlur(image, (3, 3), 0)
        
        return image
    
    def detect_candlestick_colors(self, image: np.ndarray) -> Tuple[int, int]:
        """
        Count green (bullish) and red (bearish) pixels in the image.
        Returns (green_count, red_count)
        """
        if not CV2_AVAILABLE:
            # Fallback: analyze RGB channels
            return self._detect_colors_rgb(image)
        
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Detect green pixels
        green_mask = cv2.inRange(hsv, 
                                 np.array(self.GREEN_RANGE[0]), 
                                 np.array(self.GREEN_RANGE[1]))
        green_count = cv2.countNonZero(green_mask)
        
        # Detect red pixels (two ranges because red wraps in HSV)
        red_mask1 = cv2.inRange(hsv, 
                                np.array(self.RED_RANGE[0]), 
                                np.array(self.RED_RANGE[1]))
        red_mask2 = cv2.inRange(hsv, 
                                np.array(self.RED_RANGE_2[0]), 
                                np.array(self.RED_RANGE_2[1]))
        red_count = cv2.countNonZero(red_mask1) + cv2.countNonZero(red_mask2)
        
        return green_count, red_count
    
    def _detect_colors_rgb(self, image: np.ndarray) -> Tuple[int, int]:
        """Fallback color detection using RGB analysis"""
        # Simple heuristic: compare red vs green channel intensity
        r_channel = image[:, :, 0].astype(float)
        g_channel = image[:, :, 1].astype(float)
        
        # Pixels where green dominates
        green_dominant = np.sum((g_channel > r_channel * 1.3) & (g_channel > 100))
        # Pixels where red dominates
        red_dominant = np.sum((r_channel > g_channel * 1.3) & (r_channel > 100))
        
        return int(green_dominant), int(red_dominant)
    
    def estimate_trend(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Estimate the overall trend direction from the image.
        Analyzes the slope of price movement from left to right.
        
        Returns: (direction, strength)
        """
        h, w = image.shape[:2]
        
        if not CV2_AVAILABLE:
            # Simple fallback: compare left vs right brightness
            left_half = image[:, :w//2]
            right_half = image[:, w//2:]
            
            # For charts, lighter areas might indicate higher prices
            left_mean = np.mean(left_half)
            right_mean = np.mean(right_half)
            
            diff = right_mean - left_mean
            if abs(diff) < 5:
                return "sideways", 0.3
            elif diff > 0:
                return "uptrend", min(abs(diff) / 50, 1.0)
            else:
                return "downtrend", min(abs(diff) / 50, 1.0)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Divide image into vertical strips
        num_strips = 10
        strip_width = w // num_strips
        strip_means = []
        
        for i in range(num_strips):
            strip = gray[:, i*strip_width:(i+1)*strip_width]
            # Find average "price level" (y-position of darkest pixels)
            # In charts, candles are typically darker than background
            threshold = np.percentile(strip, 30)
            dark_mask = strip < threshold
            if np.any(dark_mask):
                y_positions = np.where(dark_mask)[0]
                avg_y = np.mean(y_positions)
                # Invert because y increases downward in images
                strip_means.append(h - avg_y)
            else:
                strip_means.append(h / 2)
        
        # Calculate trend using linear regression
        x = np.arange(len(strip_means))
        slope, _ = np.polyfit(x, strip_means, 1)
        
        # Normalize slope to determine trend strength
        normalized_slope = slope / (h / num_strips)
        
        if abs(normalized_slope) < 0.05:
            return "sideways", 0.3 + abs(normalized_slope) * 2
        elif normalized_slope > 0:
            return "uptrend", min(0.5 + normalized_slope * 2, 1.0)
        else:
            return "downtrend", min(0.5 + abs(normalized_slope) * 2, 1.0)
    
    def detect_candlesticks(self, image: np.ndarray) -> List[Candlestick]:
        """
        Attempt to detect individual candlesticks in the image.
        
        This is a simplified detection that works best with clear charts.
        For complex charts, the AI service provides better results.
        """
        if not CV2_AVAILABLE:
            return self._estimate_candlesticks_simple(image)
        
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Detect green and red regions
        green_mask = cv2.inRange(hsv, 
                                 np.array(self.GREEN_RANGE[0]), 
                                 np.array(self.GREEN_RANGE[1]))
        red_mask1 = cv2.inRange(hsv, 
                                np.array(self.RED_RANGE[0]), 
                                np.array(self.RED_RANGE[1]))
        red_mask2 = cv2.inRange(hsv, 
                                np.array(self.RED_RANGE_2[0]), 
                                np.array(self.RED_RANGE_2[1]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Combine masks
        candle_mask = cv2.bitwise_or(green_mask, red_mask)
        
        # Find contours (potential candlesticks)
        contours, _ = cv2.findContours(candle_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        candlesticks = []
        for i, contour in enumerate(contours):
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # Filter by aspect ratio and size
            if cw < self.min_candle_width or cw > self.max_candle_width:
                continue
            if ch < 10:  # Too short
                continue
            
            # Determine if bullish or bearish based on color
            roi = hsv[y:y+ch, x:x+cw]
            green_in_roi = cv2.countNonZero(cv2.inRange(roi, 
                                            np.array(self.GREEN_RANGE[0]), 
                                            np.array(self.GREEN_RANGE[1])))
            red_in_roi = cv2.countNonZero(cv2.inRange(roi,
                                          np.array(self.RED_RANGE[0]),
                                          np.array(self.RED_RANGE[1])))
            is_bullish = green_in_roi > red_in_roi
            
            # Estimate OHLC from position (normalized 0-100)
            # Note: These are relative values, not actual prices
            high_price = (h - y) / h * 100
            low_price = (h - (y + ch)) / h * 100
            
            # Estimate open/close based on color
            body_height = ch * 0.6  # Assume body is 60% of total height
            shadow_portion = ch * 0.2
            
            if is_bullish:
                open_price = low_price + shadow_portion / h * 100
                close_price = high_price - shadow_portion / h * 100
            else:
                close_price = low_price + shadow_portion / h * 100
                open_price = high_price - shadow_portion / h * 100
            
            candlestick = Candlestick(
                index=i,
                x_position=x,
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                close_price=close_price,
                is_bullish=is_bullish,
                body_height=abs(close_price - open_price),
                upper_shadow=high_price - max(open_price, close_price),
                lower_shadow=min(open_price, close_price) - low_price,
                total_range=high_price - low_price
            )
            candlesticks.append(candlestick)
        
        # Sort by x position (left to right)
        candlesticks.sort(key=lambda c: c.x_position)
        
        # Re-index after sorting
        for i, candle in enumerate(candlesticks):
            candle.index = i
        
        return candlesticks
    
    def _estimate_candlesticks_simple(self, image: np.ndarray) -> List[Candlestick]:
        """
        Simple candlestick estimation when OpenCV is not available.
        Uses basic color analysis to estimate the chart composition.
        """
        h, w = image.shape[:2]
        green_count, red_count = self._detect_colors_rgb(image)
        total = green_count + red_count + 1
        
        # Estimate number of candles based on image width
        estimated_candles = max(5, min(50, w // 20))
        
        # Create synthetic candlesticks based on color distribution
        bullish_ratio = green_count / total
        candlesticks = []
        
        for i in range(estimated_candles):
            # Determine if this candle should be bullish based on overall distribution
            is_bullish = (i % 3 == 0 and bullish_ratio > 0.5) or \
                        (i % 3 != 0 and bullish_ratio > 0.7)
            
            # Create semi-random but realistic candle data
            base_price = 50 + (i / estimated_candles) * 20 * (1 if bullish_ratio > 0.5 else -1)
            volatility = 5 + abs(50 - base_price) * 0.1
            
            high_price = base_price + volatility
            low_price = base_price - volatility
            
            if is_bullish:
                open_price = low_price + volatility * 0.3
                close_price = high_price - volatility * 0.3
            else:
                open_price = high_price - volatility * 0.3
                close_price = low_price + volatility * 0.3
            
            candlesticks.append(Candlestick(
                index=i,
                x_position=int(i * w / estimated_candles),
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                close_price=close_price,
                is_bullish=is_bullish,
                body_height=abs(close_price - open_price),
                upper_shadow=high_price - max(open_price, close_price),
                lower_shadow=min(open_price, close_price) - low_price,
                total_range=high_price - low_price
            ))
        
        return candlesticks
    
    def analyze_chart(self, image_data: bytes) -> Optional[ChartAnalysis]:
        """
        Perform complete analysis of a chart image.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            ChartAnalysis object with extracted information
        """
        image = self.load_image(image_data)
        if image is None:
            return None
        
        h, w = image.shape[:2]
        
        # Preprocess
        processed = self.preprocess_image(image)
        
        # Detect colors
        green_count, red_count = self.detect_candlestick_colors(processed)
        total_colored = green_count + red_count + 1
        
        if green_count > red_count * 1.5:
            dominant_color = "green"
        elif red_count > green_count * 1.5:
            dominant_color = "red"
        else:
            dominant_color = "mixed"
        
        # Estimate trend
        trend_direction, trend_strength = self.estimate_trend(processed)
        
        # Detect candlesticks
        candlesticks = self.detect_candlesticks(processed)
        
        # Estimate chart quality based on detection success
        if len(candlesticks) >= 10:
            chart_quality = 0.8
        elif len(candlesticks) >= 5:
            chart_quality = 0.6
        elif len(candlesticks) >= 1:
            chart_quality = 0.4
        else:
            chart_quality = 0.2
        
        return ChartAnalysis(
            candlesticks=candlesticks,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            dominant_color=dominant_color,
            chart_quality=chart_quality,
            image_dimensions=(w, h)
        )
