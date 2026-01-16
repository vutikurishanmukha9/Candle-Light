"""
Enhanced AI Service

Advanced AI-powered chart analysis with multi-provider support, caching,
retry logic, and comprehensive pattern detection.
"""

import base64
import json
import time
import hashlib
import asyncio
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from functools import wraps

from app.config import settings
from app.schemas.analysis import PatternResult, AnalysisResult, EntryTiming
from app.core.exceptions import AIServiceError


# Cache for analysis results
_analysis_cache: Dict[str, Tuple[AnalysisResult, datetime]] = {}
CACHE_DURATION = timedelta(hours=1)


def cache_result(func):
    """Decorator to cache analysis results based on image hash."""
    @wraps(func)
    async def wrapper(self, image_path: str, image_bytes: Optional[bytes] = None, use_cache: bool = True):
        if not use_cache:
            return await func(self, image_path, image_bytes)
        
        # Generate cache key from image
        if image_bytes is None:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
        
        cache_key = hashlib.sha256(image_bytes).hexdigest()
        
        # Check cache
        if cache_key in _analysis_cache:
            cached_result, cached_time = _analysis_cache[cache_key]
            if datetime.now() - cached_time < CACHE_DURATION:
                return cached_result
        
        # Perform analysis
        result = await func(self, image_path, image_bytes)
        
        # Store in cache
        _analysis_cache[cache_key] = (result, datetime.now())
        
        return result
    
    return wrapper


class AIService:
    """Enhanced service for AI-powered chart analysis."""
    
    def __init__(self):
        self.provider = settings.ai_provider
        self.max_retries = 3
        self.timeout = 30
        self._provider_health: Dict[str, bool] = {}
    
    @cache_result
    async def analyze_chart(
        self,
        image_path: str,
        image_bytes: Optional[bytes] = None,
        use_cache: bool = True
    ) -> AnalysisResult:
        """
        Analyze a chart image using the configured AI provider with fallback support.
        
        Args:
            image_path: Path to the image file
            image_bytes: Optional pre-loaded image bytes
            use_cache: Whether to use cached results
            
        Returns:
            AnalysisResult with patterns, bias, timing, and reasoning
        """
        start_time = time.time()
        
        # Load image if not provided
        if image_bytes is None:
            try:
                with open(image_path, "rb") as f:
                    image_bytes = f.read()
            except Exception as e:
                raise AIServiceError(f"Failed to load image: {str(e)}")
        
        # Validate image
        self._validate_image(image_bytes)
        
        # Try primary provider
        try:
            result = await self._analyze_with_provider(
                self.provider, image_path, image_bytes
            )
            self._provider_health[self.provider] = True
            
        except Exception as primary_error:
            self._provider_health[self.provider] = False
            
            # Try fallback providers
            result = await self._try_fallback_providers(
                image_path, image_bytes, primary_error
            )
        
        # Add metadata
        processing_time = int((time.time() - start_time) * 1000)
        result.processing_time_ms = processing_time
        
        return result
    
    async def _analyze_with_provider(
        self,
        provider: str,
        image_path: str,
        image_bytes: bytes
    ) -> AnalysisResult:
        """Route to specific provider with timeout."""
        try:
            async with asyncio.timeout(self.timeout):
                if provider == "openai":
                    return await self._analyze_with_openai(image_path, image_bytes)
                elif provider == "gemini":
                    return await self._analyze_with_gemini(image_path, image_bytes)
                elif provider == "anthropic":
                    return await self._analyze_with_anthropic(image_path, image_bytes)
                elif provider == "inhouse":
                    return await self._analyze_with_inhouse(image_path, image_bytes)
                else:
                    return await self._analyze_demo(image_path)
        except asyncio.TimeoutError:
            raise AIServiceError(f"{provider} provider timeout after {self.timeout}s")
    
    async def _try_fallback_providers(
        self,
        image_path: str,
        image_bytes: bytes,
        primary_error: Exception
    ) -> AnalysisResult:
        """Try fallback providers in order of preference."""
        fallback_order = ["inhouse", "openai", "gemini", "anthropic", "demo"]
        
        # Remove primary provider from fallbacks
        if self.provider in fallback_order:
            fallback_order.remove(self.provider)
        
        last_error = primary_error
        
        for fallback in fallback_order:
            try:
                result = await self._analyze_with_provider(
                    fallback, image_path, image_bytes
                )
                result.ai_provider = f"{fallback} (fallback from {self.provider})"
                return result
            except Exception as e:
                last_error = e
                continue
        
        raise AIServiceError(
            message=f"All providers failed. Last error: {str(last_error)}",
            details={
                "primary_provider": self.provider,
                "primary_error": str(primary_error),
                "last_error": str(last_error)
            }
        )
    
    def _validate_image(self, image_bytes: bytes) -> None:
        """Validate image format and size."""
        # Check size (max 10MB)
        max_size = 10 * 1024 * 1024
        if len(image_bytes) > max_size:
            raise AIServiceError(f"Image too large: {len(image_bytes)} bytes (max {max_size})")
        
        # Check minimum size
        if len(image_bytes) < 100:
            raise AIServiceError("Image too small or corrupted")
        
        # Verify it's a valid image
        valid_headers = [
            b'\xff\xd8\xff',  # JPEG
            b'\x89PNG',        # PNG
            b'GIF8',           # GIF
            b'RIFF',           # WEBP
        ]
        
        if not any(image_bytes.startswith(header) for header in valid_headers):
            raise AIServiceError("Invalid image format. Supported: JPEG, PNG, GIF, WEBP")
    
    async def _analyze_with_openai(
        self,
        image_path: str,
        image_bytes: bytes
    ) -> AnalysisResult:
        """
        Analyze chart using OpenAI GPT-4 Vision with enhanced prompts.
        Uses structured output and retry logic with exponential backoff.
        """
        try:
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI(api_key=settings.openai_api_key)
            
            # Encode to base64
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            
            # Determine image type
            ext = Path(image_path).suffix.lower()
            media_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".webp": "image/webp",
                ".gif": "image/gif",
            }.get(ext, "image/jpeg")
            
            # Retry with exponential backoff
            last_error = None
            
            for attempt in range(self.max_retries):
                try:
                    response = await client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "system",
                                "content": self._get_enhanced_system_prompt()
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": self._get_enhanced_analysis_prompt()
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{media_type};base64,{base64_image}",
                                            "detail": "high"
                                        }
                                    }
                                ]
                            }
                        ],
                        max_tokens=2000,
                        temperature=0.2,
                        response_format={"type": "json_object"},
                    )
                    
                    content = response.choices[0].message.content
                    return self._parse_ai_response(content, "openai", "gpt-4o")
                    
                except Exception as e:
                    last_error = e
                    if attempt < self.max_retries - 1:
                        wait_time = (2 ** attempt) * 0.5
                        await asyncio.sleep(wait_time)
                    continue
            
            raise last_error
            
        except ImportError:
            raise AIServiceError("OpenAI package not installed. Run: pip install openai")
        except Exception as e:
            raise AIServiceError(f"OpenAI API error: {str(e)}")
    
    async def _analyze_with_gemini(
        self,
        image_path: str,
        image_bytes: bytes
    ) -> AnalysisResult:
        """Analyze chart using Google Gemini with enhanced prompts."""
        try:
            import google.generativeai as genai
            from PIL import Image
            import io
            
            genai.configure(api_key=settings.google_ai_api_key)
            
            model = genai.GenerativeModel(
                "gemini-1.5-flash",
                generation_config=genai.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=2000,
                    response_mime_type="application/json",
                )
            )
            
            # Load image from bytes
            image = Image.open(io.BytesIO(image_bytes))
            
            prompt = f"{self._get_enhanced_system_prompt()}\n\n{self._get_enhanced_analysis_prompt()}"
            
            # Retry logic
            last_error = None
            for attempt in range(self.max_retries):
                try:
                    response = await model.generate_content_async([prompt, image])
                    content = response.text
                    return self._parse_ai_response(content, "gemini", "gemini-1.5-flash")
                except Exception as e:
                    last_error = e
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep((2 ** attempt) * 0.5)
                    continue
            
            raise last_error
            
        except ImportError:
            raise AIServiceError("Google Generative AI package not installed. Run: pip install google-generativeai pillow")
        except Exception as e:
            raise AIServiceError(f"Gemini API error: {str(e)}")
    
    async def _analyze_with_anthropic(
        self,
        image_path: str,
        image_bytes: bytes
    ) -> AnalysisResult:
        """Analyze chart using Anthropic Claude with vision."""
        try:
            import anthropic
            
            client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
            
            # Encode to base64
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            
            # Determine media type
            ext = Path(image_path).suffix.lower()
            media_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".webp": "image/webp",
                ".gif": "image/gif",
            }.get(ext, "image/jpeg")
            
            # Retry logic
            last_error = None
            for attempt in range(self.max_retries):
                try:
                    response = await client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=2000,
                        temperature=0.2,
                        system=self._get_enhanced_system_prompt(),
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": media_type,
                                            "data": base64_image,
                                        },
                                    },
                                    {
                                        "type": "text",
                                        "text": self._get_enhanced_analysis_prompt()
                                    }
                                ],
                            }
                        ],
                    )
                    
                    content = response.content[0].text
                    return self._parse_ai_response(content, "anthropic", "claude-3-5-sonnet")
                    
                except Exception as e:
                    last_error = e
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep((2 ** attempt) * 0.5)
                    continue
            
            raise last_error
            
        except ImportError:
            raise AIServiceError("Anthropic package not installed. Run: pip install anthropic")
        except Exception as e:
            raise AIServiceError(f"Anthropic API error: {str(e)}")
    
    async def _analyze_with_inhouse(
        self,
        image_path: str,
        image_bytes: bytes
    ) -> AnalysisResult:
        """
        Analyze chart using the in-house ML model.
        Enhanced with additional pattern detection logic.
        """
        from app.ml import analyze_chart_image
        
        # Run in-house analysis
        inhouse_result = analyze_chart_image(image_bytes)
        
        # Convert to API schema format with enhanced descriptions
        patterns = [
            PatternResult(
                name=detected.pattern.name,
                type=detected.pattern.bias.value,
                confidence=int(detected.confidence * 100),
                description=self._enhance_pattern_description(
                    detected.pattern.name,
                    detected.pattern.description,
                    detected.reasoning
                )
            )
            for detected in inhouse_result.patterns
        ]
        
        # Enhanced reasoning
        reasoning = self._build_enhanced_reasoning(
            patterns,
            inhouse_result.market_bias,
            inhouse_result.reasoning
        )
        
        return AnalysisResult(
            patterns=patterns,
            market_bias=inhouse_result.market_bias,
            confidence=int(inhouse_result.overall_confidence * 100),
            reasoning=reasoning,
            ai_provider="inhouse",
            ai_model="pattern-detector-v2",
            entry_timing=self._generate_entry_timing(patterns, inhouse_result.market_bias)
        )
    
    async def _analyze_demo(self, image_path: str) -> AnalysisResult:
        """
        Enhanced demo analysis with realistic patterns and timing.
        """
        await asyncio.sleep(1.2)
        
        demo_patterns = [
            PatternResult(
                name="Bullish Engulfing",
                type="bullish",
                confidence=88,
                description="Strong bullish reversal pattern where a large green candle completely engulfs the previous red candle, indicating a shift in momentum from sellers to buyers."
            ),
            PatternResult(
                name="Double Bottom",
                type="bullish",
                confidence=82,
                description="Classic W-shaped reversal pattern showing two distinct lows at approximately the same price level, with a moderate peak between them. Indicates strong support and potential trend reversal."
            ),
            PatternResult(
                name="Higher High, Higher Low",
                type="bullish",
                confidence=75,
                description="Uptrend structure with each swing high exceeding the previous high and each swing low staying above the previous low, confirming bullish momentum."
            ),
            PatternResult(
                name="Minor Resistance Zone",
                type="neutral",
                confidence=68,
                description="Price approaching a historical resistance level around the 50% Fibonacci retracement. Watch for breakout or rejection."
            ),
        ]
        
        demo_reasoning = """## Comprehensive Chart Analysis

### Executive Summary
The chart displays a strong **bullish bias** with multiple confirming signals. A double bottom pattern has formed at key support, followed by a bullish engulfing candle indicating strong buyer interest. The trend structure shows higher highs and higher lows, confirming upward momentum.

---

### Pattern Breakdown

**1. Bullish Engulfing (Confidence: 88%)**
- Located at the second bottom of the double bottom formation
- The green candle body completely engulfs the previous red candle
- Significant increase in buying volume (if visible)
- Strong reversal signal at support level

**2. Double Bottom Formation (Confidence: 82%)**
- First bottom: Tested support and bounced
- Second bottom: Re-tested same level with bullish engulfing confirmation
- Neckline resistance clearly defined
- Target projection: ~12-15% above neckline break

**3. Trend Structure (Confidence: 75%)**
- Clean sequence of higher highs and higher lows
- No lower lows since the double bottom
- Momentum appears to be accelerating

---

### Entry Strategy

**Signal Status:** **READY** (Wait for final confirmation)

**Entry Conditions:**
1. [MET] Break above resistance zone with strong candle close
2. [MET] Volume increase on breakout (at least 1.5x average)
3. [MET] Retest of broken resistance as new support (optional but safer)
4. [PENDING] Confirmation candle closing above entry zone

**Recommended Entry Zones:**
- **Aggressive:** Current level to +2% (for experienced traders)
- **Conservative:** After breakout and successful retest

**Risk Management:**
- **Stop Loss:** Below the second bottom (3-5% risk)
- **Take Profit 1:** First resistance zone (+8-10%)
- **Take Profit 2:** Measured move target (+15-18%)
- **Risk:Reward Ratio:** Approximately 1:3 to 1:4

**Timeframe:** 5-15 trading days for primary targets

---

### Key Levels to Monitor

**Support Levels:**
- **Strong Support:** Double bottom low (primary invalidation level)
- **Secondary Support:** 20-period moving average (if visible)

**Resistance Levels:**
- **Immediate:** Current minor resistance zone
- **Major:** Previous swing high
- **Target:** Measured move from double bottom

---

### Risk Factors

1. **Volume Confirmation Needed:** Ensure breakout occurs on increasing volume
2. **Market Context:** Monitor overall market conditions and sector performance
3. **False Breakout Risk:** Small cap or low volume stocks may fake out
4. **Overbought Conditions:** If RSI >70, consider waiting for pullback

---

### Final Recommendation

**Overall Bias:** Bullish (Confidence: 82%)

**Action Plan:**
1. Set price alerts at key resistance levels
2. Prepare entry order with stop loss pre-defined
3. Wait for volume confirmation on breakout
4. Consider scaling in: 50% on breakout, 50% on retest
5. Trail stop loss as price advances toward targets

**Best for:** Swing traders with 1-3 week holding period
**Risk Level:** Moderate (proper stop loss reduces downside)

---

*Disclaimer: This analysis is for educational purposes. Always conduct your own research and manage risk appropriately.*"""

        entry_timing = EntryTiming(
            signal="ready",
            timing_description="Setup is well-developed. Wait for breakout confirmation above resistance with volume.",
            conditions=[
                "Price breaks above current resistance zone",
                "Breakout candle closes strong (near high)",
                "Volume increases by at least 50% on breakout",
                "No bearish divergence on momentum indicators"
            ],
            entry_price_zone="Current price + 1-3% (on breakout confirmation)",
            stop_loss="Below double bottom low (-3.5% to -5%)",
            take_profit="TP1: +8-10% | TP2: +15-18% (measured move)",
            risk_reward="1:3 to 1:4",
            timeframe="5-15 trading days"
        )
        
        return AnalysisResult(
            patterns=demo_patterns,
            market_bias="bullish",
            confidence=82,
            reasoning=demo_reasoning,
            ai_provider="demo",
            ai_model="demo-v2-enhanced",
            entry_timing=entry_timing
        )
    
    def _enhance_pattern_description(
        self,
        name: str,
        description: str,
        reasoning: str
    ) -> str:
        """Enhance pattern description with additional context."""
        enhanced = f"{description}"
        if reasoning:
            enhanced += f" {reasoning}"
        
        # Add actionable insight based on pattern type
        pattern_insights = {
            "hammer": "Look for confirmation with next candle closing higher.",
            "shooting star": "Watch for follow-through selling in next session.",
            "doji": "Indicates indecision; wait for directional move.",
            "engulfing": "Strong reversal signal, especially at key levels.",
            "double bottom": "Measure from bottom to neckline for price target.",
            "double top": "Measure from top to neckline for downside target.",
            "head and shoulders": "Neckline break confirms pattern completion.",
        }
        
        for key, insight in pattern_insights.items():
            if key in name.lower():
                enhanced += f" {insight}"
                break
        
        return enhanced
    
    def _build_enhanced_reasoning(
        self,
        patterns: List[PatternResult],
        market_bias: str,
        base_reasoning: str
    ) -> str:
        """Build comprehensive reasoning with structure."""
        sections = []
        
        # Summary
        sections.append("## Analysis Summary\n")
        sections.append(f"Market Bias: **{market_bias.upper()}**\n")
        sections.append(f"Patterns Detected: {len(patterns)}\n")
        sections.append(f"\n{base_reasoning}\n")
        
        # Pattern details
        if patterns:
            sections.append("\n## Detected Patterns\n")
            for i, pattern in enumerate(patterns, 1):
                sections.append(
                    f"**{i}. {pattern.name}** ({pattern.type.title()}) "
                    f"- Confidence: {pattern.confidence}%\n"
                    f"   {pattern.description}\n"
                )
        
        # Risk management
        sections.append("\n## Risk Considerations\n")
        sections.append("- Always use stop losses\n")
        sections.append("- Confirm patterns with volume\n")
        sections.append("- Consider overall market context\n")
        sections.append("- Wait for candle close confirmations\n")
        
        return "\n".join(sections)
    
    def _generate_entry_timing(
        self,
        patterns: List[PatternResult],
        market_bias: str
    ) -> Optional[EntryTiming]:
        """Generate entry timing based on patterns."""
        if not patterns:
            return None
        
        # Determine signal strength
        avg_confidence = sum(p.confidence for p in patterns) / len(patterns)
        
        if avg_confidence >= 80:
            signal = "ready"
        elif avg_confidence >= 65:
            signal = "prepare"
        else:
            signal = "wait"
        
        return EntryTiming(
            signal=signal,
            timing_description=f"Based on {len(patterns)} patterns with {avg_confidence:.0f}% average confidence.",
            conditions=[
                "Confirm pattern completion",
                "Check volume for confirmation",
                "Wait for candle close above/below key level"
            ],
            timeframe="1-5 trading days"
        )
    
    def _get_enhanced_system_prompt(self) -> str:
        """Enhanced system prompt with better instructions."""
        return """You are an elite technical analyst with 20+ years of experience in pattern recognition and market timing.

Your analysis must be:
1. **Precise**: Identify exact patterns with specific names
2. **Actionable**: Provide clear entry/exit strategies
3. **Risk-Aware**: Include stop loss and position sizing guidance
4. **Comprehensive**: Cover multiple timeframes and scenarios

You MUST respond in valid JSON format with this exact structure:
{
    "patterns": [
        {
            "name": "Exact pattern name (e.g., Bullish Engulfing, Double Bottom)",
            "type": "bullish" | "bearish" | "neutral",
            "confidence": 0-100,
            "description": "Detailed description with context and implications"
        }
    ],
    "market_bias": "bullish" | "bearish" | "neutral",
    "confidence": 0-100,
    "entry_timing": {
        "signal": "wait" | "prepare" | "ready" | "now",
        "timing_description": "Specific explanation of when and why to enter",
        "conditions": ["Array of specific conditions that must be met"],
        "entry_price_zone": "Specific price range or percentage from current",
        "stop_loss": "Exact level with percentage risk",
        "take_profit": "Multiple targets with percentages",
        "risk_reward": "Calculated ratio (e.g., 1:2.5)",
        "timeframe": "Expected duration (e.g., 3-7 days, 2-4 weeks)"
    },
    "reasoning": "Comprehensive markdown-formatted analysis with sections"
}

ENTRY SIGNAL DEFINITIONS:
- **wait**: Pattern forming but not confirmed; <50% confidence; missing key confirmations
- **prepare**: Pattern developing; 50-70% confidence; watch for final confirmation triggers
- **ready**: Strong setup; 70-85% confidence; wait for specific entry trigger (breakout, retest)
- **now**: Optimal entry; >85% confidence; all conditions met; clear risk/reward

ANALYSIS REQUIREMENTS:
- Identify ALL visible candlestick patterns (minimum 2-5 patterns)
- Specify exact support/resistance levels with price zones
- Provide measured move targets based on pattern geometry
- Calculate realistic risk:reward ratios
- Consider volume, momentum, and trend context
- Format reasoning with clear sections using markdown headers
- Include both bullish AND bearish scenarios

REASONING STRUCTURE:
Use this format for the reasoning field:
## Summary
Brief overview of market state

## Pattern Analysis  
Detailed breakdown of each pattern

## Entry Strategy
Specific entry conditions and timing

## Key Levels
Support and resistance with price targets

## Risks
What could invalidate the setup

## Recommendation
Final verdict with action steps"""

    def _get_enhanced_analysis_prompt(self) -> str:
        """Enhanced analysis prompt with specific requirements."""
        return """Analyze this candlestick chart with institutional-grade precision:

REQUIRED ANALYSIS COMPONENTS:

1. **PATTERN IDENTIFICATION** (Find 3-6 patterns):
   - Candlestick patterns (Hammer, Doji, Engulfing, Stars, Marubozu, etc.)
   - Chart patterns (Double Top/Bottom, H&S, Triangles, Flags, Wedges)
   - Trend patterns (Higher Highs/Lows, Trendlines, Channels)
   - Volume patterns (if visible)
   
2. **PRICE ACTION ANALYSIS**:
   - Current trend direction and strength
   - Support/resistance levels (specify exact price zones)
   - Key pivot points
   - Fibonacci levels (if applicable)

3. **ENTRY TIMING STRATEGY**:
   Determine the EXACT conditions for entry:
   - Is the pattern complete or still forming?
   - What confirmation signals are needed? (volume, follow-through candle, indicator confirmation)
   - What is the optimal entry price zone?
   - Where should stop loss be placed? (specific level + % risk)
   - What are the take profit targets? (TP1, TP2, TP3 with percentages)
   - What is the risk:reward ratio?
   - What is the expected timeframe for this trade?
   - What could invalidate this setup?

4. **RISK ASSESSMENT**:
   - Primary risks to the setup
   - Alternative scenarios (bear case if bullish, bull case if bearish)
   - Market context considerations
   - Probability assessment

5. **COMPREHENSIVE REASONING**:
   Structure your analysis with clear markdown sections:
   - Summary (2-3 sentences)
   - Pattern Breakdown (detailed analysis of each pattern)
   - Entry Strategy (specific timing and conditions)
   - Key Levels (support, resistance, targets)
   - Risk Factors (what could go wrong)
   - Final Recommendation (action plan)

CONFIDENCE SCORING GUIDE:
- 85-100%: High conviction, multiple confirming signals, clear structure
- 70-84%: Good setup, waiting for final confirmation
- 50-69%: Developing pattern, monitor for clarity
- Below 50%: Unclear or conflicting signals

QUALITY STANDARDS:
- Be specific with price levels (not vague)
- Provide measurable targets (percentages and prices)
- Include both best-case and worst-case scenarios
- Consider multiple timeframes if visible
- Explain WHY each pattern matters
- Give actionable next steps

Return a properly formatted JSON object following the schema exactly."""

    def _parse_ai_response(
        self,
        content: str,
        provider: str,
        model: str
    ) -> AnalysisResult:
        """Enhanced parsing with better error handling and validation."""
        try:
            # Clean content
            content = content.strip()
            
            # Extract JSON from various formats
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            # Parse JSON
            data = json.loads(content)
            
            # Validate and extract patterns
            patterns = []
            for p in data.get("patterns", []):
                try:
                    patterns.append(PatternResult(
                        name=p.get("name", "Unknown Pattern"),
                        type=p.get("type", "neutral").lower(),
                        confidence=max(0, min(100, p.get("confidence", 50))),
                        description=p.get("description", "No description provided")
                    ))
                except Exception as e:
                    # Skip invalid patterns
                    continue
            
            # Extract entry timing if available
            entry_timing = None
            if "entry_timing" in data:
                try:
                    et = data["entry_timing"]
                    entry_timing = EntryTiming(
                        signal=et.get("signal", "wait"),
                        timing_description=et.get("timing_description", ""),
                        conditions=et.get("conditions", []),
                        entry_price_zone=et.get("entry_price_zone"),
                        stop_loss=et.get("stop_loss"),
                        take_profit=et.get("take_profit"),
                        risk_reward=et.get("risk_reward"),
                        timeframe=et.get("timeframe")
                    )
                except Exception:
                    pass
            
            # Validate market bias
            market_bias = data.get("market_bias", "neutral").lower()
            if market_bias not in ["bullish", "bearish", "neutral"]:
                market_bias = "neutral"
            
            # Validate confidence
            confidence = max(0, min(100, data.get("confidence", 50)))
            
            # Get reasoning
            reasoning = data.get("reasoning", "Analysis completed successfully.")
            
            return AnalysisResult(
                patterns=patterns,
                market_bias=market_bias,
                confidence=confidence,
                reasoning=reasoning,
                ai_provider=provider,
                ai_model=model,
                entry_timing=entry_timing
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # If JSON parsing fails completely, create a fallback result
            return AnalysisResult(
                patterns=[
                    PatternResult(
                        name="Analysis Error",
                        type="neutral",
                        confidence=0,
                        description=f"Failed to parse AI response: {str(e)}"
                    )
                ],
                market_bias="neutral",
                confidence=0,
                reasoning=f"**Parsing Error**\n\nThe AI response could not be parsed correctly.\n\n**Error:** {str(e)}\n\n**Raw Response Preview:**\n```\n{content[:500]}\n```",
                ai_provider=provider,
                ai_model=model,
            )
    
    async def batch_analyze(
        self,
        image_paths: List[str],
        max_concurrent: int = 3
    ) -> List[AnalysisResult]:
        """
        Analyze multiple charts concurrently with rate limiting.
        
        Args:
            image_paths: List of image file paths
            max_concurrent: Maximum number of concurrent analyses
            
        Returns:
            List of AnalysisResults
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_with_semaphore(path: str) -> AnalysisResult:
            async with semaphore:
                try:
                    return await self.analyze_chart(path)
                except Exception as e:
                    # Return error result instead of failing entire batch
                    return AnalysisResult(
                        patterns=[],
                        market_bias="neutral",
                        confidence=0,
                        reasoning=f"Failed to analyze {path}: {str(e)}",
                        ai_provider="error",
                        ai_model="error"
                    )
        
        tasks = [analyze_with_semaphore(path) for path in image_paths]
        results = await asyncio.gather(*tasks)
        
        return results
    
    def clear_cache(self) -> int:
        """Clear the analysis cache and return number of items cleared."""
        count = len(_analysis_cache)
        _analysis_cache.clear()
        return count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        now = datetime.now()
        valid_count = sum(
            1 for _, cached_time in _analysis_cache.values()
            if now - cached_time < CACHE_DURATION
        )
        
        return {
            "total_cached": len(_analysis_cache),
            "valid_cached": valid_count,
            "expired_cached": len(_analysis_cache) - valid_count,
            "cache_duration_hours": CACHE_DURATION.total_seconds() / 3600,
            "provider_health": self._provider_health
        }
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all providers."""
        return {
            "current_provider": self.provider,
            "provider_health": self._provider_health,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the current provider.
        
        Returns:
            Status information including provider availability
        """
        try:
            # Try a minimal test (you could use a small test image)
            status = {
                "provider": self.provider,
                "healthy": True,
                "message": f"{self.provider} provider is configured",
                "timestamp": datetime.now().isoformat()
            }
            
            # Check if API keys are configured
            if self.provider == "openai":
                if not settings.openai_api_key:
                    status["healthy"] = False
                    status["message"] = "OpenAI API key not configured"
            elif self.provider == "gemini":
                if not settings.google_ai_api_key:
                    status["healthy"] = False
                    status["message"] = "Google AI API key not configured"
            elif self.provider == "anthropic":
                if not settings.anthropic_api_key:
                    status["healthy"] = False
                    status["message"] = "Anthropic API key not configured"
            
            return status
            
        except Exception as e:
            return {
                "provider": self.provider,
                "healthy": False,
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }


# Singleton instance
ai_service = AIService()


# Utility functions for external use
async def analyze_chart_with_ai(
    image_path: str,
    image_bytes: Optional[bytes] = None,
    use_cache: bool = True
) -> AnalysisResult:
    """
    Convenience function to analyze a chart image with AI.
    
    Args:
        image_path: Path to the image file
        image_bytes: Optional pre-loaded image bytes
        use_cache: Whether to use cached results
        
    Returns:
        AnalysisResult with comprehensive analysis
    """
    return await ai_service.analyze_chart(image_path, image_bytes, use_cache)


async def analyze_multiple_charts(
    image_paths: List[str],
    max_concurrent: int = 3
) -> List[AnalysisResult]:
    """
    Analyze multiple charts concurrently.
    
    Args:
        image_paths: List of image file paths
        max_concurrent: Maximum concurrent analyses
        
    Returns:
        List of AnalysisResults
    """
    return await ai_service.batch_analyze(image_paths, max_concurrent)


def get_analysis_stats() -> Dict[str, Any]:
    """Get current analysis service statistics."""
    return {
        "cache_stats": ai_service.get_cache_stats(),
        "provider_status": ai_service.get_provider_status()
    }


def clear_analysis_cache() -> int:
    """Clear the analysis cache."""
    return ai_service.clear_cache()


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Single analysis
        result = await analyze_chart_with_ai("path/to/chart.png")
        print(f"Market Bias: {result.market_bias}")
        print(f"Confidence: {result.confidence}%")
        print(f"Patterns Found: {len(result.patterns)}")
        
        if result.entry_timing:
            print(f"Entry Signal: {result.entry_timing.signal}")
            print(f"Risk:Reward: {result.entry_timing.risk_reward}")
        
        print(f"\nReasoning:\n{result.reasoning}")
        
        # Batch analysis
        charts = ["chart1.png", "chart2.png", "chart3.png"]
        results = await analyze_multiple_charts(charts)
        print(f"\nAnalyzed {len(results)} charts")
        
        # Stats
        stats = get_analysis_stats()
        print(f"\nCache Stats: {stats['cache_stats']}")
    
    asyncio.run(main())