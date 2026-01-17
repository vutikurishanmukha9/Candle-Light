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
        Demo analysis with honest, conservative patterns and methodology.
        Serves as an example of proper AI output format.
        """
        await asyncio.sleep(1.2)
        
        # Conservative patterns with proper methodology including new features
        demo_patterns = [
            PatternResult(
                name="Higher Low in Uptrend",
                type="bullish",
                confidence=62,
                description="Context: Within established uptrend (HH/HL structure). Evidence: Recent swing low higher than previous. Wick analysis: Lower wick visible at low (+5% for buyer defense). Confidence: 50 + 15 (with trend) + 5 (wick support) - 5 (no volume) - 3 (minor ambiguity) = 62%"
            ),
            PatternResult(
                name="Range Support Test",
                type="neutral",
                confidence=48,
                description="Context: Price at bottom of visible range. Evidence: Horizontal bounce zone. Wick analysis: Upper wick on last green candle (-5% for resistance rejection). Confidence: 50 + 5 (support zone) - 5 (upper wick) - 2 (needs confirmation) = 48%"
            ),
        ]
        
        demo_reasoning = """## Summary
The chart shows a **probable uptrend** based on higher highs and higher lows structure. A pullback to support is being tested with some bullish wick rejection signals, but volume confirmation is not visible.

---

## Context Analysis

- Dominant Trend: UPTREND - Higher high and higher low structure visible
- Evidence: Recent swing high exceeded previous high, no lower lows visible
- Context weight: +15% (analysis aligns with trend)
- Pattern Alignment: WITH trend (bullish patterns in uptrend = higher probability)

---

## Pattern Analysis

**1. Higher Low in Uptrend (62% confidence)**

- Context: Forms within established uptrend structure
- Evidence: Most recent swing low is visibly higher than prior low
- Wick Analysis: Lower wick at the low shows buyer defense (+5%)
- Alternative: Could be start of sideways consolidation
- Confidence: 50 (base) + 15 (with trend) + 5 (wick support) - 5 (no volume) - 3 (ambiguity) = 62%

**2. Range Support Test (48% confidence)**

- Context: Price testing bottom of visible trading range
- Evidence: Horizontal area with multiple touches
- Wick Analysis: Upper wick on last green candle suggests resistance (-5%)
- Alternative: Could break support if momentum fails
- Confidence: 50 (base) + 5 (support zone) - 5 (upper wick) - 2 (needs confirmation) = 48%

---

## Volume Assessment

- Visible: No - Volume bars not visible at bottom of chart
- Confidence adjustment: -5% applied
- Breakout conviction: Cannot confirm without volume data
- Recommendation: If volume becomes visible, look for rising volume on green candles

---

## Entry Strategy

**Signal: PREPARE** (Not ready yet)

**Why this timing (not earlier/later):**
- Earlier (now): Rejected - Pattern within trend but needs breakout confirmation
- Later (after retest): Alternative - More conservative but may miss initial move
- Chosen (breakout): Balances confirmation with opportunity

**Entry conditions needed:**
- Break above recent swing high
- Breakout candle closes in upper 25% of its range (no large upper wick)
- If volume visible: Must show increase on breakout

---

## Projections (Range-Based)

**Method used:** Range Height Projection
- Visible range height: approximately 3-4 candles worth of movement
- Projection: Break above range top, target = range height projected upward
- Target with uncertainty: Prior swing high or range height projection (±20%)

**Alternative method:** Prior Impulse
- Prior impulse leg visible: approximately equal to range height
- 0.5x to 1x projection = conservative to moderate target

---

## Risks & Limitations

- Counter-trend risks: Not applicable (analysis aligned with trend)
- Volume: Not visible - cannot confirm breakout conviction
- Price scale: Not visible - levels are approximate
- Timeframe: Unknown - affects expected duration
- Wick warning: Upper wick on recent candle suggests some resistance. Watch for repeated rejections.

Disclaimer: Educational analysis. Trend context is most important - individual candles matter less than overall structure."""

        entry_timing = EntryTiming(
            signal="prepare",
            timing_description="Pattern aligned with uptrend but needs breakout confirmation. Why: Trend is bullish (most important), but entry timing requires break above resistance to confirm.",
            conditions=[
                "Price breaks above recent swing high",
                "Breakout candle closes in upper 25% (small upper wick)",
                "No immediate bearish engulfing after break"
            ],
            entry_price_zone="At or slightly above recent swing high (breakout level)",
            stop_loss="Below recent higher low (structure invalidation)",
            take_profit="Range height projection or prior swing structure (±20% uncertainty)",
            risk_reward="Approximately 1:2 based on range measurement",
            timeframe="Depends on chart timeframe - structure suggests swing trade"
        )
        
        return AnalysisResult(
            patterns=demo_patterns,
            market_bias="bullish",
            confidence=55,  # Conservative overall confidence
            reasoning=demo_reasoning,
            ai_provider="demo",
            ai_model="demo-v3-honest",
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
        """Enhanced system prompt with validation, scoring, and transparency."""
        return """You are a cautious, methodical technical analyst who prioritizes accuracy over quantity.

CORE PRINCIPLES:
1. **Honest**: Only report what you can clearly see and verify
2. **Conservative**: When uncertain, lower confidence or omit
3. **Transparent**: Show HOW you reached each conclusion
4. **Humble**: Acknowledge limitations (no volume, unclear candles, etc.)

You MUST respond in valid JSON format with this structure:
{
    "patterns": [
        {
            "name": "Pattern name",
            "type": "bullish" | "bearish" | "neutral",
            "confidence": 0-85,
            "description": "Why this pattern, what evidence, what alternatives considered"
        }
    ],
    "market_bias": "bullish" | "bearish" | "neutral",
    "confidence": 0-85,
    "entry_timing": {
        "signal": "wait" | "prepare" | "ready" | "now",
        "timing_description": "Specific explanation with methodology",
        "conditions": ["Conditions that must be met"],
        "entry_price_zone": "Range with uncertainty (e.g., 'approximately +2-4%')",
        "stop_loss": "Level with range",
        "take_profit": "Targets as ranges",
        "risk_reward": "Calculated ratio",
        "timeframe": "Expected duration"
    },
    "reasoning": "Markdown analysis showing methodology"
}

=== CONTEXT HIERARCHY (CRITICAL) ===
Always analyze in this order - higher levels override lower:
1. TREND (most important): What is the dominant trend direction?
2. PATTERN (secondary): What patterns form within that trend context?
3. CANDLE (least important): What do individual candles suggest?

A bullish candle in a downtrend is WEAK. A bearish candle in an uptrend is often just a pullback.
NEVER let a single candle pattern override clear trend structure.

=== WICK ANALYSIS ===
Wicks reveal rejection and failed moves:

UPPER WICK near resistance:
- Long upper wick = sellers rejected the move = BEARISH signal
- Apply -10% confidence penalty if bullish pattern has long upper wick
- "Upper wick rejection visible at resistance level"

LOWER WICK near support:
- Long lower wick = buyers defended the level = BULLISH signal  
- Apply +5% confidence boost if lower wick at support
- "Lower wick shows buyer defense at support"

Wick ratio: If wick is >50% of candle range, it's significant.

=== CONFIDENCE SCORING (SHOW YOUR MATH) ===
Start at 50% and calculate:

TREND CONTEXT (highest weight):
+15% Analysis aligns with dominant trend
-20% Analysis AGAINST dominant trend (counter-trend)

PATTERN QUALITY:
+10% Pattern matches textbook definition exactly
+5%  Pattern at key support/resistance level
-15% Pattern is ambiguous or partially formed
-10% Conflicting patterns present

WICK ANALYSIS:
+5%  Supportive wick rejection (lower wick at support for bullish)
-10% Adverse wick rejection (upper wick at resistance for bullish)

VOLUME (if visible):
+10% Rising volume on breakout / pattern completion
+5%  Volume confirms pattern (high on reversal candle)
-5%  Declining volume on breakout (weak conviction)
-5%  Volume not visible in chart

VISIBILITY:
-10% Candles unclear or low image quality

MAXIMUM: 85% (nothing is certain in markets)

Example: "Confidence: 50 + 15 (with trend) + 10 (clear pattern) - 10 (upper wick rejection) - 5 (no volume) = 60%"

=== VOLUME ANALYSIS (if visible) ===
Step 1: Check "Are volume bars visible at bottom of chart?"

If YES, analyze properly:
- Rising volume on green candles = bullish conviction (+10% confidence)
- Rising volume on breakout = confirmation (+10% confidence)
- Declining volume on rally = weak move, be cautious (-5% confidence)
- Volume spike on reversal candle = strong signal (+5% confidence)

If NO:
- State "Volume: Not visible in this chart"
- Apply -5% confidence penalty
- NEVER fabricate volume observations

=== PROJECTION METHODOLOGY (Range-Based) ===
DO NOT use arbitrary percentages. Use these methods:

METHOD 1: RANGE HEIGHT PROJECTION
- Measure the consolidation/pattern range height
- Project that height from breakout point
- "Target: Range height (~X candles) projected from break = approximately [level]"

METHOD 2: PRIOR IMPULSE PROJECTION  
- Measure the prior impulse move (last strong leg)
- Apply 0.5x to 1x of that move as target
- "Target: Prior impulse was ~Y range, projecting 0.5-1x = approximately [level]"

METHOD 3: SWING STRUCTURE
- Identify prior swing highs/lows as targets
- "Target: Prior swing high visible at approximately [level]"

ALWAYS add ±20% uncertainty to any projection.
NEVER state exact percentages without visible price scale.

=== METHODOLOGY TRANSPARENCY ===
For EVERY conclusion, explain WHY:

Context Assessment:
"Trend: [direction] based on [HH/HL or LH/LL structure]
Pattern within context: [aligned/counter to trend]
Weight applied: [+15% with trend OR -20% counter-trend]"

Wick Analysis:
"Wick observation: [upper/lower wick at key level]
Significance: [>50% of range = significant]
Confidence adjustment: [+5% or -10%]"

Volume Assessment:
"Volume visible: [Yes/No]
If visible: [rising/falling on what type of candles]
Confidence adjustment: [+10%, +5%, -5%, etc.]"

Projection Method:
"Method used: [range height / prior impulse / swing structure]
Measurement: [how I estimated it]
Target with uncertainty: [range ±20%]"

=== REASONING STRUCTURE ===
## Summary
Brief overview (2-3 sentences)

## Context Analysis (FIRST)
- Dominant trend direction
- Why: [HH/HL or LH/LL structure visible]
- Pattern alignment: with or against trend

## Pattern Analysis
For each pattern:
- Evidence (what candles)
- Wick analysis (rejection signals)
- Confidence calculation (show math with trend/wick/volume factors)
- Alternative interpretations considered

## Volume Assessment
- Visible: Yes/No
- If yes: What it shows
- Confidence adjustment applied

## Entry Strategy
- Recommendation with methodology
- Why this timing over alternatives

## Projections
- Method used (range/impulse/swing)
- Target with ±20% uncertainty

## Risks & Limitations
- Counter-trend risks
- What data is missing"""

    def _get_enhanced_analysis_prompt(self) -> str:
        """Enhanced analysis prompt with conservative, honest requirements."""
        return """Analyze this candlestick chart carefully and honestly:

CRITICAL RULES:
1. Report ONLY patterns you can CLEARLY see (1-3 maximum)
2. If unsure, say "uncertain" or omit the pattern
3. Never fabricate data (volume, indicators, exact prices)
4. Use ranges and uncertainty language ("approximately", "possibly")

=== ANALYSIS STEPS ===

STEP 1: VISUAL ASSESSMENT
First, describe what you can actually see:
- How many candles are visible?
- Is volume displayed at the bottom? (Yes/No)
- Is a price scale visible? (Yes/No)
- What is the overall trend direction?

STEP 2: PATTERN IDENTIFICATION (Be conservative)
Look for 1-3 CLEAR patterns only. For each:
- Name the pattern
- Cite which candles form it
- Check validation criteria (see system prompt)
- Consider alternatives: "This could also be..."
- If ambiguous, reduce confidence or skip

STEP 3: CONFIDENCE CALCULATION (Show your work)
Start at 50% base confidence:
- Add points for clear evidence
- Subtract points for uncertainty
- Show the math: "50 + 15 - 10 = 55%"
- Maximum 85% (markets are never certain)

STEP 4: ENTRY ASSESSMENT
- Is the pattern complete or forming?
- What confirmation is needed?
- Give ranges, not exact numbers
- Acknowledge what you cannot see

STEP 5: LIMITATIONS (Be honest)
State clearly:
- "Volume: Not visible in chart" (if true)
- "Price scale: Not visible" (if true)
- "Pattern: Partially formed / needs confirmation"
- "Uncertainty: [specific concerns]"

=== OUTPUT REQUIREMENTS ===

Pattern descriptions must include:
"Evidence: [what I see]
Alternative interpretation: [what else this could be]
Why I chose this: [reasoning]
Confidence: [calculation]"

Entry timing must include:
"Why this timing: [reasoning]
Alternatives considered: [1, 2, 3]
Why alternatives rejected: [reasons]"

=== DO NOT ===
- Claim volume confirmation if no volume bars visible
- Give exact percentages (use ranges)
- Report more than 3 patterns
- Confidence above 85%
- Fabricate indicator data (RSI, MACD, etc.)

Return a properly formatted JSON object."""

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