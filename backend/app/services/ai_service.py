"""
AI Service

Handles AI-powered chart analysis using OpenAI, Gemini, In-House model, or demo mode.
Provides a unified interface for different AI providers.
"""

import base64
import json
import time
from typing import Optional, List, Dict, Any
from pathlib import Path

from app.config import settings
from app.schemas.analysis import PatternResult, AnalysisResult
from app.core.exceptions import AIServiceError


class AIService:
    """Service for AI-powered chart analysis."""
    
    def __init__(self):
        self.provider = settings.ai_provider
    
    async def analyze_chart(
        self,
        image_path: str,
        image_bytes: Optional[bytes] = None
    ) -> AnalysisResult:
        """
        Analyze a chart image using the configured AI provider.
        
        Args:
            image_path: Path to the image file
            image_bytes: Optional pre-loaded image bytes
            
        Returns:
            AnalysisResult with patterns, bias, and reasoning
        """
        start_time = time.time()
        
        try:
            if self.provider == "openai":
                result = await self._analyze_with_openai(image_path, image_bytes)
            elif self.provider == "gemini":
                result = await self._analyze_with_gemini(image_path, image_bytes)
            elif self.provider == "inhouse":
                result = await self._analyze_with_inhouse(image_path, image_bytes)
            else:
                result = await self._analyze_demo(image_path)
            
            # Add processing time
            processing_time = int((time.time() - start_time) * 1000)
            result.processing_time_ms = processing_time
            
            return result
            
        except Exception as e:
            # Fallback to in-house model if external AI fails
            if self.provider in ["openai", "gemini"]:
                try:
                    result = await self._analyze_with_inhouse(image_path, image_bytes)
                    processing_time = int((time.time() - start_time) * 1000)
                    result.processing_time_ms = processing_time
                    result.ai_provider = f"inhouse (fallback from {self.provider})"
                    return result
                except:
                    pass
            
            raise AIServiceError(
                message=f"Chart analysis failed: {str(e)}",
                details={"provider": self.provider}
            )
    
    async def _analyze_with_inhouse(
        self,
        image_path: str,
        image_bytes: Optional[bytes] = None
    ) -> AnalysisResult:
        """
        Analyze chart using the in-house ML model.
        
        This provides pattern detection without external AI dependencies.
        """
        from app.ml import analyze_chart_image
        
        # Load image if not provided
        if image_bytes is None:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
        
        # Run in-house analysis
        inhouse_result = analyze_chart_image(image_bytes)
        
        # Convert to API schema format
        patterns = [
            PatternResult(
                name=detected.pattern.name,
                type=detected.pattern.bias.value,
                confidence=int(detected.confidence * 100),
                description=f"{detected.pattern.description} {detected.reasoning}"
            )
            for detected in inhouse_result.patterns
        ]
        
        return AnalysisResult(
            patterns=patterns,
            market_bias=inhouse_result.market_bias,
            confidence=int(inhouse_result.overall_confidence * 100),
            reasoning=inhouse_result.reasoning,
            ai_provider="inhouse",
            ai_model="pattern-detector-v1",
        )

    
    async def _analyze_with_openai(
        self,
        image_path: str,
        image_bytes: Optional[bytes] = None
    ) -> AnalysisResult:
        """
        Analyze chart using OpenAI GPT-4 Vision with structured JSON output.
        
        Uses response_format for guaranteed JSON schema compliance.
        Includes retry logic with exponential backoff.
        """
        try:
            from openai import AsyncOpenAI
            import asyncio
            
            client = AsyncOpenAI(api_key=settings.openai_api_key)
            
            # Load image if not provided
            if image_bytes is None:
                with open(image_path, "rb") as f:
                    image_bytes = f.read()
            
            # Encode to base64
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            
            # Determine image type
            ext = Path(image_path).suffix.lower()
            media_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".webp": "image/webp",
            }.get(ext, "image/jpeg")
            
            # Retry with exponential backoff
            max_retries = 3
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    response = await client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "system",
                                "content": self._get_system_prompt()
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": self._get_analysis_prompt()
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
                        max_tokens=1500,
                        temperature=0.3,
                        # STRUCTURED OUTPUT: Guarantees valid JSON response
                        response_format={"type": "json_object"},
                    )
                    
                    content = response.choices[0].message.content
                    return self._parse_ai_response(content, "openai", "gpt-4o")
                    
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 0.5  # 0.5s, 1s, 2s
                        await asyncio.sleep(wait_time)
                    continue
            
            raise last_error
            
        except ImportError:
            raise AIServiceError("OpenAI package not installed")
        except Exception as e:
            raise AIServiceError(f"OpenAI API error: {str(e)}")
    
    async def _analyze_with_gemini(
        self,
        image_path: str,
        image_bytes: Optional[bytes] = None
    ) -> AnalysisResult:
        """Analyze chart using Google Gemini."""
        try:
            import google.generativeai as genai
            from PIL import Image
            
            genai.configure(api_key=settings.google_ai_api_key)
            
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            # Load image
            image = Image.open(image_path)
            
            prompt = f"{self._get_system_prompt()}\n\n{self._get_analysis_prompt()}"
            
            response = await model.generate_content_async(
                [prompt, image],
                generation_config=genai.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=1500,
                )
            )
            
            content = response.text
            return self._parse_ai_response(content, "gemini", "gemini-1.5-flash")
            
        except ImportError:
            raise AIServiceError("Google Generative AI package not installed")
        except Exception as e:
            raise AIServiceError(f"Gemini API error: {str(e)}")
    
    async def _analyze_demo(self, image_path: str) -> AnalysisResult:
        """
        Return demo analysis result.
        Used when no AI API key is configured.
        """
        # Simulate processing time
        import asyncio
        await asyncio.sleep(1.5)
        
        # Demo patterns
        demo_patterns = [
            PatternResult(
                name="Double Bottom",
                type="bullish",
                confidence=85,
                description="A classic reversal pattern indicating potential upward movement. The price has tested a support level twice and bounced."
            ),
            PatternResult(
                name="Hammer",
                type="bullish",
                confidence=78,
                description="A bullish candlestick pattern showing strong buying pressure. The long lower wick indicates buyers stepping in."
            ),
            PatternResult(
                name="Rising Wedge",
                type="bearish",
                confidence=65,
                description="A bearish continuation pattern. While price is rising, the narrowing range suggests weakening momentum."
            ),
        ]
        
        demo_reasoning = """**Chart Analysis Summary**

Based on the uploaded chart, I've identified several key patterns and market signals:

**Primary Pattern: Double Bottom Formation**
The chart shows a clear double bottom pattern near the support level. This is a classic bullish reversal signal, indicating that sellers have attempted to push the price lower twice but failed, with buyers stepping in at similar price levels.

**Supporting Signals:**
1. A hammer candlestick has formed at the second bottom, showing strong buying pressure
2. Volume appears to be increasing on the upward moves
3. The RSI (if visible) would likely show bullish divergence

**Caution:**
A rising wedge pattern is forming on the shorter timeframe, which could limit upside potential in the near term. Watch for a breakout above the wedge's upper boundary for confirmation.

**Key Levels to Watch:**
- Support: Current bottom level
- Resistance: Wedge upper boundary and previous swing highs

**Recommendation:** 
The overall bias is bullish, but wait for confirmation before entering. A break above the wedge with volume would be ideal."""

        return AnalysisResult(
            patterns=demo_patterns,
            market_bias="bullish",
            confidence=78,
            reasoning=demo_reasoning,
            ai_provider="demo",
            ai_model="demo-v1",
        )
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for AI analysis."""
        return """You are an expert technical analyst specializing in candlestick chart pattern recognition. 
Your task is to analyze trading charts and provide detailed, actionable insights.

You must respond in valid JSON format with the following structure:
{
    "patterns": [
        {
            "name": "Pattern Name",
            "type": "bullish" | "bearish" | "neutral",
            "confidence": 0-100,
            "description": "Brief description of the pattern"
        }
    ],
    "market_bias": "bullish" | "bearish" | "neutral",
    "confidence": 0-100,
    "reasoning": "Detailed markdown-formatted analysis explaining your findings"
}

Be specific about pattern locations and provide actionable insights.
Always include key support/resistance levels when visible.
Format the reasoning with headers and bullet points for readability."""

    def _get_analysis_prompt(self) -> str:
        """Get the user prompt for chart analysis."""
        return """Analyze this candlestick chart image and identify:

1. All visible candlestick patterns (e.g., Hammer, Doji, Engulfing, Double Top/Bottom, Head and Shoulders)
2. The overall market bias (bullish, bearish, or neutral)
3. Your confidence level (0-100) in the analysis
4. Detailed reasoning explaining your analysis

Consider:
- Recent price action and trend direction
- Key support and resistance levels
- Volume patterns if visible
- Any divergences or confirmations

Respond with a JSON object following the specified format."""

    def _parse_ai_response(
        self,
        content: str,
        provider: str,
        model: str
    ) -> AnalysisResult:
        """Parse AI response into AnalysisResult."""
        try:
            # Try to extract JSON from the response
            content = content.strip()
            
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            data = json.loads(content)
            
            patterns = [
                PatternResult(
                    name=p.get("name", "Unknown"),
                    type=p.get("type", "neutral"),
                    confidence=p.get("confidence", 50),
                    description=p.get("description")
                )
                for p in data.get("patterns", [])
            ]
            
            return AnalysisResult(
                patterns=patterns,
                market_bias=data.get("market_bias", "neutral"),
                confidence=data.get("confidence", 50),
                reasoning=data.get("reasoning"),
                ai_provider=provider,
                ai_model=model,
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            # If parsing fails, create a minimal result
            return AnalysisResult(
                patterns=[],
                market_bias="neutral",
                confidence=0,
                reasoning=f"Analysis completed but response parsing failed. Raw response: {content[:500]}",
                ai_provider=provider,
                ai_model=model,
            )


# Singleton instance
ai_service = AIService()
