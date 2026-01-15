"""
Rate Limiting Middleware

Provides rate limiting for API endpoints to prevent abuse.
Uses in-memory storage (suitable for single-instance deployments).
For production with multiple instances, use Redis-based storage.
"""

from datetime import datetime, timezone
from collections import defaultdict
from typing import Dict, Tuple
import asyncio

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.config import settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using sliding window algorithm.
    
    Limits requests per IP address with configurable rate and burst.
    """
    
    def __init__(self, app, rate_per_minute: int = 60, burst: int = 10):
        super().__init__(app)
        self.rate_per_minute = rate_per_minute
        self.burst = burst
        self.requests: Dict[str, list] = defaultdict(list)
        self._lock = asyncio.Lock()
        
        # Paths exempt from rate limiting
        self.exempt_paths = {
            "/health",
            "/health/ai",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/",
        }
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with rate limiting."""
        
        # Skip rate limiting for exempt paths
        if request.url.path in self.exempt_paths:
            return await call_next(request)
        
        # Get client IP
        client_ip = self._get_client_ip(request)
        
        # Check rate limit
        async with self._lock:
            is_allowed, retry_after = self._check_rate_limit(client_ip)
        
        if not is_allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too many requests",
                    "detail": "Rate limit exceeded. Please try again later.",
                    "retry_after": retry_after
                },
                headers={"Retry-After": str(retry_after)}
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        async with self._lock:
            remaining = self._get_remaining_requests(client_ip)
        
        response.headers["X-RateLimit-Limit"] = str(self.rate_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(60)
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        # Check for forwarded header (behind proxy)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        # Check for real IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to direct client IP
        return request.client.host if request.client else "unknown"
    
    def _check_rate_limit(self, client_ip: str) -> Tuple[bool, int]:
        """
        Check if request is within rate limit.
        
        Returns:
            Tuple of (is_allowed, retry_after_seconds)
        """
        now = datetime.now(timezone.utc).timestamp()
        window_start = now - 60  # 1-minute window
        
        # Clean old requests
        self.requests[client_ip] = [
            ts for ts in self.requests[client_ip] 
            if ts > window_start
        ]
        
        # Check burst limit (requests in last second)
        recent_requests = [
            ts for ts in self.requests[client_ip]
            if ts > now - 1
        ]
        if len(recent_requests) >= self.burst:
            return False, 1
        
        # Check rate limit
        if len(self.requests[client_ip]) >= self.rate_per_minute:
            oldest = min(self.requests[client_ip])
            retry_after = int(oldest + 60 - now) + 1
            return False, max(retry_after, 1)
        
        # Allow request and record
        self.requests[client_ip].append(now)
        return True, 0
    
    def _get_remaining_requests(self, client_ip: str) -> int:
        """Get remaining requests in current window."""
        return max(0, self.rate_per_minute - len(self.requests[client_ip]))
    
    async def cleanup_old_entries(self):
        """Periodically clean up old entries to prevent memory leaks."""
        while True:
            await asyncio.sleep(300)  # Every 5 minutes
            async with self._lock:
                now = datetime.now(timezone.utc).timestamp()
                window_start = now - 60
                for ip in list(self.requests.keys()):
                    self.requests[ip] = [
                        ts for ts in self.requests[ip]
                        if ts > window_start
                    ]
                    if not self.requests[ip]:
                        del self.requests[ip]
