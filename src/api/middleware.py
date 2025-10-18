# File: src/api/middleware.py
"""
FastAPI Middleware
Logging, rate limiting, and monitoring middleware
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time
import logging
from typing import Callable, Dict
import json
from datetime import datetime, timedelta
from collections import defaultdict, deque

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class LoggingMiddleware(BaseHTTPMiddleware):
    """Request/response logging middleware"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with logging"""
        start_time = time.time()
        
        # Log request
        logger.info(
            f"🔄 {request.method} {request.url.path} - "
            f"Client: {request.client.host} - "
            f"User-Agent: {request.headers.get('user-agent', 'unknown')}"
        )
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            f"✅ {request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.3f}s"
        )
        
        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting middleware"""
    
    def __init__(self, app, calls: int = 100, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.clients = defaultdict(deque)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting"""
        client_ip = request.client.host
        now = datetime.utcnow()
        
        # Clean old requests
        cutoff = now - timedelta(seconds=self.period)
        client_requests = self.clients[client_ip]
        
        while client_requests and client_requests[0] < cutoff:
            client_requests.popleft()
        
        # Check rate limit
        if len(client_requests) >= self.calls:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return Response(
                content=json.dumps({
                    "error": True,
                    "message": f"Rate limit exceeded: {self.calls} calls per {self.period} seconds",
                    "retry_after": self.period
                }),
                status_code=429,
                headers={"Content-Type": "application/json"}
            )
        
        # Add current request
        client_requests.append(now)
        
        # Process request
        response = await call_next(request)
        return response

class MonitoringMiddleware(BaseHTTPMiddleware):
    """Performance monitoring middleware"""
    
    def __init__(self, app):
        super().__init__(app)
        self.request_count = 0
        self.error_count = 0
        self.total_time = 0.0
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with monitoring"""
        start_time = time.time()
        self.request_count += 1
        
        try:
            response = await call_next(request)
            
            # Track timing
            process_time = time.time() - start_time
            self.total_time += process_time
            
            # Track errors
            if response.status_code >= 400:
                self.error_count += 1
            
            return response
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Request processing error: {e}")
            raise
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics"""
        return {
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "average_response_time": self.total_time / max(self.request_count, 1)
        }