"""
Custom Exceptions

Application-specific exceptions for consistent error handling.
"""

from typing import Optional, Dict, Any


class AppException(Exception):
    """Base application exception."""
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class AuthenticationError(AppException):
    """Raised when authentication fails."""
    
    def __init__(
        self,
        message: str = "Authentication failed",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, status_code=401, details=details)


class AuthorizationError(AppException):
    """Raised when user doesn't have permission."""
    
    def __init__(
        self,
        message: str = "Not authorized to perform this action",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, status_code=403, details=details)


class NotFoundError(AppException):
    """Raised when a resource is not found."""
    
    def __init__(
        self,
        message: str = "Resource not found",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, status_code=404, details=details)


class ValidationError(AppException):
    """Raised when validation fails."""
    
    def __init__(
        self,
        message: str = "Validation error",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, status_code=422, details=details)


class AIServiceError(AppException):
    """Raised when AI service fails."""
    
    def __init__(
        self,
        message: str = "AI service error",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, status_code=503, details=details)


class StorageError(AppException):
    """Raised when storage operations fail."""
    
    def __init__(
        self,
        message: str = "Storage error",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, status_code=500, details=details)
