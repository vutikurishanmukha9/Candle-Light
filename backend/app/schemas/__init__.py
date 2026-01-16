"""Pydantic Schemas Package"""

from app.schemas.user import (
    UserCreate,
    UserUpdate,
    UserResponse,
    UserInDB,
    UserPreferences,
    UserWithPreferences,
    OAuthProvider,
    OAuthProvidersResponse,
    OAuthCallbackResponse,
)
from app.schemas.auth import (
    Token,
    TokenPayload,
    LoginRequest,
    RegisterRequest,
    RefreshTokenRequest,
)
from app.schemas.analysis import (
    PatternResult,
    AnalysisCreate,
    AnalysisResult,
    AnalysisResponse,
    AnalysisListResponse,
    AnalysisUploadResponse,
    ExportFormat,
    ExportResponse,
    HistoryExportRequest,
)

__all__ = [
    # User
    "UserCreate",
    "UserUpdate", 
    "UserResponse",
    "UserInDB",
    "UserPreferences",
    "UserWithPreferences",
    "OAuthProvider",
    "OAuthProvidersResponse",
    "OAuthCallbackResponse",
    # Auth
    "Token",
    "TokenPayload",
    "LoginRequest",
    "RegisterRequest",
    "RefreshTokenRequest",
    # Analysis
    "PatternResult",
    "AnalysisCreate",
    "AnalysisResult",
    "AnalysisResponse",
    "AnalysisListResponse",
    "AnalysisUploadResponse",
    "ExportFormat",
    "ExportResponse",
    "HistoryExportRequest",
]

