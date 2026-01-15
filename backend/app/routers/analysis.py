"""
Analysis Router

Handles chart image upload and analysis operations.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query
from sqlalchemy.ext.asyncio import AsyncSession
import math

from app.database import get_db
from app.schemas.analysis import (
    AnalysisResponse,
    AnalysisListResponse,
    AnalysisUploadResponse,
    PatternResult,
)
from app.services.analysis_service import AnalysisService
from app.core.dependencies import get_current_user
from app.core.exceptions import NotFoundError, ValidationError, AIServiceError
from app.models.user import User


router = APIRouter(prefix="/analysis", tags=["Analysis"])


@router.post("/upload", response_model=AnalysisUploadResponse)
async def upload_and_analyze(
    file: UploadFile = File(..., description="Chart image to analyze"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Upload a chart image and get AI-powered analysis.
    
    Supported formats: JPG, JPEG, PNG, WebP
    Max file size: 10MB
    
    Returns detected patterns, market bias, confidence score, and detailed reasoning.
    """
    try:
        # Read file content
        content = await file.read()
        
        # Analyze chart
        analysis_service = AnalysisService(db)
        analysis = await analysis_service.analyze_chart(
            user=current_user,
            file_content=content,
            filename=file.filename or "chart.png"
        )
        
        # Convert to response
        response = AnalysisResponse(
            id=analysis.id,
            image_url=analysis.image_url,
            patterns=[PatternResult(**p) for p in (analysis.patterns or [])],
            market_bias=analysis.market_bias,
            confidence=analysis.confidence,
            reasoning=analysis.reasoning,
            status=analysis.status,
            created_at=analysis.created_at,
            ai_provider=analysis.ai_provider,
            processing_time_ms=analysis.processing_time_ms,
        )
        
        return AnalysisUploadResponse(
            message="Analysis completed successfully",
            analysis=response
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message
        )
    except AIServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"AI analysis failed: {e.message}"
        )


@router.get("/history", response_model=AnalysisListResponse)
async def get_analysis_history(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=50, description="Items per page"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get authenticated user's analysis history.
    
    Paginated results ordered by most recent first.
    """
    analysis_service = AnalysisService(db)
    analyses, total = await analysis_service.get_user_history(
        user=current_user,
        page=page,
        page_size=page_size
    )
    
    items = [
        AnalysisResponse(
            id=a.id,
            image_url=a.image_url,
            patterns=[PatternResult(**p) for p in (a.patterns or [])],
            market_bias=a.market_bias,
            confidence=a.confidence,
            reasoning=a.reasoning,
            status=a.status,
            created_at=a.created_at,
            ai_provider=a.ai_provider,
            processing_time_ms=a.processing_time_ms,
        )
        for a in analyses
    ]
    
    return AnalysisListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        pages=math.ceil(total / page_size) if total > 0 else 0
    )


@router.get("/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(
    analysis_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get a specific analysis by ID.
    
    Only returns analyses owned by the authenticated user.
    """
    try:
        analysis_service = AnalysisService(db)
        analysis = await analysis_service.get_analysis(analysis_id, current_user)
        
        return AnalysisResponse(
            id=analysis.id,
            image_url=analysis.image_url,
            patterns=[PatternResult(**p) for p in (analysis.patterns or [])],
            market_bias=analysis.market_bias,
            confidence=analysis.confidence,
            reasoning=analysis.reasoning,
            status=analysis.status,
            created_at=analysis.created_at,
            ai_provider=analysis.ai_provider,
            processing_time_ms=analysis.processing_time_ms,
        )
        
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message
        )


@router.delete("/{analysis_id}")
async def delete_analysis(
    analysis_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete an analysis and its associated image.
    
    Only the owner can delete their analyses.
    """
    try:
        analysis_service = AnalysisService(db)
        await analysis_service.delete_analysis(analysis_id, current_user)
        
        return {
            "message": "Analysis deleted successfully",
            "analysis_id": analysis_id
        }
        
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message
        )
