"""
Analysis Service

Orchestrates the chart analysis workflow.
"""

from typing import List, Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func
from PIL import Image
import io

from app.models.analysis import Analysis
from app.models.user import User
from app.schemas.analysis import AnalysisResult, AnalysisResponse, PatternResult
from app.services.ai_service import ai_service
from app.services.storage_service import storage_service
from app.core.exceptions import NotFoundError, ValidationError


# Allowed image types
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB


class AnalysisService:
    """Service for chart analysis operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def analyze_chart(
        self,
        user: User,
        file_content: bytes,
        filename: str
    ) -> Analysis:
        """
        Analyze a chart image.
        
        Args:
            user: The user requesting analysis
            file_content: Image file content
            filename: Original filename
            
        Returns:
            Analysis object with results
        """
        # Validate file
        await self._validate_image(file_content, filename)
        
        # Get image dimensions
        width, height = self._get_image_dimensions(file_content)
        
        # Generate unique filename and save
        new_filename = storage_service.generate_filename(filename)
        file_path, file_url = await storage_service.save_file(
            file_content,
            new_filename,
            subfolder=user.id
        )
        
        # Create analysis record
        analysis = Analysis(
            user_id=user.id,
            image_filename=new_filename,
            image_path=file_path,
            image_url=file_url,
            original_filename=filename,
            image_size=len(file_content),
            image_width=width,
            image_height=height,
            status="processing"
        )
        
        self.db.add(analysis)
        await self.db.commit()
        
        try:
            # Run AI analysis
            result = await ai_service.analyze_chart(file_path, file_content)
            
            # Update analysis with results
            analysis.patterns = [p.model_dump() for p in result.patterns]
            analysis.market_bias = result.market_bias
            analysis.confidence = result.confidence
            analysis.reasoning = result.reasoning
            analysis.ai_provider = result.ai_provider
            analysis.ai_model = result.ai_model
            analysis.processing_time_ms = result.processing_time_ms
            analysis.status = "completed"
            
            await self.db.commit()
            await self.db.refresh(analysis)
            
            return analysis
            
        except Exception as e:
            # Update status to failed
            analysis.status = "failed"
            analysis.error_message = str(e)
            await self.db.commit()
            raise
    
    async def get_analysis(
        self,
        analysis_id: str,
        user: User
    ) -> Analysis:
        """
        Get a specific analysis by ID.
        
        Args:
            analysis_id: The analysis ID
            user: The requesting user
            
        Returns:
            Analysis object
            
        Raises:
            NotFoundError: If analysis not found or not owned by user
        """
        result = await self.db.execute(
            select(Analysis).where(
                Analysis.id == analysis_id,
                Analysis.user_id == user.id
            )
        )
        analysis = result.scalar_one_or_none()
        
        if not analysis:
            raise NotFoundError(
                message="Analysis not found",
                details={"analysis_id": analysis_id}
            )
        
        return analysis
    
    async def get_user_history(
        self,
        user: User,
        page: int = 1,
        page_size: int = 10
    ) -> tuple[List[Analysis], int]:
        """
        Get user's analysis history with pagination.
        
        Args:
            user: The user
            page: Page number (1-indexed)
            page_size: Items per page
            
        Returns:
            Tuple of (list of analyses, total count)
        """
        # Get total count
        count_result = await self.db.execute(
            select(func.count(Analysis.id)).where(
                Analysis.user_id == user.id
            )
        )
        total = count_result.scalar()
        
        # Get paginated results
        offset = (page - 1) * page_size
        result = await self.db.execute(
            select(Analysis)
            .where(Analysis.user_id == user.id)
            .order_by(desc(Analysis.created_at))
            .offset(offset)
            .limit(page_size)
        )
        analyses = result.scalars().all()
        
        return list(analyses), total
    
    async def delete_analysis(
        self,
        analysis_id: str,
        user: User
    ) -> bool:
        """
        Delete an analysis.
        
        Args:
            analysis_id: The analysis ID
            user: The requesting user
            
        Returns:
            True if deleted
            
        Raises:
            NotFoundError: If analysis not found
        """
        analysis = await self.get_analysis(analysis_id, user)
        
        # Delete file
        await storage_service.delete_file(analysis.image_path)
        
        # Delete record
        await self.db.delete(analysis)
        await self.db.commit()
        
        return True
    
    async def _validate_image(
        self,
        content: bytes,
        filename: str
    ) -> None:
        """Validate uploaded image."""
        from pathlib import Path
        
        # Check file size
        if len(content) > MAX_IMAGE_SIZE:
            raise ValidationError(
                message="Image too large",
                details={"max_size_mb": MAX_IMAGE_SIZE / (1024 * 1024)}
            )
        
        # Check extension
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValidationError(
                message="Invalid file type",
                details={"allowed": list(ALLOWED_EXTENSIONS)}
            )
        
        # Validate it's a real image
        try:
            img = Image.open(io.BytesIO(content))
            img.verify()
        except Exception:
            raise ValidationError(
                message="Invalid or corrupted image file"
            )
    
    def _get_image_dimensions(self, content: bytes) -> tuple[int, int]:
        """Get image width and height."""
        try:
            img = Image.open(io.BytesIO(content))
            return img.size
        except Exception:
            return (0, 0)
