"""
Storage Service

Handles file storage operations (local and S3).
"""

import os
import uuid
import aiofiles
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime

from app.config import settings
from app.core.exceptions import StorageError


class StorageService:
    """Service for file storage operations."""
    
    def __init__(self):
        self.storage_type = settings.storage_type
        self.upload_dir = Path(settings.upload_dir)
        
        # Ensure upload directory exists
        if self.storage_type == "local":
            self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_filename(self, original_filename: str) -> str:
        """
        Generate a unique filename for storage.
        
        Args:
            original_filename: Original file name
            
        Returns:
            Unique filename with timestamp and UUID
        """
        ext = Path(original_filename).suffix.lower()
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"{timestamp}_{unique_id}{ext}"
    
    async def save_file(
        self,
        content: bytes,
        filename: str,
        subfolder: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Save file to storage.
        
        Args:
            content: File content as bytes
            filename: Filename to save as
            subfolder: Optional subfolder (e.g., user_id)
            
        Returns:
            Tuple of (file_path, file_url)
            
        Raises:
            StorageError: If save fails
        """
        if self.storage_type == "local":
            return await self._save_local(content, filename, subfolder)
        else:
            # S3 implementation for future
            raise StorageError("S3 storage not implemented yet")
    
    async def _save_local(
        self,
        content: bytes,
        filename: str,
        subfolder: Optional[str] = None
    ) -> Tuple[str, str]:
        """Save file to local filesystem."""
        try:
            # Create subfolder if specified
            if subfolder:
                save_dir = self.upload_dir / subfolder
                save_dir.mkdir(parents=True, exist_ok=True)
            else:
                save_dir = self.upload_dir
            
            file_path = save_dir / filename
            
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(content)
            
            # Generate URL path
            relative_path = str(file_path.relative_to(self.upload_dir))
            file_url = f"/uploads/{relative_path.replace(os.sep, '/')}"
            
            return str(file_path), file_url
            
        except Exception as e:
            raise StorageError(
                message=f"Failed to save file: {str(e)}",
                details={"filename": filename}
            )
    
    async def delete_file(self, file_path: str) -> bool:
        """
        Delete file from storage.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if deleted, False if not found
        """
        if self.storage_type == "local":
            return await self._delete_local(file_path)
        else:
            raise StorageError("S3 storage not implemented yet")
    
    async def _delete_local(self, file_path: str) -> bool:
        """Delete file from local filesystem."""
        try:
            path = Path(file_path)
            if path.exists():
                os.remove(path)
                return True
            return False
        except Exception as e:
            raise StorageError(
                message=f"Failed to delete file: {str(e)}",
                details={"file_path": file_path}
            )
    
    async def get_file(self, file_path: str) -> Optional[bytes]:
        """
        Read file from storage.
        
        Args:
            file_path: Path to file
            
        Returns:
            File content as bytes, or None if not found
        """
        if self.storage_type == "local":
            return await self._get_local(file_path)
        else:
            raise StorageError("S3 storage not implemented yet")
    
    async def _get_local(self, file_path: str) -> Optional[bytes]:
        """Read file from local filesystem."""
        try:
            path = Path(file_path)
            if not path.exists():
                return None
            
            async with aiofiles.open(path, "rb") as f:
                return await f.read()
        except Exception as e:
            raise StorageError(
                message=f"Failed to read file: {str(e)}",
                details={"file_path": file_path}
            )


# Singleton instance
storage_service = StorageService()
