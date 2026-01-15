"""
Storage Service with S3 Support

Provides abstracted file storage with support for:
- Local filesystem (development)
- AWS S3 / S3-compatible storage (production)

Production deployments should use S3 to persist files across deployments
since platforms like Render and Railway use ephemeral filesystems.
"""

import os
import uuid
import asyncio
import aiofiles
import logging
from pathlib import Path
from typing import Optional, Tuple, BinaryIO, AsyncIterator
from datetime import datetime, timezone
from io import BytesIO

from app.config import settings
from app.core.exceptions import StorageError

logger = logging.getLogger(__name__)


# Chunk size for streaming (64KB)
STREAM_CHUNK_SIZE = 64 * 1024


class StorageService:
    """
    Abstracted storage service supporting local and S3 backends.
    
    For production cloud deployments:
    - Set STORAGE_TYPE=s3
    - Configure AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME
    - Optionally set S3_ENDPOINT_URL for S3-compatible services (MinIO, DigitalOcean Spaces)
    """
    
    def __init__(self):
        self.storage_type = settings.storage_type
        self.upload_dir = Path(settings.upload_dir)
        
        # S3 client (lazy initialization)
        self._s3_client = None
        self._s3_bucket = getattr(settings, 's3_bucket_name', None)
        self._s3_region = getattr(settings, 's3_region', 'us-east-1')
        self._s3_endpoint = getattr(settings, 's3_endpoint_url', None)
        
        # Ensure upload directory exists for local storage
        if self.storage_type == "local":
            self.upload_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Local storage initialized at {self.upload_dir}")
        elif self.storage_type == "s3":
            logger.info(f"S3 storage configured for bucket: {self._s3_bucket}")
    
    def _get_s3_client(self):
        """Lazy initialization of S3 client."""
        if self._s3_client is None:
            try:
                import boto3
                from botocore.config import Config
                
                config = Config(
                    retries={'max_attempts': 3, 'mode': 'adaptive'},
                    connect_timeout=5,
                    read_timeout=30
                )
                
                if self._s3_endpoint:
                    # S3-compatible service (MinIO, DigitalOcean Spaces, etc.)
                    self._s3_client = boto3.client(
                        's3',
                        endpoint_url=self._s3_endpoint,
                        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                        region_name=self._s3_region,
                        config=config
                    )
                else:
                    # Standard AWS S3
                    self._s3_client = boto3.client('s3', config=config)
                    
            except ImportError:
                raise StorageError(
                    message="boto3 is required for S3 storage. Install with: pip install boto3"
                )
        return self._s3_client
    
    def generate_filename(self, original_filename: str) -> str:
        """Generate a unique filename for storage."""
        ext = Path(original_filename).suffix.lower()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
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
            Tuple of (file_path/key, file_url)
        """
        if self.storage_type == "s3":
            return await self._save_s3(content, filename, subfolder)
        else:
            return await self._save_local(content, filename, subfolder)
    
    async def save_file_streaming(
        self,
        file_stream: BinaryIO,
        filename: str,
        subfolder: Optional[str] = None,
        content_length: Optional[int] = None
    ) -> Tuple[str, str]:
        """
        Save file using streaming to minimize memory usage.
        
        Args:
            file_stream: File-like object to read from
            filename: Filename to save as
            subfolder: Optional subfolder
            content_length: Total file size (optional, for S3 multipart)
            
        Returns:
            Tuple of (file_path/key, file_url)
        """
        if self.storage_type == "s3":
            return await self._save_s3_streaming(file_stream, filename, subfolder, content_length)
        else:
            return await self._save_local_streaming(file_stream, filename, subfolder)
    
    async def _save_local(
        self,
        content: bytes,
        filename: str,
        subfolder: Optional[str] = None
    ) -> Tuple[str, str]:
        """Save file to local filesystem."""
        try:
            save_dir = self.upload_dir / subfolder if subfolder else self.upload_dir
            save_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = save_dir / filename
            
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(content)
            
            relative_path = str(file_path.relative_to(self.upload_dir))
            file_url = f"/uploads/{relative_path.replace(os.sep, '/')}"
            
            return str(file_path), file_url
            
        except Exception as e:
            logger.error(f"Failed to save file locally: {e}")
            raise StorageError(message=f"Failed to save file: {str(e)}")
    
    async def _save_local_streaming(
        self,
        file_stream: BinaryIO,
        filename: str,
        subfolder: Optional[str] = None
    ) -> Tuple[str, str]:
        """Save file to local filesystem using streaming."""
        try:
            save_dir = self.upload_dir / subfolder if subfolder else self.upload_dir
            save_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = save_dir / filename
            
            async with aiofiles.open(file_path, "wb") as f:
                while True:
                    chunk = file_stream.read(STREAM_CHUNK_SIZE)
                    if not chunk:
                        break
                    await f.write(chunk)
            
            relative_path = str(file_path.relative_to(self.upload_dir))
            file_url = f"/uploads/{relative_path.replace(os.sep, '/')}"
            
            return str(file_path), file_url
            
        except Exception as e:
            logger.error(f"Failed to stream file locally: {e}")
            raise StorageError(message=f"Failed to save file: {str(e)}")
    
    async def _save_s3(
        self,
        content: bytes,
        filename: str,
        subfolder: Optional[str] = None
    ) -> Tuple[str, str]:
        """Save file to S3 bucket."""
        try:
            s3 = self._get_s3_client()
            
            # Build S3 key
            key = f"{subfolder}/{filename}" if subfolder else filename
            
            # Run upload in thread pool (boto3 is synchronous)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: s3.put_object(
                    Bucket=self._s3_bucket,
                    Key=key,
                    Body=content,
                    ContentType=self._get_content_type(filename)
                )
            )
            
            # Generate URL
            if self._s3_endpoint:
                file_url = f"{self._s3_endpoint}/{self._s3_bucket}/{key}"
            else:
                file_url = f"https://{self._s3_bucket}.s3.{self._s3_region}.amazonaws.com/{key}"
            
            return key, file_url
            
        except Exception as e:
            logger.error(f"Failed to upload to S3: {e}")
            raise StorageError(message=f"S3 upload failed: {str(e)}")
    
    async def _save_s3_streaming(
        self,
        file_stream: BinaryIO,
        filename: str,
        subfolder: Optional[str] = None,
        content_length: Optional[int] = None
    ) -> Tuple[str, str]:
        """Save file to S3 using multipart upload for large files."""
        try:
            s3 = self._get_s3_client()
            key = f"{subfolder}/{filename}" if subfolder else filename
            
            # Use multipart upload for streaming
            loop = asyncio.get_event_loop()
            
            if content_length and content_length > 5 * 1024 * 1024:  # > 5MB
                # Multipart upload
                mp = await loop.run_in_executor(
                    None,
                    lambda: s3.create_multipart_upload(
                        Bucket=self._s3_bucket,
                        Key=key,
                        ContentType=self._get_content_type(filename)
                    )
                )
                upload_id = mp['UploadId']
                parts = []
                part_number = 1
                
                try:
                    while True:
                        chunk = file_stream.read(5 * 1024 * 1024)  # 5MB chunks
                        if not chunk:
                            break
                        
                        response = await loop.run_in_executor(
                            None,
                            lambda c=chunk, p=part_number: s3.upload_part(
                                Bucket=self._s3_bucket,
                                Key=key,
                                UploadId=upload_id,
                                PartNumber=p,
                                Body=c
                            )
                        )
                        parts.append({'PartNumber': part_number, 'ETag': response['ETag']})
                        part_number += 1
                    
                    # Complete multipart upload
                    await loop.run_in_executor(
                        None,
                        lambda: s3.complete_multipart_upload(
                            Bucket=self._s3_bucket,
                            Key=key,
                            UploadId=upload_id,
                            MultipartUpload={'Parts': parts}
                        )
                    )
                except Exception:
                    # Abort on failure
                    await loop.run_in_executor(
                        None,
                        lambda: s3.abort_multipart_upload(
                            Bucket=self._s3_bucket,
                            Key=key,
                            UploadId=upload_id
                        )
                    )
                    raise
            else:
                # Small file - read all and upload
                content = file_stream.read()
                return await self._save_s3(content, filename, subfolder)
            
            # Generate URL
            if self._s3_endpoint:
                file_url = f"{self._s3_endpoint}/{self._s3_bucket}/{key}"
            else:
                file_url = f"https://{self._s3_bucket}.s3.{self._s3_region}.amazonaws.com/{key}"
            
            return key, file_url
            
        except Exception as e:
            logger.error(f"Failed to stream upload to S3: {e}")
            raise StorageError(message=f"S3 streaming upload failed: {str(e)}")
    
    async def delete_file(self, file_path: str) -> bool:
        """Delete file from storage."""
        if self.storage_type == "s3":
            return await self._delete_s3(file_path)
        else:
            return await self._delete_local(file_path)
    
    async def _delete_local(self, file_path: str) -> bool:
        """Delete file from local filesystem."""
        try:
            path = Path(file_path)
            if path.exists():
                os.remove(path)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete local file: {e}")
            raise StorageError(message=f"Failed to delete file: {str(e)}")
    
    async def _delete_s3(self, key: str) -> bool:
        """Delete file from S3."""
        try:
            s3 = self._get_s3_client()
            loop = asyncio.get_event_loop()
            
            await loop.run_in_executor(
                None,
                lambda: s3.delete_object(Bucket=self._s3_bucket, Key=key)
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete S3 object: {e}")
            raise StorageError(message=f"S3 delete failed: {str(e)}")
    
    async def get_file(self, file_path: str) -> Optional[bytes]:
        """Read file from storage."""
        if self.storage_type == "s3":
            return await self._get_s3(file_path)
        else:
            return await self._get_local(file_path)
    
    async def _get_local(self, file_path: str) -> Optional[bytes]:
        """Read file from local filesystem."""
        try:
            path = Path(file_path)
            if not path.exists():
                return None
            
            async with aiofiles.open(path, "rb") as f:
                return await f.read()
        except Exception as e:
            logger.error(f"Failed to read local file: {e}")
            raise StorageError(message=f"Failed to read file: {str(e)}")
    
    async def _get_s3(self, key: str) -> Optional[bytes]:
        """Read file from S3."""
        try:
            s3 = self._get_s3_client()
            loop = asyncio.get_event_loop()
            
            response = await loop.run_in_executor(
                None,
                lambda: s3.get_object(Bucket=self._s3_bucket, Key=key)
            )
            return response['Body'].read()
            
        except s3.exceptions.NoSuchKey:
            return None
        except Exception as e:
            logger.error(f"Failed to get S3 object: {e}")
            raise StorageError(message=f"S3 get failed: {str(e)}")
    
    def _get_content_type(self, filename: str) -> str:
        """Get MIME type for file."""
        ext = Path(filename).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.webp': 'image/webp',
            '.gif': 'image/gif',
        }
        return mime_types.get(ext, 'application/octet-stream')


# Singleton instance
storage_service = StorageService()
