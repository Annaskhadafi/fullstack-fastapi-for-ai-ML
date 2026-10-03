import os
import uuid
import logging
from typing import Optional, Dict, Any
import boto3
from botocore.config import Config
from app.core.config import settings

logger = logging.getLogger(__name__)

LOCAL_UPLOAD_DIR = os.path.join("app", "static", "uploads")
os.makedirs(LOCAL_UPLOAD_DIR, exist_ok=True)


class StorageService:
    """
    Unified Object Storage Service compatible with:
    - AWS S3
    - Cloudflare R2
    - MinIO / DigitalOcean Spaces / Wasabi
    - Local Filesystem fallback
    """

    def __init__(self):
        self._s3_client = None

    def is_s3_configured(self) -> bool:
        """Checks if S3 / Cloudflare R2 credentials are set."""
        return bool(
            settings.S3_ACCESS_KEY_ID
            and settings.S3_SECRET_ACCESS_KEY
            and settings.S3_BUCKET_NAME
        )

    def get_client(self):
        """Lazily creates and returns a boto3 S3 client configured for AWS or Cloudflare R2."""
        if self._s3_client is None and self.is_s3_configured():
            try:
                boto_config = Config(
                    signature_version="s3v4",
                    retries={"max_attempts": 3, "mode": "standard"}
                )

                kwargs = {
                    "service_name": "s3",
                    "aws_access_key_id": settings.S3_ACCESS_KEY_ID,
                    "aws_secret_access_key": settings.S3_SECRET_ACCESS_KEY,
                    "config": boto_config,
                }

                if settings.S3_ENDPOINT_URL:
                    kwargs["endpoint_url"] = settings.S3_ENDPOINT_URL

                if settings.S3_REGION_NAME and settings.S3_REGION_NAME != "auto":
                    kwargs["region_name"] = settings.S3_REGION_NAME
                else:
                    kwargs["region_name"] = "auto"  # Cloudflare R2 uses 'auto'

                self._s3_client = boto3.client(**kwargs)
                logger.info("Initialized S3/Cloudflare R2 storage client.")
            except Exception as e:
                logger.warning(f"Failed to initialize S3 client: {e}. Falling back to local storage.")
                self._s3_client = None

        return self._s3_client

    async def upload_file(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str = "application/octet-stream",
        folder: str = "uploads"
    ) -> Dict[str, Any]:
        """
        Uploads file to S3/Cloudflare R2 if configured, or saves to local disk.
        Returns a dictionary with file URL, key, and storage backend used.
        """
        # Generate unique file key
        ext = os.path.splitext(filename)[1].lower()
        unique_name = f"{uuid.uuid4().hex}{ext}"
        file_key = f"{folder}/{unique_name}" if folder else unique_name

        client = self.get_client()

        # 1. Upload to S3 / Cloudflare R2
        if client is not None:
            try:
                client.put_object(
                    Bucket=settings.S3_BUCKET_NAME,
                    Key=file_key,
                    Body=file_bytes,
                    ContentType=content_type
                )

                if settings.S3_PUBLIC_DOMAIN:
                    public_domain = settings.S3_PUBLIC_DOMAIN.rstrip("/")
                    url = f"{public_domain}/{file_key}"
                elif settings.S3_ENDPOINT_URL:
                    url = f"{settings.S3_ENDPOINT_URL.rstrip('/')}/{settings.S3_BUCKET_NAME}/{file_key}"
                else:
                    url = f"https://{settings.S3_BUCKET_NAME}.s3.amazonaws.com/{file_key}"

                logger.info(f"File uploaded to S3/R2: {file_key}")
                return {
                    "success": True,
                    "backend": "s3/r2",
                    "file_key": file_key,
                    "url": url,
                    "filename": filename,
                    "size_bytes": len(file_bytes)
                }
            except Exception as e:
                logger.error(f"S3/R2 upload failed: {e}. Falling back to local storage.")

        # 2. Fallback: Local Filesystem Storage
        local_folder = os.path.join("app", "static", folder)
        os.makedirs(local_folder, exist_ok=True)
        local_path = os.path.join(local_folder, unique_name)

        with open(local_path, "wb") as f:
            f.write(file_bytes)

        local_url = f"/static/{folder}/{unique_name}"
        return {
            "success": True,
            "backend": "local",
            "file_key": file_key,
            "url": local_url,
            "filename": filename,
            "size_bytes": len(file_bytes)
        }

    async def delete_file(self, file_key: str) -> bool:
        """Deletes file from S3/R2 or local filesystem."""
        client = self.get_client()
        if client is not None:
            try:
                client.delete_object(Bucket=settings.S3_BUCKET_NAME, Key=file_key)
                return True
            except Exception as e:
                logger.warning(f"Failed to delete {file_key} from S3/R2: {e}")

        # Local delete fallback
        local_path = os.path.join("app", "static", file_key)
        if os.path.exists(local_path):
            try:
                os.remove(local_path)
                return True
            except Exception:
                return False

        return False

    def get_storage_info(self) -> Dict[str, Any]:
        """Returns active storage backend status."""
        configured = self.is_s3_configured()
        is_r2 = bool(settings.S3_ENDPOINT_URL and "r2.cloudflarestorage.com" in settings.S3_ENDPOINT_URL)
        return {
            "backend": "Cloudflare R2" if is_r2 else ("AWS S3" if configured else "Local Filesystem"),
            "is_cloud_active": configured,
            "bucket": settings.S3_BUCKET_NAME if configured else "local",
            "public_domain": settings.S3_PUBLIC_DOMAIN or "default"
        }


storage_service = StorageService()
