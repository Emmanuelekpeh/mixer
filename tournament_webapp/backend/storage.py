from pathlib import Path
import os
import shutil
from typing import Optional

# Conditional import for boto3 – only required for S3 backend
try:
    import boto3  # type: ignore
    from botocore.exceptions import ClientError  # type: ignore
except ImportError:  # boto3 not installed or not needed
    boto3 = None  # type: ignore


class StorageBackend:
    """Abstract base class for storage back-ends (local volume, S3, etc.)."""

    def save(self, source: str, dest_name: str) -> str:
        """Persist the file and return absolute path or URL."""
        raise NotImplementedError

    def url_for(self, path_or_key: str) -> str:
        """Return a public URL (or relative API path) for the stored object."""
        raise NotImplementedError


class LocalFileStorage(StorageBackend):
    """Simple volume-based storage (e.g., Railway volume)."""

    def __init__(self, root_dir: Optional[str] = None):
        self.root_dir = Path(root_dir or os.getenv("STORAGE_ROOT", "/app/processed_audio"))
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def save(self, source: str, dest_name: str) -> str:
        dest = self.root_dir / dest_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(source, dest)
        return str(dest)

    def url_for(self, path_or_key: str) -> str:
        # For local storage we expose via static files under /processed_audio
        filename = Path(path_or_key).name
        return f"/processed_audio/{filename}"


class S3Storage(StorageBackend):
    """S3-compatible storage backend (AWS S3, Cloudflare R2, MinIO)."""

    def __init__(self):
        if boto3 is None:
            raise RuntimeError("boto3 is required for S3 backend; install with pip install boto3")

        self.bucket = os.getenv("S3_BUCKET")
        if not self.bucket:
            raise RuntimeError("S3_BUCKET environment variable not set")

        region = os.getenv("AWS_REGION", "us-east-1")
        endpoint_url = os.getenv("S3_ENDPOINT_URL")  # allow custom endpoints (e.g., R2)
        self.s3 = boto3.client(
            "s3",
            region_name=region,
            endpoint_url=endpoint_url,
        )

    def save(self, source: str, dest_name: str) -> str:
        key = dest_name.replace("\\", "/")
        try:
            self.s3.upload_file(source, self.bucket, key, ExtraArgs={"ACL": "public-read"})
        except ClientError as exc:
            raise RuntimeError(f"S3 upload failed: {exc}")
        # After upload, optionally delete local file
        os.remove(source)
        return key

    def url_for(self, path_or_key: str) -> str:
        # If custom endpoint_url is used, construct URL; else default AWS pattern
        endpoint_url = os.getenv("S3_PUBLIC_ENDPOINT") or os.getenv("S3_ENDPOINT_URL")
        if endpoint_url:
            return f"{endpoint_url.rstrip('/')}/{self.bucket}/{path_or_key}"
        # Default AWS S3 URL
        return f"https://{self.bucket}.s3.amazonaws.com/{path_or_key}"


_default_backend: Optional[StorageBackend] = None


def get_storage() -> StorageBackend:
    global _default_backend
    if _default_backend is None:
        backend_name = os.getenv("STORAGE_BACKEND", "local")
        if backend_name == "local":
            _default_backend = LocalFileStorage()
        elif backend_name == "s3":
            _default_backend = S3Storage()
        else:
            raise RuntimeError(f"Unsupported STORAGE_BACKEND: {backend_name}")
    return _default_backend 