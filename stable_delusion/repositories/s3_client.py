"""
S3 client utilities for AWS integration.
Provides reusable S3 client setup and common operations.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

import logging
from typing import Optional, Dict, Any, Tuple, TYPE_CHECKING

from stable_delusion.config import Config
from stable_delusion.exceptions import ConfigurationError, FileOperationError

try:
    import boto3
    from botocore.config import Config as BotocoreConfig
    from botocore.exceptions import ClientError, NoCredentialsError

    BOTO3_AVAILABLE = True
except ImportError as e:
    logging.warning("AWS SDK not available: %s", e)
    boto3 = None  # type: ignore[assignment]
    BOTO3_AVAILABLE = False

    # Create dummy exception classes to avoid import errors
    class ClientError(Exception):  # type: ignore[misc,no-redef]
        """Dummy ClientError class when boto3 is not available."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.response = {"Error": {"Code": "ImportError"}}

    class NoCredentialsError(Exception):  # type: ignore[misc,no-redef]
        """Dummy NoCredentialsError class when boto3 is not available."""

    BotocoreConfig = None  # type: ignore[misc,assignment]

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client


class S3ClientManager:
    """Manages S3 client creation and configuration."""

    @staticmethod
    def _check_boto3_availability() -> None:
        if not BOTO3_AVAILABLE:
            raise ConfigurationError(
                "AWS SDK (boto3) is not installed. Install with: pip install boto3",
                config_key="boto3",
            )

    @staticmethod
    def _build_s3_client_config(config: Config) -> Tuple[Any, Dict[str, Any]]:
        # Configure boto3 client settings
        boto_config = BotocoreConfig(
            region_name=config.s3_region,
            retries={"max_attempts": 3, "mode": "standard"},
            max_pool_connections=10,
        )

        # Create client with explicit credentials if provided
        client_kwargs: Dict[str, Any] = {"service_name": "s3", "config": boto_config}

        # Use explicit credentials if provided, otherwise rely on AWS default credential chain
        if config.aws_access_key_id and config.aws_secret_access_key:
            client_kwargs.update(
                {
                    "aws_access_key_id": config.aws_access_key_id,
                    "aws_secret_access_key": config.aws_secret_access_key,
                }
            )

        return boto_config, client_kwargs

    @staticmethod
    def _create_and_validate_client(
        client_kwargs: Dict[str, Any], bucket_name: Optional[str]
    ) -> "S3Client":
        s3_client = boto3.client(**client_kwargs)  # type: ignore[misc]
        S3ClientManager._validate_s3_access(s3_client, bucket_name)
        return s3_client

    @staticmethod
    def create_s3_client(config: Config) -> "S3Client":
        """
        Create and configure an S3 client.

        Args:
            config: Application configuration containing S3 settings

        Returns:
            Configured boto3 S3 client

        Raises:
            ConfigurationError: If S3 configuration is invalid
            FileOperationError: If S3 client creation fails
        """
        S3ClientManager._check_boto3_availability()

        try:
            _, client_kwargs = S3ClientManager._build_s3_client_config(config)
            return S3ClientManager._create_and_validate_client(client_kwargs, config.s3_bucket)

        except NoCredentialsError as e:
            raise ConfigurationError(
                "AWS credentials not found. Configure credentials using AWS CLI, "
                "environment variables, or IAM roles. See: "
                "https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html",
                config_key="AWS_CREDENTIALS",
            ) from e

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            raise FileOperationError(
                f"Failed to create S3 client: {error_code} - {str(e)}",
                operation="s3_client_creation",
            ) from e

    @staticmethod
    def _validate_s3_access(s3_client: "S3Client", bucket_name: Optional[str]) -> None:
        """
        Validate S3 access by checking bucket accessibility.

        Args:
            s3_client: Configured S3 client
            bucket_name: S3 bucket name to validate

        Raises:
            ConfigurationError: If bucket is not accessible
            FileOperationError: If validation fails
        """
        S3ClientManager._validate_bucket_name_provided(bucket_name)
        # After validation, bucket_name is guaranteed to not be None
        if bucket_name is None:
            raise ConfigurationError("S3 bucket name validation failed: bucket_name is None")
        S3ClientManager._perform_bucket_access_check(s3_client, bucket_name)

    @staticmethod
    def _validate_bucket_name_provided(bucket_name: Optional[str]) -> None:
        """Validate that bucket name is provided."""
        if not bucket_name:
            raise ConfigurationError(
                "S3 bucket name is required for S3 storage", config_key="AWS_S3_BUCKET"
            )

    @staticmethod
    def _perform_bucket_access_check(s3_client: "S3Client", bucket_name: str) -> None:
        """Perform actual bucket access validation."""
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            logging.info("S3 bucket '%s' is accessible", bucket_name)
        except ClientError as e:
            S3ClientManager._handle_bucket_access_error(e, bucket_name)

    @staticmethod
    def _handle_bucket_access_error(error: ClientError, bucket_name: str) -> None:
        """Handle bucket access validation errors."""
        error_code = error.response.get("Error", {}).get("Code", "Unknown")

        if error_code == "NoSuchBucket":
            raise ConfigurationError(
                f"S3 bucket '{bucket_name}' does not exist or is not accessible",
                config_key="AWS_S3_BUCKET",
            ) from error

        if error_code in ["AccessDenied", "Forbidden"]:
            raise ConfigurationError(
                f"Access denied to S3 bucket '{bucket_name}'. Check IAM permissions.",
                config_key="AWS_CREDENTIALS",
            ) from error

        raise FileOperationError(
            f"Failed to validate S3 bucket access: {error_code} - {str(error)}",
            operation="s3_bucket_validation",
        ) from error


def generate_s3_key(file_path: str, prefix: Optional[str] = None) -> str:
    """
    Generate S3 object key from file path.

    Args:
        file_path: Local file path or filename
        prefix: Optional prefix for the S3 key

    Returns:
        S3 object key string
    """
    # Remove any leading slashes and normalize path separators
    clean_path = file_path.lstrip("/").replace("\\", "/")

    if prefix:
        # Ensure prefix ends with / if it doesn't already
        normalized_prefix = prefix.rstrip("/") + "/"
        return f"{normalized_prefix}{clean_path}"

    return clean_path


def parse_s3_url(s3_url: str) -> Tuple[str, str]:
    """
    Parse S3 URL to extract bucket and key.

    Args:
        s3_url: S3 URL in format s3://bucket/key

    Returns:
        Tuple of (bucket_name, object_key)

    Raises:
        ValueError: If URL format is invalid
    """
    if not s3_url.startswith("s3://"):
        raise ValueError(f"Invalid S3 URL format: {s3_url}")

    # Remove s3:// prefix and split on first /
    path = s3_url[5:]  # Remove 's3://'
    parts = path.split("/", 1)

    if len(parts) != 2:
        raise ValueError(f"Invalid S3 URL format: {s3_url}")

    bucket_name, object_key = parts
    return bucket_name, object_key


def parse_https_s3_url(https_url: str) -> Tuple[str, str]:
    """
    Parse HTTPS S3 URL to extract bucket and key.

    Args:
        https_url: HTTPS S3 URL in format https://bucket.s3.region.amazonaws.com/key

    Returns:
        Tuple of (bucket_name, object_key)

    Raises:
        ValueError: If URL format is invalid
    """
    normalized_url = _normalize_url_protocol(https_url)
    _validate_url_protocol(normalized_url)
    domain, object_key = _extract_domain_and_key(normalized_url)
    bucket_name = _extract_bucket_from_domain(domain)
    return bucket_name, object_key


def _normalize_url_protocol(url: str) -> str:
    """Normalize URL protocol format to handle Path normalization issues."""
    if url.startswith("https:/") and not url.startswith("https://"):
        return url.replace("https:/", "https://", 1)
    if url.startswith("http:/") and not url.startswith("http://"):
        return url.replace("http:/", "http://", 1)
    return url


def _validate_url_protocol(url: str) -> None:
    """Validate URL has correct protocol format."""
    if not url.startswith(("https://", "http://")):
        raise ValueError(f"Invalid HTTPS S3 URL format: {url}")


def _extract_domain_and_key(url: str) -> Tuple[str, str]:
    """Extract domain and object key from URL."""
    url_without_protocol = url.split("://", 1)[1]
    parts = url_without_protocol.split("/", 1)

    if len(parts) != 2:
        raise ValueError(f"Invalid HTTPS S3 URL format: {url}")

    return parts[0], parts[1]


def _extract_bucket_from_domain(domain: str) -> str:
    """Extract bucket name from S3 domain."""
    domain_parts = domain.split(".")
    if len(domain_parts) < 4 or not domain.endswith(".amazonaws.com"):
        raise ValueError(f"Invalid S3 domain format: {domain}")

    return domain_parts[0]


def build_s3_url(bucket_name: str, object_key: str) -> str:
    """
    Build S3 URL from bucket and key components.

    Args:
        bucket_name: S3 bucket name
        object_key: S3 object key

    Returns:
        S3 URL in format s3://bucket/key
    """
    return f"s3://{bucket_name}/{object_key}"
