import io
import os
from datetime import datetime
from typing import Optional

import boto3
import numpy as np
from botocore.client import Config

# -------------------------------------------------------------------
# Configuration (via environment variables)
# -------------------------------------------------------------------

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "ml-artifacts")
MINIO_REGION = os.getenv("MINIO_REGION", "us-east-1")

# -------------------------------------------------------------------
# Client factory
# -------------------------------------------------------------------

def get_s3_client():
    """
    Returns an S3-compatible client for MinIO.
    """
    # Configure boto3 for MinIO compatibility
    config = Config(
        signature_version="s3v4",
        s3={"addressing_style": "path"},  # Use path-style addressing for MinIO
    )
    
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        region_name=MINIO_REGION,
        config=config,
        verify=False,  # Disable SSL verification for MinIO
    )


# -------------------------------------------------------------------
# Snapshot persistence
# -------------------------------------------------------------------

def save_agent_snapshot_to_s3(
    agent,
    session_count: int,
    *,
    bucket: Optional[str] = None,
) -> str:
    """
    Persist full LinTS agent parameters (W_mean, W_precision) to MinIO.

    Returns:
        s3://<bucket>/<key>
    """
    bucket = bucket or MINIO_BUCKET
    s3 = get_s3_client()

    params = agent.return_parameters()

    # Defensive checks
    if "W_mean" not in params or "W_precision" not in params:
        raise ValueError("Agent parameters missing W_mean or W_precision")

    # Serialize to compressed NumPy archive
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        W_mean=params["W_mean"].astype(np.float32),
        W_precision=params["W_precision"].astype(np.float32),
        model_version=agent.version,
        session_count=session_count,
        timestamp=datetime.utcnow().isoformat(),
    )
    buffer.seek(0)

    # Deterministic object key
    key = (
        f"lints-agent/"
        f"version={agent.version}/"
        f"session_{session_count:08d}.npz"
    )

    # Ensure bucket exists (idempotent)
    _ensure_bucket_exists(s3, bucket)

    # Upload
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=buffer,
        ContentType="application/octet-stream",
    )

    return f"s3://{bucket}/{key}"


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _ensure_bucket_exists(s3, bucket: str):
    """
    Create bucket if it does not exist.
    Safe to call repeatedly.
    """
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:
        s3.create_bucket(Bucket=bucket)
