import io
from datetime import datetime, timezone
from typing import Optional

import boto3
import numpy as np
from botocore.client import Config

from app.config import settings

# -------------------------------------------------------------------
# Configuration defaults via settings
# -------------------------------------------------------------------

MINIO_ENDPOINT = settings.minio_endpoint
MINIO_ACCESS_KEY = settings.minio_access_key
MINIO_SECRET_KEY = settings.minio_secret_key
MINIO_BUCKET = settings.data_bucket_dev
MINIO_REGION = settings.minio_region

# -------------------------------------------------------------------
# Client factory
# -------------------------------------------------------------------


def get_s3_client(
    *,
    endpoint_url: Optional[str] = None,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    region_name: Optional[str] = None,
):
    """Return an S3-compatible client for MinIO with optional overrides."""

    config = Config(
        signature_version="s3v4",
        s3={"addressing_style": "path"},
    )

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url or MINIO_ENDPOINT,
        aws_access_key_id=access_key or MINIO_ACCESS_KEY,
        aws_secret_access_key=secret_key or MINIO_SECRET_KEY,
        region_name=region_name or MINIO_REGION,
        config=config,
        verify=False,
    )


def get_endpoint_for_env(env: str) -> str:
    env = (env or "dev").lower()
    if env == "stage":
        return settings.minio_stage_endpoint
    if env == "prod":
        return settings.minio_prod_endpoint
    # Treat offline/dev/default the same
    return settings.minio_dev_endpoint


def get_s3_client_for_env(env: str):
    return get_s3_client(endpoint_url=get_endpoint_for_env(env))


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
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    buffer.seek(0)

    # Deterministic object key
    key = f"lints-agent/" f"version={agent.version}/" f"session_{session_count:08d}.npz"

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
