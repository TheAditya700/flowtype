import json
import tempfile
from typing import Any, Optional

from .s3_utils import (
    get_s3_client,
    get_s3_client_for_env,
    _ensure_bucket_exists,
)
from app.config import settings


def get_env_data_bucket(env: str) -> str:
    env = (env or "dev").lower()
    if env == "offline":
        return settings.data_bucket_offline
    if env == "stage":
        return settings.data_bucket_stage
    if env == "prod":
        return settings.data_bucket_prod
    # default dev
    return settings.data_bucket_dev


def _resolve_client(env: Optional[str], s3=None):
    if s3 is not None:
        return s3
    if env is not None:
        return get_s3_client_for_env(env)
    return get_s3_client()


def read_json(bucket: str, key: str, *, env: Optional[str] = None, s3=None) -> Any:
    s3 = _resolve_client(env, s3)
    _ensure_bucket_exists(s3, bucket)
    obj = s3.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()
    return json.loads(data.decode("utf-8"))


def write_json(
    bucket: str, key: str, payload: Any, *, env: Optional[str] = None, s3=None
) -> str:
    s3 = _resolve_client(env, s3)
    _ensure_bucket_exists(s3, bucket)
    body = json.dumps(payload, indent=2).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    return f"s3://{bucket}/{key}"


def download_to_temp(
    bucket: str, key: str, *, env: Optional[str] = None, s3=None
) -> str:
    s3 = _resolve_client(env, s3)
    _ensure_bucket_exists(s3, bucket)
    tmp = tempfile.NamedTemporaryFile(delete=False)
    resp = s3.get_object(Bucket=bucket, Key=key)
    data = resp["Body"].read()
    tmp.write(data)
    tmp.flush()
    return tmp.name


def upload_bytes(
    bucket: str,
    key: str,
    body: bytes,
    content_type: str = "application/octet-stream",
    *,
    env: Optional[str] = None,
    s3=None,
) -> str:
    s3 = _resolve_client(env, s3)
    _ensure_bucket_exists(s3, bucket)
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type)
    return f"s3://{bucket}/{key}"
