#!/usr/bin/env python3
"""Upload local backend/data/* directories to MinIO buckets.

- offline  -> settings.data_bucket_offline
- dev      -> settings.data_bucket_dev
- stage    -> settings.data_bucket_stage
- prod     -> settings.data_bucket_prod

Uploads all files preserving filenames at the bucket root.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict

from botocore.exceptions import EndpointConnectionError

from app.config import settings
from app.utils.s3_data import upload_bytes
from app.utils.s3_utils import (
    _ensure_bucket_exists,
    get_endpoint_for_env,
    get_s3_client_for_env,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

BUCKETS: Dict[str, str] = {
    "offline": settings.data_bucket_offline,
    "dev": settings.data_bucket_dev,
    "stage": settings.data_bucket_stage,
    "prod": settings.data_bucket_prod,
}


def upload_dir_to_bucket(subdir: str, *, target_env: str | None = None) -> None:
    bucket = BUCKETS[subdir]
    src = DATA_DIR / subdir
    if not src.exists():
        print(f"⚠️  Skipping {subdir}: {src} does not exist")
        return

    client_env = (target_env or subdir).lower()
    client = get_s3_client_for_env(client_env)
    try:
        _ensure_bucket_exists(client, bucket)
    except EndpointConnectionError as exc:
        endpoint = get_endpoint_for_env(client_env)
        print(
            f"❌ Cannot reach {client_env} MinIO endpoint {endpoint}: {exc}. Start the {client_env} stack before migrating."
        )
        return

    suffix = f" ({client_env})" if target_env else ""
    print(f"⤴ Uploading {subdir}{suffix} → s3://{bucket}/")
    count = 0
    for p in src.rglob("*"):
        if p.is_file():
            key = p.name
            upload_bytes(bucket, key, p.read_bytes(), s3=client)
            count += 1
    print(f"✅ Uploaded {count} files to {bucket}")


def main() -> int:
    for env in ["dev", "stage", "prod"]:
        upload_dir_to_bucket("offline", target_env=env)

    for subdir in ["dev", "stage", "prod"]:
        upload_dir_to_bucket(subdir)
    print("✔ Migration complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
