#!/usr/bin/env python3
"""Promote FAISS index and metadata from dev → stage using MinIO/S3 buckets."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import faiss
from botocore.exceptions import ClientError, EndpointConnectionError

from app.config import settings
from app.utils.s3_data import (
    download_to_temp,
    get_env_data_bucket,
    read_json,
    upload_bytes,
)
from app.utils.s3_utils import _ensure_bucket_exists, get_endpoint_for_env, get_s3_client_for_env


DEV_BUCKET = get_env_data_bucket("dev")
STAGE_BUCKET = get_env_data_bucket("stage")
INDEX_KEY = settings.faiss_index_key
METADATA_KEY = settings.snippet_metadata_key


def _require_stage_client():
    try:
        client = get_s3_client_for_env("stage")
    except EndpointConnectionError as exc:
        endpoint = get_endpoint_for_env("stage")
        raise RuntimeError(
            f"Stage MinIO endpoint {endpoint} is unreachable. Start the stage stack."
        ) from exc

    try:
        _ensure_bucket_exists(client, STAGE_BUCKET)
    except EndpointConnectionError as exc:
        raise RuntimeError(
            f"Stage MinIO endpoint {get_endpoint_for_env('stage')} is unreachable. Start the stage stack."
        ) from exc
    except ClientError as exc:
        raise RuntimeError(
            f"Unable to access stage bucket {STAGE_BUCKET}: {exc}"
        ) from exc
    return client


def validate_artifacts() -> tuple[str, list[dict[str, object]]]:
    """Validate dev artifacts before promotion."""
    print("🔍 Validating dev artifacts…")

    try:
        index_path = download_to_temp(DEV_BUCKET, INDEX_KEY, env="dev")
    except (ClientError, EndpointConnectionError) as exc:
        print(f"❌ Error downloading index from dev bucket: {exc}")
        return "", []

    try:
        index = faiss.read_index(index_path)
        vector_count = index.ntotal
        print(f"✅ FAISS index loaded: {vector_count} vectors")
        if vector_count == 0:
            print("❌ Error: FAISS index is empty")
            return "", []
    except Exception as exc:  # pragma: no cover - faiss errors are rare
        print(f"❌ Error loading FAISS index: {exc}")
        return "", []

    try:
        metadata = read_json(DEV_BUCKET, METADATA_KEY, env="dev")
    except (ClientError, EndpointConnectionError) as exc:
        print(f"❌ Error fetching metadata from dev bucket: {exc}")
        return "", []

    if not isinstance(metadata, list):
        print("❌ Error: Metadata is not a list")
        return "", []

    snippet_count = len(metadata)
    print(f"✅ Metadata loaded: {snippet_count} snippets")

    if snippet_count == 0:
        print("❌ Error: Metadata is empty")
        return "", []

    if vector_count != snippet_count:
        print(
            f"❌ Error: Vector count ({vector_count}) doesn't match metadata count ({snippet_count})"
        )
        return "", []

    required_fields = ["id", "words"]
    for i, item in enumerate(metadata[:5]):
        if not all(field in item for field in required_fields):
            print(f"❌ Error: Metadata entry {i} missing required fields")
            return "", []

    print("✅ Metadata structure validated")
    return index_path, metadata


def backup_stage(stage_client) -> None:
    """Backup existing stage objects into a timestamped prefix."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    backed_up = False

    for key, content_type in (
        (INDEX_KEY, "application/octet-stream"),
        (METADATA_KEY, "application/json"),
    ):
        try:
            obj = stage_client.get_object(Bucket=STAGE_BUCKET, Key=key)
        except ClientError as exc:
            code = (
                exc.response.get("Error", {}).get("Code")
                if hasattr(exc, "response")
                else None
            )
            if code in {"404", "NoSuchKey", "NotFound"}:
                continue
            raise

        backup_key = f"backups/{timestamp}/{key}"
        upload_bytes(
            STAGE_BUCKET,
            backup_key,
            obj["Body"].read(),
            content_type=content_type,
            s3=stage_client,
        )
        print(f"✅ Backed up s3://{STAGE_BUCKET}/{key} → s3://{STAGE_BUCKET}/{backup_key}")
        backed_up = True

    if not backed_up:
        print("ℹ️ No existing stage artifacts to back up.")


def promote(stage_client, index_path: str, metadata: list[dict[str, object]]) -> None:
    """Copy dev artifacts to the stage bucket."""
    print("🚀 Promoting to stage…")

    index_bytes = Path(index_path).read_bytes()
    upload_bytes(STAGE_BUCKET, INDEX_KEY, index_bytes, s3=stage_client)
    print(f"✅ Uploaded FAISS index to s3://{STAGE_BUCKET}/{INDEX_KEY}")

    metadata_bytes = json.dumps(metadata, indent=2).encode("utf-8")
    upload_bytes(
        STAGE_BUCKET,
        METADATA_KEY,
        metadata_bytes,
        content_type="application/json",
        s3=stage_client,
    )
    print(f"✅ Uploaded metadata to s3://{STAGE_BUCKET}/{METADATA_KEY}")


def main() -> None:
    print("=" * 60)
    print("PROMOTE DEV → STAGE")
    print("=" * 60)

    index_path, metadata = validate_artifacts()
    if not index_path:
        print("\n❌ Validation failed. Aborting promotion.")
        sys.exit(1)

    try:
        stage_client = _require_stage_client()
    except RuntimeError as exc:
        Path(index_path).unlink(missing_ok=True)
        print(f"\n❌ {exc}")
        sys.exit(1)

    try:
        backup_stage(stage_client)
        promote(stage_client, index_path, metadata)
    finally:
        Path(index_path).unlink(missing_ok=True)

    print("\n" + "=" * 60)
    print("✅ Promotion complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Test the stage environment")
    print("2. Run promote_to_prod.py when ready for production")


if __name__ == "__main__":
    main()
