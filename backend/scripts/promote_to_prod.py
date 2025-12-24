#!/usr/bin/env python3
"""Promote FAISS index and metadata from stage → prod using MinIO/S3 buckets."""

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
from app.utils.s3_utils import (
    _ensure_bucket_exists,
    get_endpoint_for_env,
    get_s3_client_for_env,
)


STAGE_BUCKET = get_env_data_bucket("stage")
PROD_BUCKET = get_env_data_bucket("prod")
INDEX_KEY = settings.faiss_index_key
METADATA_KEY = settings.snippet_metadata_key


def _require_client(env: str, bucket: str):
    try:
        client = get_s3_client_for_env(env)
    except EndpointConnectionError as exc:
        endpoint = get_endpoint_for_env(env)
        raise RuntimeError(
            f"{env.capitalize()} MinIO endpoint {endpoint} is unreachable. Start the {env} stack."
        ) from exc

    try:
        _ensure_bucket_exists(client, bucket)
    except EndpointConnectionError as exc:
        raise RuntimeError(
            f"{env.capitalize()} MinIO endpoint {get_endpoint_for_env(env)} is unreachable. Start the {env} stack."
        ) from exc
    except ClientError as exc:
        raise RuntimeError(f"Unable to access {env} bucket {bucket}: {exc}") from exc
    return client


def validate_artifacts(stage_client) -> tuple[str, list[dict[str, object]]]:
    """Validate stage artifacts before promotion."""
    print("🔍 Validating stage artifacts…")

    try:
        index_path = download_to_temp(STAGE_BUCKET, INDEX_KEY, s3=stage_client)
    except ClientError as exc:
        print(f"❌ Error downloading index from stage bucket: {exc}")
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
        metadata = read_json(STAGE_BUCKET, METADATA_KEY, s3=stage_client)
    except ClientError as exc:
        print(f"❌ Error fetching metadata from stage bucket: {exc}")
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


def backup_prod(prod_client) -> None:
    """Backup existing prod objects into a timestamped prefix."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    backed_up = False

    for key, content_type in (
        (INDEX_KEY, "application/octet-stream"),
        (METADATA_KEY, "application/json"),
    ):
        try:
            obj = prod_client.get_object(Bucket=PROD_BUCKET, Key=key)
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
            PROD_BUCKET,
            backup_key,
            obj["Body"].read(),
            content_type=content_type,
            s3=prod_client,
        )
        print(
            f"✅ Backed up s3://{PROD_BUCKET}/{key} → s3://{PROD_BUCKET}/{backup_key}"
        )
        backed_up = True

    if not backed_up:
        print("ℹ️ No existing prod artifacts to back up.")


def get_confirmation() -> bool:
    """Ask for manual confirmation before prod deployment."""
    print("\n" + "!" * 60)
    print("⚠️  WARNING: You are about to deploy to PRODUCTION")
    print("!" * 60)
    print("\nThis will replace the production FAISS index and snippet metadata.")
    print("Make sure you have:")
    print("  1. Tested thoroughly in stage environment")
    print("  2. Reviewed all changes")
    print("  3. Notified the team (if applicable)")
    print()

    response = input("Type 'deploy to prod' to confirm: ").strip()
    return response == "deploy to prod"


def promote(prod_client, index_path: str, metadata: list[dict[str, object]]) -> None:
    """Copy stage artifacts to the prod bucket."""
    print("🚀 Promoting to prod…")

    index_bytes = Path(index_path).read_bytes()
    upload_bytes(PROD_BUCKET, INDEX_KEY, index_bytes, s3=prod_client)
    print(f"✅ Uploaded FAISS index to s3://{PROD_BUCKET}/{INDEX_KEY}")

    metadata_bytes = json.dumps(metadata, indent=2).encode("utf-8")
    upload_bytes(
        PROD_BUCKET,
        METADATA_KEY,
        metadata_bytes,
        content_type="application/json",
        s3=prod_client,
    )
    print(f"✅ Uploaded metadata to s3://{PROD_BUCKET}/{METADATA_KEY}")


def main() -> None:
    print("=" * 60)
    print("PROMOTE STAGE → PROD")
    print("=" * 60)

    try:
        stage_client = _require_client("stage", STAGE_BUCKET)
    except RuntimeError as exc:
        print(f"\n❌ {exc}")
        sys.exit(1)

    index_path, metadata = validate_artifacts(stage_client)
    if not index_path:
        print("\n❌ Validation failed. Aborting promotion.")
        sys.exit(1)

    if not get_confirmation():
        print("\n❌ Promotion cancelled.")
        Path(index_path).unlink(missing_ok=True)
        sys.exit(0)

    try:
        prod_client = _require_client("prod", PROD_BUCKET)
    except RuntimeError as exc:
        Path(index_path).unlink(missing_ok=True)
        print(f"\n❌ {exc}")
        sys.exit(1)

    try:
        backup_prod(prod_client)
        promote(prod_client, index_path, metadata)
    finally:
        Path(index_path).unlink(missing_ok=True)

    print("\n" + "=" * 60)
    print("✅ Production deployment complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Monitor production logs and metrics")
    print("2. Verify snippet retrieval is working correctly")
    print("3. Check user feedback for any issues")


if __name__ == "__main__":
    main()
