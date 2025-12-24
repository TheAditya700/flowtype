#!/usr/bin/env python3
"""
Build FAISS index from database snippets for a specific environment.

Usage:
    python build_index.py --env dev     # Build for dev
    python build_index.py --env stage   # Build for stage
    python build_index.py --env prod    # Build for prod
"""
import argparse
import sys
from pathlib import Path

# Add parent dir to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
import faiss
from scripts.condense_snippet_embeddings import condense_embeddings
from botocore.exceptions import EndpointConnectionError

from app.database import SessionLocal
from app.models.db_models import Snippet


def build_index(env: str):
    """Build FAISS index for specified environment."""
    print(f"🔧 Building FAISS index for {env.upper()}...")

    from app.utils.s3_data import get_env_data_bucket, upload_bytes
    from app.config import settings
    from app.utils.s3_utils import (
        _ensure_bucket_exists,
        get_endpoint_for_env,
        get_s3_client_for_env,
    )

    bucket = get_env_data_bucket(env)
    s3_client = get_s3_client_for_env(env)
    endpoint = get_endpoint_for_env(env)

    try:
        _ensure_bucket_exists(s3_client, bucket)
    except EndpointConnectionError as exc:
        print(
            f"❌ Unable to reach {env.upper()} MinIO endpoint {endpoint}. Start the {env} stack before building."
        )
        return False

    # Fetch snippets from database
    db = SessionLocal()
    try:
        snippets = db.query(Snippet).all()
        print(f"📊 Found {len(snippets)} snippets in database")

        # Backfill embeddings if missing
        missing_embeddings = [s for s in snippets if not s.embedding]
        if missing_embeddings:
            print(
                f"⚠️  {len(missing_embeddings)} snippets missing embeddings. Running condense_snippet_embeddings..."
            )
            db.close()
            condense_embeddings()
            db = SessionLocal()
            snippets = db.query(Snippet).all()
            missing_embeddings = [s for s in snippets if not s.embedding]
            if missing_embeddings:
                print("❌ Embedding backfill failed; aborting index build.")
                return False

        if not snippets:
            print("❌ No snippets found in database. Run seed_data.py first.")
            return False

        # Collect embeddings and metadata
        embeddings = []
        metadata = []

        for snippet in snippets:
            if snippet.embedding is None:
                print(f"⚠️  Warning: Snippet {snippet.id} has no embedding, skipping")
                continue

            embeddings.append(snippet.embedding)
            metadata.append(
                {
                    "id": snippet.id,
                    "words": snippet.words,
                }
            )

        print(f"✅ Prepared {len(embeddings)} snippets with embeddings")

        # Build FAISS index
        embeddings_array = np.array(embeddings, dtype=np.float32)
        dimension = embeddings_array.shape[1]

        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)  # type: ignore[arg-type]

        # Save index to temp and upload to S3
        tmp_idx = Path("/tmp/faiss_index.bin")
        faiss.write_index(index, str(tmp_idx))
        upload_bytes(
            bucket,
            settings.faiss_index_key,
            tmp_idx.read_bytes(),
            s3=s3_client,
        )
        print(f"✅ Uploaded FAISS index to s3://{bucket}/{settings.faiss_index_key}")

        # Upload metadata JSON to S3
        upload_bytes(
            bucket,
            settings.snippet_metadata_key,
            json.dumps(metadata, indent=2).encode("utf-8"),
            content_type="application/json",
            s3=s3_client,
        )
        print(f"✅ Uploaded metadata to s3://{bucket}/{settings.snippet_metadata_key}")

        # Verify
        loaded_index = index  # Already built in-memory
        print(f"✅ Verification: Index contains {loaded_index.ntotal} vectors")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index for environment")
    parser.add_argument(
        "--env",
        choices=["dev", "stage", "prod"],
        required=True,
        help="Target environment",
    )

    args = parser.parse_args()

    print("=" * 60)
    print(f"BUILD INDEX FOR {args.env.upper()}")
    print("=" * 60)

    success = build_index(args.env)

    if success:
        print("\n✅ Build complete!")
        print(f"\nNext steps for {args.env}:")
        if args.env == "dev":
            print("1. Test the index locally")
            print("2. Run promote_to_stage.py when ready")
        elif args.env == "stage":
            print("1. Test in stage environment")
            print("2. Run promote_to_prod.py when ready")
        elif args.env == "prod":
            print("1. Monitor production carefully")
    else:
        print("\n❌ Build failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
