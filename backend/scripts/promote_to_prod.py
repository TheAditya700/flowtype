#!/usr/bin/env python3
"""
Promote FAISS index and snippet metadata from stage to prod.

This script validates the stage artifacts and copies them to prod if validation passes.
Requires manual confirmation before promotion.
"""
import shutil
import json
from pathlib import Path
import faiss
import sys

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
STAGE_DIR = DATA_DIR / "stage"
PROD_DIR = DATA_DIR / "prod"

STAGE_INDEX = STAGE_DIR / "faiss_index.bin"
STAGE_METADATA = STAGE_DIR / "snippet_metadata.json"

PROD_INDEX = PROD_DIR / "faiss_index.bin"
PROD_METADATA = PROD_DIR / "snippet_metadata.json"


def validate_artifacts():
    """Validate stage artifacts before promotion."""
    print("🔍 Validating stage artifacts...")

    # Check files exist
    if not STAGE_INDEX.exists():
        print(f"❌ Error: {STAGE_INDEX} not found")
        return False
    if not STAGE_METADATA.exists():
        print(f"❌ Error: {STAGE_METADATA} not found")
        return False

    # Validate FAISS index
    try:
        index = faiss.read_index(str(STAGE_INDEX))
        vector_count = index.ntotal
        print(f"✅ FAISS index loaded: {vector_count} vectors")

        if vector_count == 0:
            print("❌ Error: FAISS index is empty")
            return False
    except Exception as e:
        print(f"❌ Error loading FAISS index: {e}")
        return False

    # Validate metadata
    try:
        with open(STAGE_METADATA) as f:
            metadata = json.load(f)

        if not isinstance(metadata, list):
            print("❌ Error: Metadata is not a list")
            return False

        snippet_count = len(metadata)
        print(f"✅ Metadata loaded: {snippet_count} snippets")

        if snippet_count == 0:
            print("❌ Error: Metadata is empty")
            return False

        # Check vector count matches metadata count
        if vector_count != snippet_count:
            print(
                f"❌ Error: Vector count ({vector_count}) doesn't match metadata count ({snippet_count})"
            )
            return False

        # Validate metadata structure
        required_fields = ["id", "words", "difficulty"]
        for i, item in enumerate(metadata[:5]):  # Check first 5
            if not all(field in item for field in required_fields):
                print(f"❌ Error: Snippet {i} missing required fields")
                return False

        print("✅ Metadata structure validated")

    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        return False

    return True


def backup_prod():
    """Backup existing prod artifacts if they exist."""
    if PROD_INDEX.exists() or PROD_METADATA.exists():
        print("📦 Backing up existing prod artifacts...")
        backup_dir = PROD_DIR / "backup"
        backup_dir.mkdir(exist_ok=True)

        if PROD_INDEX.exists():
            shutil.copy2(PROD_INDEX, backup_dir / "faiss_index.bin.bak")
            print(f"✅ Backed up {PROD_INDEX}")

        if PROD_METADATA.exists():
            shutil.copy2(PROD_METADATA, backup_dir / "snippet_metadata.json.bak")
            print(f"✅ Backed up {PROD_METADATA}")


def get_confirmation():
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


def promote():
    """Copy stage artifacts to prod."""
    print("🚀 Promoting to prod...")

    PROD_DIR.mkdir(exist_ok=True)

    shutil.copy2(STAGE_INDEX, PROD_INDEX)
    print(f"✅ Copied {STAGE_INDEX} → {PROD_INDEX}")

    shutil.copy2(STAGE_METADATA, PROD_METADATA)
    print(f"✅ Copied {STAGE_METADATA} → {PROD_METADATA}")


def main():
    print("=" * 60)
    print("PROMOTE STAGE → PROD")
    print("=" * 60)

    # Validate
    if not validate_artifacts():
        print("\n❌ Validation failed. Aborting promotion.")
        sys.exit(1)

    # Get confirmation
    if not get_confirmation():
        print("\n❌ Promotion cancelled.")
        sys.exit(0)

    # Backup
    backup_prod()

    # Promote
    promote()

    print("\n" + "=" * 60)
    print("✅ Production deployment complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Monitor production logs and metrics")
    print("2. Verify snippet retrieval is working correctly")
    print("3. Check user feedback for any issues")


if __name__ == "__main__":
    main()
