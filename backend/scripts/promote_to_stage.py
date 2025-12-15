#!/usr/bin/env python3
"""
Promote FAISS index and snippet metadata from dev to stage.

This script validates the dev artifacts and copies them to stage if validation passes.
"""
import shutil
import json
from pathlib import Path
import faiss
import sys

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
DEV_DIR = DATA_DIR / "dev"
STAGE_DIR = DATA_DIR / "stage"

DEV_INDEX = DEV_DIR / "faiss_index.bin"
DEV_METADATA = DEV_DIR / "snippet_metadata.json"

STAGE_INDEX = STAGE_DIR / "faiss_index.bin"
STAGE_METADATA = STAGE_DIR / "snippet_metadata.json"


def validate_artifacts():
    """Validate dev artifacts before promotion."""
    print("🔍 Validating dev artifacts...")

    # Check files exist
    if not DEV_INDEX.exists():
        print(f"❌ Error: {DEV_INDEX} not found")
        return False
    if not DEV_METADATA.exists():
        print(f"❌ Error: {DEV_METADATA} not found")
        return False

    # Validate FAISS index
    try:
        index = faiss.read_index(str(DEV_INDEX))
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
        with open(DEV_METADATA) as f:
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


def backup_stage():
    """Backup existing stage artifacts if they exist."""
    if STAGE_INDEX.exists() or STAGE_METADATA.exists():
        print("📦 Backing up existing stage artifacts...")
        backup_dir = STAGE_DIR / "backup"
        backup_dir.mkdir(exist_ok=True)

        if STAGE_INDEX.exists():
            shutil.copy2(STAGE_INDEX, backup_dir / "faiss_index.bin.bak")
            print(f"✅ Backed up {STAGE_INDEX}")

        if STAGE_METADATA.exists():
            shutil.copy2(STAGE_METADATA, backup_dir / "snippet_metadata.json.bak")
            print(f"✅ Backed up {STAGE_METADATA}")


def promote():
    """Copy dev artifacts to stage."""
    print("🚀 Promoting to stage...")

    STAGE_DIR.mkdir(exist_ok=True)

    shutil.copy2(DEV_INDEX, STAGE_INDEX)
    print(f"✅ Copied {DEV_INDEX} → {STAGE_INDEX}")

    shutil.copy2(DEV_METADATA, STAGE_METADATA)
    print(f"✅ Copied {DEV_METADATA} → {STAGE_METADATA}")


def main():
    print("=" * 60)
    print("PROMOTE DEV → STAGE")
    print("=" * 60)

    # Validate
    if not validate_artifacts():
        print("\n❌ Validation failed. Aborting promotion.")
        sys.exit(1)

    # Backup
    backup_stage()

    # Promote
    promote()

    print("\n" + "=" * 60)
    print("✅ Promotion complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Test the stage environment")
    print("2. Run promote_to_prod.py when ready for production")


if __name__ == "__main__":
    main()
