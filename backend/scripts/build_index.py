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

from app.database import SessionLocal
from app.models.db_models import Snippet
from app.ml.vector_store import VectorStore
import numpy as np
import json
import faiss


def build_index(env: str):
    """Build FAISS index for specified environment."""
    print(f"🔧 Building FAISS index for {env.upper()}...")

    data_dir = Path(__file__).parent.parent / "data" / env
    data_dir.mkdir(exist_ok=True)

    index_path = data_dir / "faiss_index.bin"
    metadata_path = data_dir / "snippet_metadata.json"

    # Fetch snippets from database
    db = SessionLocal()
    try:
        snippets = db.query(Snippet).all()
        print(f"📊 Found {len(snippets)} snippets in database")

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

        # Save index
        faiss.write_index(index, str(index_path))
        print(f"✅ Saved FAISS index to {index_path}")

        # Save metadata
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Saved metadata to {metadata_path}")

        # Verify
        loaded_index = faiss.read_index(str(index_path))
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
