"""Compatibility wrapper for building FAISS indices.

Prefer `python scripts/build_index.py --env {dev|stage|prod}`.
This script defaults to dev to keep existing tooling working.
"""

from pathlib import Path
import sys
import argparse

# Allow importing the shared build_index implementation
sys.path.insert(0, str(Path(__file__).parent))
from build_index import build_index


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build FAISS index (compat wrapper; defaults to dev).",
    )
    parser.add_argument(
        "--env",
        choices=["dev", "stage", "prod"],
        default="dev",
        help="Target environment (default: dev)",
    )
    args = parser.parse_args()

    ok = build_index(args.env)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
