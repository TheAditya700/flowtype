#!/usr/bin/env python3
"""One-shot bootstrap for a fresh environment.

Steps (in order):
1) Generate snippet artifacts (word features + snippets).
2) Ensure tables exist.
3) Populate snippets into the DB (clears existing by default).
4) Build the FAISS index for the chosen env (dev|stage|prod).

Skip any step with flags if you already ran it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure backend modules are importable when run as a script
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.database import Base, engine
from app.generator import generate
from app.generator.populate_db import populate_snippet_database
from scripts.build_index import build_index
from app.generator import config as gen_config


def ensure_prereqs() -> None:
    required = [
        gen_config.ENRICHED_WORDLIST_PATH,
        gen_config.WORDLIST_PATH,
        gen_config.BIGRAM_PATH,
        gen_config.TRIGRAM_PATH,
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        msg = "Missing required data files: " + ", ".join(str(p) for p in missing)
        raise FileNotFoundError(msg)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap environment: generate → populate → build index"
    )
    parser.add_argument(
        "--env",
        choices=["dev", "stage", "prod"],
        default="dev",
        help="Target environment for FAISS artifacts",
    )
    parser.add_argument(
        "--skip-generate", action="store_true", help="Skip snippet generation step"
    )
    parser.add_argument(
        "--skip-populate", action="store_true", help="Skip DB population step"
    )
    parser.add_argument(
        "--skip-index", action="store_true", help="Skip FAISS index build step"
    )
    parser.add_argument(
        "--keep-existing-snippets",
        action="store_true",
        help="Do not clear existing snippets before populate",
    )
    args = parser.parse_args()

    try:
        ensure_prereqs()
    except FileNotFoundError as exc:
        print(f"❌ {exc}")
        print("Provide the wordlists/ngram files in backend/data before bootstrapping.")
        return 1

    if not args.skip_generate:
        print("[1/3] Generating word and snippet artifacts…")
        generate.run()
    else:
        print("[1/3] Skipping generation (requested)")

    print("Ensuring tables exist…")
    Base.metadata.create_all(bind=engine)

    if not args.skip_populate:
        print("[2/3] Populating snippets into database…")
        populate_snippet_database(clear_existing=not args.keep_existing_snippets)
    else:
        print("[2/3] Skipping population (requested)")

    if not args.skip_index:
        print(f"[3/3] Building FAISS index for {args.env}…")
        ok = build_index(args.env)
        if not ok:
            return 1
    else:
        print("[3/3] Skipping index build (requested)")

    print("✔ Bootstrap complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
