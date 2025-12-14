import os
import sys
from pathlib import Path

# Ensure backend package is importable when running pytest from repo root
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Provide safe defaults so settings can initialize during tests
os.environ.setdefault("SECRET_KEY", "test-secret-key")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")
os.environ.setdefault("FAISS_INDEX_PATH", "data/test_faiss.index")
os.environ.setdefault("SNIPPET_METADATA_PATH", "data/test_snippet_metadata.json")
