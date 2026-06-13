"""Shared application state for the VeloxDB API server."""

import os
import sys
from pathlib import Path

# Ensure compiled veloxdb module is importable from repo build/
_REPO_ROOT = Path(__file__).resolve().parent.parent
_BUILD_DIR = _REPO_ROOT / "build"
if str(_BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(_BUILD_DIR))

import veloxdb  # noqa: E402

DATA_DIR = _REPO_ROOT / "data"
DATA_FILE = DATA_DIR / "vectors.fvecs"
INDEX_FILE = DATA_DIR / "index.ivf"
METADATA_FILE = DATA_DIR / "metadata.json"

db = veloxdb.VectorIndex()
is_indexed = False
vector_count = 0
dim: int | None = None


def refresh_stats() -> None:
    global vector_count, dim
    if vector_count <= 0:
        dim = None
        return
    try:
        sample = db.get_vector(0)
        dim = len(sample)
    except Exception:
        dim = None
