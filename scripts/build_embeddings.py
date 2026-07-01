"""
Precompute TF-IDF vectors for catalog items (run locally after build_catalog.py).

No PyTorch / ONNX — safe for Render 512MB free tier at build and runtime.

    python scripts/build_embeddings.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.catalog import CatalogItem  # noqa: E402

CATALOG_PATH = ROOT / "data" / "catalog.json"
VECTORIZER_PATH = ROOT / "data" / "tfidf_vectorizer.joblib"
MATRIX_PATH = ROOT / "data" / "catalog_tfidf.npz"
META_PATH = ROOT / "data" / "catalog_tfidf.meta.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def main() -> int:
    if not CATALOG_PATH.exists():
        log.error("Missing %s — run scripts/build_catalog.py first", CATALOG_PATH)
        return 1

    raw = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    items = [CatalogItem.model_validate(row) for row in raw]
    texts = [item.search_text() for item in items]
    entity_ids = [item.entity_id for item in items]

    log.info("Building TF-IDF matrix for %d catalog items …", len(texts))
    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)

    VECTORIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    save_npz(MATRIX_PATH, matrix)
    META_PATH.write_text(
        json.dumps({"count": len(items), "entity_ids": entity_ids}, indent=2) + "\n",
        encoding="utf-8",
    )

    log.info("Wrote %s and %s (%d x %d sparse)", VECTORIZER_PATH, MATRIX_PATH, *matrix.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
