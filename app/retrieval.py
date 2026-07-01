"""
Hybrid retriever: precomputed TF-IDF vectors + structured filters/boosts.

Catalog vectors are built offline (scripts/build_embeddings.py). Runtime uses
scikit-learn only — no PyTorch/sentence-transformers (~500MB saved on Render).
"""
from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import load_npz, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from app.catalog import CatalogIndex, CatalogItem, load_catalog_index
from app.role_packs import slugs_for_query

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
VECTORIZER_PATH = ROOT / "data" / "tfidf_vectorizer.joblib"
MATRIX_PATH = ROOT / "data" / "catalog_tfidf.npz"
META_PATH = ROOT / "data" / "catalog_tfidf.meta.json"

_DURATION_RE = re.compile(r"(\d+)\s*minute", re.I)
DEFAULT_OPQ_SLUG = "occupational-personality-questionnaire-opq32r"


def parse_duration_minutes(duration: str) -> int | None:
    if not duration:
        return None
    low = duration.lower()
    if "untimed" in low or "variable" in low:
        return None
    match = _DURATION_RE.search(duration)
    return int(match.group(1)) if match else None


class HybridRetriever:
    """TF-IDF semantic search over catalog with structured filtering."""

    def __init__(self, index: CatalogIndex) -> None:
        self.index = index
        self._vectorizer: Any = None
        self._catalog_matrix: csr_matrix | None = None
        self._lock = threading.Lock()
        self._ready = False

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            if not VECTORIZER_PATH.exists() or not MATRIX_PATH.exists():
                raise FileNotFoundError(
                    f"Missing TF-IDF artifacts. Run: python scripts/build_embeddings.py"
                )
            log.info("Loading precomputed TF-IDF index …")
            self._vectorizer = joblib.load(VECTORIZER_PATH)
            self._catalog_matrix = load_npz(MATRIX_PATH)
            if META_PATH.exists():
                meta = json.loads(META_PATH.read_text(encoding="utf-8"))
                expected = meta.get("entity_ids", [])
                if len(expected) == len(self.index.items):
                    for i, (eid, item) in enumerate(
                        zip(expected, self.index.items, strict=True)
                    ):
                        if eid != item.entity_id:
                            log.warning("TF-IDF row %d entity_id mismatch", i)
            self._ready = True
            log.info("TF-IDF index ready (%d items)", self._catalog_matrix.shape[0])

    def warm(self) -> None:
        self._ensure_ready()

    @property
    def is_ready(self) -> bool:
        return self._ready

    def _semantic_scores(self, query: str) -> np.ndarray:
        assert self._vectorizer is not None and self._catalog_matrix is not None
        q_vec = self._vectorizer.transform([query])
        return cosine_similarity(q_vec, self._catalog_matrix).flatten()

    def _apply_filters(
        self, candidates: list[tuple[int, float]], filters: dict[str, Any]
    ) -> list[tuple[int, float]]:
        if not filters:
            return candidates

        test_types: set[str] | None = filters.get("test_types")
        job_levels: set[str] | None = filters.get("job_levels")
        max_duration: int | None = filters.get("max_duration_minutes")
        languages: set[str] | None = filters.get("languages")
        require_remote: bool | None = filters.get("remote")

        filtered: list[tuple[int, float]] = []
        for idx, score in candidates:
            item = self.index.items[idx]

            if test_types:
                item_codes = set(item.test_type.split(","))
                if not item_codes & test_types:
                    continue

            if job_levels:
                item_levels = {lv.lower() for lv in item.job_levels}
                if item_levels and not (item_levels & {j.lower() for j in job_levels}):
                    if item.job_levels:
                        continue

            if max_duration is not None:
                mins = parse_duration_minutes(item.duration)
                if mins is not None and mins > max_duration:
                    continue

            if languages:
                item_langs = {lang.lower() for lang in item.languages}
                if item_langs and not (item_langs & {lang.lower() for lang in languages}):
                    continue

            if require_remote is True and item.remote.lower() != "yes":
                continue

            filtered.append((idx, score))
        return filtered

    def _keyword_score(self, query: str, item: CatalogItem) -> float:
        q_tokens = set(re.findall(r"[a-z0-9+]{3,}", query.lower()))
        text = item.search_text().lower()
        hits = sum(1 for t in q_tokens if t in text)
        return hits / max(len(q_tokens), 1)

    def _role_slug_boosts(self, query: str) -> dict[int, float]:
        boosts: dict[int, float] = {}
        q = query.lower()
        rules: list[tuple[list[str], list[str]]] = [
            (["leadership", "cxo", "director", "executive"], [
                "occupational-personality-questionnaire-opq32r",
                "opq-universal-competency-report-2-0",
                "opq-leadership-report",
            ]),
            (["rust", "networking", "linux"], [
                "smart-interview-live-coding",
                "linux-programming-general",
                "networking-and-implementation-new",
                "shl-verify-interactive-g",
            ]),
            (["contact centre", "contact center", "call center"], [
                "svar-spoken-english-us-new",
                "contact-center-call-simulation-new",
                "entry-level-customer-serv-retail-and-contact-center",
                "customer-service-phone-simulation",
            ]),
            (["financial analyst", "graduate", "numerical"], [
                "shl-verify-interactive-numerical-reasoning",
                "financial-accounting-new",
                "basic-statistics-new",
                "graduate-scenarios",
            ]),
            (["sales", "re-skill", "audit"], [
                "global-skills-assessment",
                "global-skills-development-report",
                "opq-mq-sales-report",
                "salestransformationreport2-0-individualcontributor",
            ]),
            (["plant operator", "safety", "chemical"], [
                "safety-and-dependability-focus-8-0",
                "workplace-health-and-safety-new",
                "dependability-and-safety-instrument-dsi",
            ]),
            (["healthcare", "hipaa", "spanish"], [
                "hipaa-security",
                "medical-terminology-new",
                "microsoft-word-365-essentials-new",
                "dependability-and-safety-instrument-dsi",
            ]),
            (["excel", "word", "admin assistant"], [
                "microsoft-excel-365-new",
                "microsoft-word-365-new",
                "ms-excel-new",
                "ms-word-new",
            ]),
            (["java", "spring", "full-stack", "full stack", "aws", "docker"], [
                "core-java-advanced-level-new",
                "spring-new",
                "sql-new",
                "amazon-web-services-aws-development-new",
                "docker-new",
                "shl-verify-interactive-g",
            ]),
            (["trainee", "management trainee", "cognitive", "situational"], [
                "shl-verify-interactive-g",
                "graduate-scenarios",
            ]),
        ]
        slug_to_idx = {item.slug(): i for i, item in enumerate(self.index.items)}
        for triggers, slugs in rules:
            if any(t in q for t in triggers):
                for slug in slugs:
                    idx = slug_to_idx.get(slug)
                    if idx is not None:
                        boosts[idx] = max(boosts.get(idx, 0.0), 0.85)
        if any(w in q for w in ("personality", "opq", "hire", "senior", "engineer", "analyst", "admin")):
            if "drop the opq" not in q and "remove the opq" not in q:
                idx = slug_to_idx.get(DEFAULT_OPQ_SLUG)
                if idx is not None:
                    boosts[idx] = max(boosts.get(idx, 0.0), 0.6)
        return boosts

    def _exact_name_boosts(self, query: str) -> dict[int, float]:
        boosts: dict[int, float] = {}
        q = query.lower()
        aliases = {
            "opq32r": "occupational personality questionnaire opq32r",
            "verify g+": "shl verify interactive g+",
            "verify g": "shl verify interactive g+",
            "graduate scenarios": "graduate scenarios",
            "dsi": "dependability and safety instrument",
            "svar": "svar spoken english us",
        }
        for alias, fragment in aliases.items():
            if alias in q:
                q = q + " " + fragment
        for idx, item in enumerate(self.index.items):
            name_l = item.name.lower()
            if name_l in q or any(tok in name_l for tok in q.split() if len(tok) > 4 and tok in name_l):
                boosts[idx] = max(boosts.get(idx, 0.0), 0.5)
            tokens = [t for t in re.split(r"[^a-z0-9+]+", name_l) if len(t) > 3]
            hits = sum(1 for t in tokens if t in q)
            if hits >= 2:
                boosts[idx] = max(boosts.get(idx, 0.0), 0.3 + 0.1 * hits)
        return boosts

    def retrieve(self, query: str, filters: dict[str, Any] | None = None, k: int = 10) -> list[CatalogItem]:
        self._ensure_ready()

        filters = filters or {}
        fetch_k = min(max(k * 5, 50), len(self.index.items))

        sem_scores = self._semantic_scores(query)
        top_indices = np.argsort(sem_scores)[::-1][:fetch_k]

        candidates: list[tuple[int, float]] = []
        name_boosts = self._exact_name_boosts(query)
        role_boosts = self._role_slug_boosts(query)
        for i in top_indices:
            i = int(i)
            kw = self._keyword_score(query, self.index.items[i])
            combined = (
                float(sem_scores[i])
                + 0.35 * kw
                + name_boosts.get(i, 0.0)
                + role_boosts.get(i, 0.0)
            )
            candidates.append((i, combined))

        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = self._apply_filters(candidates, filters)

        if not candidates:
            candidates = [
                (
                    int(i),
                    float(sem_scores[i])
                    + 0.35 * self._keyword_score(query, self.index.items[int(i)])
                    + name_boosts.get(int(i), 0.0)
                    + role_boosts.get(int(i), 0.0),
                )
                for i in top_indices
            ]
            candidates.sort(key=lambda x: x[1], reverse=True)

        seen: set[str] = set()
        results: list[CatalogItem] = []

        for slug in slugs_for_query(query):
            item = self.index.by_slug.get(slug)
            if item and item.entity_id not in seen:
                results.append(item)
                seen.add(item.entity_id)

        for idx, _ in candidates:
            item = self.index.items[idx]
            if item.entity_id in seen:
                continue
            seen.add(item.entity_id)
            results.append(item)
            if len(results) >= k:
                break
        return results

    def retrieve_by_names(self, names: list[str]) -> list[CatalogItem]:
        found: list[CatalogItem] = []
        seen: set[str] = set()
        for name in names:
            item = self.index.get_by_name_fuzzy(name)
            if item and item.entity_id not in seen:
                found.append(item)
                seen.add(item.entity_id)
        return found


_retriever: HybridRetriever | None = None
_retriever_lock = threading.Lock()


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                _retriever = HybridRetriever(load_catalog_index())
    return _retriever


def retrieve(query: str, filters: dict[str, Any] | None = None, k: int = 10) -> list[CatalogItem]:
    return get_retriever().retrieve(query, filters=filters, k=k)
