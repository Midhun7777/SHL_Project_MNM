"""Load and index the build-time ITS catalog."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data" / "catalog.json"


class CatalogItem(BaseModel):
    entity_id: str
    name: str
    url: str
    test_type: str
    description: str = ""
    duration: str = ""
    job_levels: list[str] = Field(default_factory=list)
    remote: str = ""
    adaptive: str = ""
    languages: list[str] = Field(default_factory=list)
    keys: list[str] = Field(default_factory=list)

    def slug(self) -> str:
        return self.url.rstrip("/").split("/")[-1]

    def search_text(self) -> str:
        slug_words = self.slug().replace("-", " ")
        parts = [self.name, slug_words, self.description, " ".join(self.keys), " ".join(self.job_levels)]
        return " ".join(p for p in parts if p)


class CatalogIndex:
    """In-memory catalog with URL/name lookup helpers."""

    def __init__(self, items: list[CatalogItem]) -> None:
        self.items = items
        self.by_url: dict[str, CatalogItem] = {i.url: i for i in items}
        self.by_entity_id: dict[str, CatalogItem] = {i.entity_id: i for i in items}
        self.by_slug: dict[str, CatalogItem] = {i.slug(): i for i in items}
        self.by_name_lower: dict[str, CatalogItem] = {i.name.lower(): i for i in items}
        self.all_urls: set[str] = set(self.by_url)

    def get_by_name_fuzzy(self, name: str) -> CatalogItem | None:
        needle = name.strip().lower()
        if not needle:
            return None
        if needle in self.by_name_lower:
            return self.by_name_lower[needle]
        for item in self.items:
            if needle in item.name.lower() or item.name.lower() in needle:
                return item
        return None

    def find_by_slugs(self, slugs: list[str]) -> list[CatalogItem]:
        found: list[CatalogItem] = []
        for slug in slugs:
            item = self.by_slug.get(slug)
            if item:
                found.append(item)
        return found


@lru_cache(maxsize=1)
def load_catalog_index() -> CatalogIndex:
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CATALOG_PATH}. Run: python scripts/build_catalog.py"
        )
    raw: list[dict[str, Any]] = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    items = [CatalogItem.model_validate(row) for row in raw]
    return CatalogIndex(items)
