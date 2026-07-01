"""Unit tests for catalog build output."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data" / "catalog.json"


@pytest.fixture(scope="module")
def catalog() -> list[dict]:
    assert CATALOG_PATH.exists(), "Run scripts/build_catalog.py first"
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def test_catalog_not_empty(catalog: list[dict]) -> None:
    assert len(catalog) >= 300


def test_required_fields(catalog: list[dict]) -> None:
    required = {
        "entity_id",
        "name",
        "url",
        "test_type",
        "description",
        "duration",
        "job_levels",
        "remote",
        "adaptive",
        "languages",
    }
    for item in catalog:
        assert required.issubset(item.keys())
        assert item["url"].startswith("https://www.shl.com/products/product-catalog/view/")


def test_no_job_solution_bundles(catalog: list[dict]) -> None:
    for item in catalog:
        slug = item["url"].rstrip("/").split("/")[-1]
        assert not slug.endswith("-solution"), f"Job bundle leaked: {item['name']}"


def test_unique_entity_id_and_url(catalog: list[dict]) -> None:
    assert len({x["entity_id"] for x in catalog}) == len(catalog)
    assert len({x["url"] for x in catalog}) == len(catalog)


def test_test_type_codes(catalog: list[dict]) -> None:
    allowed = set("ABCDEKPS,")
    for item in catalog:
        assert item["test_type"]
        assert all(ch in allowed for ch in item["test_type"])


def test_trace_urls_present(catalog: list[dict]) -> None:
    slugs = {x["url"].rstrip("/").split("/")[-1] for x in catalog}
    must_have = [
        "customer-service-phone-simulation",
        "sql-new",
        "graduate-scenarios",
        "opq-leadership-report",
    ]
    for slug in must_have:
        assert slug in slugs
