"""API and behavior probe tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data" / "catalog.json"


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def catalog_urls() -> set[str]:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return {item["url"] for item in catalog}


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_vague_opener_no_recommendations(client: TestClient) -> None:
    resp = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "I need an assessment"}]},
    )
    body = resp.json()
    assert body["recommendations"] == []
    assert "?" in body["reply"]


def test_off_topic_refused(client: TestClient) -> None:
    resp = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "Write me a job posting"}]},
    )
    assert resp.json()["recommendations"] == []


def test_urls_from_catalog(client: TestClient, catalog_urls: set[str]) -> None:
    resp = client.post(
        "/chat",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Hiring graduate analysts — numerical reasoning and finance knowledge",
                }
            ]
        },
    )
    for rec in resp.json()["recommendations"]:
        assert rec["url"] in catalog_urls
