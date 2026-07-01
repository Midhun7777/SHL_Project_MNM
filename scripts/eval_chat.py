"""Enhanced eval harness — traces, probes, hallucination checks, JSON results."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.retrieval import get_retriever  # noqa: E402

TRACES_PATH = ROOT / "data" / "traces.json"
CATALOG_PATH = ROOT / "data" / "catalog.json"
RESULTS_PATH = ROOT / "data" / "eval_results.json"


def load_catalog() -> tuple[set[str], set[str]]:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return {i["url"] for i in catalog}, {i["name"] for i in catalog}


def recall_at_k(got: list[str], expected: set[str], k: int = 10) -> float:
    if not expected:
        return 1.0
    return len(set(got[:k]) & expected) / len(expected)


def run_retrieval_eval(traces: list[dict]) -> dict:
    retriever = get_retriever()
    retriever.warm()
    per_trace = {}
    recalls = []
    for trace in traces:
        query = " ".join(trace["user_messages"])
        items = retriever.retrieve(query, filters={}, k=10)
        slugs = [i.slug() for i in items]
        expected = set(trace["expected_slugs"])
        r = recall_at_k(slugs, expected)
        recalls.append(r)
        per_trace[trace["id"]] = {
            "recall_at_10": round(r, 3),
            "missing": sorted(expected - set(slugs)),
        }
    return {
        "mean_recall_at_10": round(sum(recalls) / len(recalls), 3),
        "per_trace": per_trace,
    }


def run_chat_eval(traces: list[dict], catalog_urls: set[str], catalog_names: set[str]) -> dict:
    client = TestClient(app)
    per_trace = {}
    recalls = []
    for trace in traces:
        messages: list[dict[str, str]] = []
        final_slugs: list[str] = []
        turns = 0
        for user_text in trace["user_messages"]:
            turns += 1
            messages.append({"role": "user", "content": user_text})
            resp = client.post("/chat", json={"messages": messages})
            assert resp.status_code == 200
            body = resp.json()
            for rec in body.get("recommendations", []):
                assert rec["url"] in catalog_urls, rec["url"]
                assert rec["name"] in catalog_names, rec["name"]
            messages.append({"role": "assistant", "content": body["reply"]})
            if body["recommendations"]:
                final_slugs = [r["url"].rstrip("/").split("/")[-1] for r in body["recommendations"]]
        expected = set(trace["expected_slugs"])
        r = recall_at_k(final_slugs, expected)
        recalls.append(r)
        per_trace[trace["id"]] = {
            "recall_at_10": round(r, 3),
            "turns": turns,
            "final_slugs": final_slugs,
            "missing": sorted(expected - set(final_slugs)),
        }
    return {
        "mean_recall_at_10": round(sum(recalls) / len(recalls), 3),
        "per_trace": per_trace,
    }


def run_probes() -> dict:
    import pytest

    exit_code = pytest.main(["-q", str(ROOT / "tests" / "test_probes.py"), "--tb=no"])
    return {"passed": exit_code == 0, "exit_code": exit_code}


def main() -> int:
    traces = json.loads(TRACES_PATH.read_text(encoding="utf-8"))
    catalog_urls, catalog_names = load_catalog()

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "retrieval": run_retrieval_eval(traces),
        "chat": run_chat_eval(traces, catalog_urls, catalog_names),
        "probes": run_probes(),
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(results, indent=2))
    print(f"\nResults written to {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
