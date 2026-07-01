"""Retrieval-only Recall@10 evaluation against labeled traces."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.retrieval import get_retriever  # noqa: E402

TRACES_PATH = ROOT / "data" / "traces.json"


def recall_at_k(retrieved_slugs: list[str], expected: set[str], k: int = 10) -> float:
    if not expected:
        return 1.0
    top = set(retrieved_slugs[:k])
    return len(top & expected) / len(expected)


def main() -> int:
    traces = json.loads(TRACES_PATH.read_text(encoding="utf-8"))
    retriever = get_retriever()
    retriever.warm()

    print(f"{'Trace':<6} {'Recall@10':>10}  Expected  Retrieved (top 10 slugs)")
    print("-" * 90)

    recalls: list[float] = []
    for trace in traces:
        query = " ".join(trace["user_messages"])
        expected = set(trace["expected_slugs"])
        items = retriever.retrieve(query, filters={}, k=10)
        got_slugs = [i.slug() for i in items]
        r = recall_at_k(got_slugs, expected, k=10)
        recalls.append(r)
        hit = f"{r:.0%}"
        print(f"{trace['id']:<6} {hit:>10}  {len(expected)} items")
        missing = expected - set(got_slugs)
        if missing:
            print(f"         missing: {', '.join(sorted(missing))}")

    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    print("-" * 90)
    print(f"Mean Recall@10: {mean_recall:.1%} ({sum(recalls):.1f}/{len(recalls)} traces fully covered at proportion)")
    print(f"Mean Recall@10 (average per-item): {mean_recall:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
