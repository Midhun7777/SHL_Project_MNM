"""One-off analysis of raw catalog for ITS scoping decisions."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "catalog_raw.json"


def repair_raw_json(text: str) -> str:
    """Fix scrape artifacts: literal newlines inside JSON string values."""
    # Collapse whitespace in name fields (primary source of broken JSON).
    text = re.sub(
        r'("name"\s*:\s*")([^"]*?)"',
        lambda m: m.group(1) + re.sub(r"\s+", " ", m.group(2).strip()) + '"',
        text,
        flags=re.DOTALL,
    )
    # Generic fix for other string fields with embedded newlines.
    def _fix_string(match: re.Match[str]) -> str:
        inner = match.group(1)
        inner = inner.replace("\r", " ").replace("\n", " ")
        inner = re.sub(r"  +", " ", inner)
        return f'"{inner}"'

    return re.sub(r'"((?:[^"\\]|\\.)*)"', _fix_string, text)


def load_raw() -> list[dict]:
    text = RAW_PATH.read_text(encoding="utf-8", errors="replace")
    return json.loads(repair_raw_json(text))


def main() -> None:
    items = load_raw()
    print(f"Total raw items: {len(items)}")
    all_keys = sorted({k for item in items for k in item})
    print(f"Fields present: {all_keys}")

    solution_names = [x["name"] for x in items if "solution" in x.get("name", "").lower()]
    print(f"\nNames containing 'solution': {len(solution_names)}")
    for n in solution_names[:20]:
        print(f"  - {n}")

    key_counts = Counter()
    for item in items:
        for k in item.get("keys", []):
            key_counts[k] += 1
    print("\nKeys distribution:", dict(key_counts))

    # Expected trace URLs (ground truth for scoping validation)
    trace_urls = {
        "opq32r": "occupational-personality-questionnaire-opq32r",
        "opq-ucf": "opq-universal-competency-report-2-0",
        "opq-leadership": "opq-leadership-report",
        "live-coding": "smart-interview-live-coding",
        "linux": "linux-programming-general",
        "networking": "networking-and-implementation-new",
        "verify-g": "shl-verify-interactive-g",
        "svar-us": "svar-spoken-english-us-new",
        "contact-center-sim": "contact-center-call-simulation-new",
        "entry-level-cs": "entry-level-customer-serv-retail-and-contact-center",
        "cs-phone-solution": "customer-service-phone-simulation",
        "numerical": "shl-verify-interactive-numerical-reasoning",
        "financial-acct": "financial-accounting-new",
        "basic-stats": "basic-statistics-new",
        "graduate-scenarios": "graduate-scenarios",
        "gsa": "global-skills-assessment",
        "gsa-dev": "global-skills-development-report",
        "opq-mq-sales": "opq-mq-sales-report",
        "sales-transformation": "salestransformationreport2-0-individualcontributor",
        "dsi": "dependability-and-safety-instrument-dsi",
        "safety-8": "safety-and-dependability-focus-8-0",
        "whs": "workplace-health-and-safety-new",
        "hipaa": "hipaa-security",
        "med-term": "medical-terminology-new",
        "word365-essentials": "microsoft-word-365-essentials-new",
        "ms-excel": "ms-excel-new",
        "ms-word": "ms-word-new",
        "excel365": "microsoft-excel-365-new",
        "word365": "microsoft-word-365-new",
        "java-adv": "core-java-advanced-level-new",
        "spring": "spring-new",
        "rest": "restful-web-services-new",
        "sql": "sql-new",
        "aws": "amazon-web-services-aws-development-new",
        "docker": "docker-new",
    }

    print("\n--- Trace URL lookup in raw feed ---")
    for label, slug in trace_urls.items():
        matches = [x for x in items if slug in x.get("link", "")]
        if matches:
            m = matches[0]
            print(f"  OK {label}: {m['name']} | keys={m.get('keys')}")
        else:
            print(f"  MISSING {label}: {slug}")

    # Heuristic candidates for exclusion
    report_like = [x for x in items if "report" in x.get("name", "").lower()]
    print(f"\nItems with 'report' in name: {len(report_like)}")

    job_solution_hints = [x for x in items if any(
        kw in x.get("name", "").lower() for kw in ["job solution", "pre-packaged", "bundle"]
    )]
    print(f"Items with job-solution keywords: {len(job_solution_hints)}")


if __name__ == "__main__":
    main()
