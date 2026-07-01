"""
Build-time catalog fetch and transform for SHL Individual Test Solutions.

Run once (or when the upstream feed changes):
    python scripts/build_catalog.py

Outputs:
    data/catalog_raw.json  — verbatim feed (with JSON repair on save)
    data/catalog.json      — normalized, ITS-scoped catalog for the service
"""
from __future__ import annotations

import json
import logging
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "catalog_raw.json"
CATALOG_PATH = DATA_DIR / "catalog.json"

FEED_URL = (
    "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
)

# SHL legend: map catalog `keys` tags to single-letter test_type codes.
KEY_TO_TEST_TYPE: dict[str, str] = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

# Stable ordering when an item has multiple keys (matches sample-conversation tables).
TEST_TYPE_ORDER = ("A", "B", "C", "D", "E", "K", "P", "S")

# ---------------------------------------------------------------------------
# ITS scoping (Individual Test Solutions vs Job-Focused / bundled Solutions)
# ---------------------------------------------------------------------------
# The raw feed has NO explicit product_type / solution_type field. Cross-checking
# link slugs against shl.com and the 10 sample traces shows:
#
#   • 377 total rows in the feed (May 2026 scrape).
#   • 7 rows whose names end with "Solution" — pre-packaged Job-Focused Assessment
#     bundles (Competencies + Personality, or multi-construct packages), NOT
#     standalone Individual Test Solutions.
#   • "Customer Service Phone Simulation" (slug …/simulation/) IS in-scope and
#     appears in trace C3; it is a distinct catalog row from
#     "Customer Service Phone Solution" (slug …/solution/), which is excluded.
#   • Report-only products (OPQ Leadership Report, OPQ MQ Sales Report, etc.) and
#     sector bundles referenced in traces (Safety & Dependability 8.0) ARE kept.
#   • Entry Level Cashier Solution page on shl.com describes "Job-Focused
#     Assessments" / pre-packaged role bundles — confirms exclusion pattern.
#
# Rule: exclude rows whose link slug ends with "-solution" (Job-Focused bundles).
# All other rows in the feed are treated as Individual Test Solutions.
# Expected retained count: 377 − 7 = 370 (sanity-checked against trace URLs).
# ---------------------------------------------------------------------------
JOB_SOLUTION_LINK_SUFFIX = "-solution"

# Known scrape artifacts: repair display names keyed by canonical URL slug fragment.
NAME_OVERRIDES_BY_SLUG: dict[str, str] = {
    "microsoft-excel-365-new": "Microsoft Excel 365 (New)",
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def repair_raw_json(text: str) -> str:
    """Fix literal newlines embedded inside JSON string values from the scraper."""
    text = re.sub(
        r'("name"\s*:\s*")([^"]*?)"',
        lambda m: m.group(1) + re.sub(r"\s+", " ", m.group(2).strip()) + '"',
        text,
        flags=re.DOTALL,
    )

    def _fix_string(match: re.Match[str]) -> str:
        inner = match.group(1).replace("\r", " ").replace("\n", " ")
        inner = re.sub(r"  +", " ", inner)
        return f'"{inner}"'

    return re.sub(r'"((?:[^"\\]|\\.)*)"', _fix_string, text)


def fetch_feed() -> str:
    log.info("Fetching catalog from %s", FEED_URL)
    request = urllib.request.Request(FEED_URL, headers={"User-Agent": "SHL-Catalog-Builder/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read().decode("utf-8", errors="replace")


def load_raw_items(raw_text: str) -> list[dict[str, Any]]:
    repaired = repair_raw_json(raw_text)
    items = json.loads(repaired)
    if not isinstance(items, list):
        raise ValueError("Expected top-level JSON array in catalog feed")
    return items


def slug_from_link(link: str) -> str:
    return link.rstrip("/").split("/")[-1]


def derive_test_type(keys: list[str]) -> str:
    """
    Map `keys` array to SHL legend code(s).

    Sample conversations show multi-key products as comma-separated codes (e.g. "P,C",
    "B,S"). We emit all applicable codes in TEST_TYPE_ORDER for stable output.
    """
    codes: list[str] = []
    for key in keys or []:
        code = KEY_TO_TEST_TYPE.get(key)
        if code and code not in codes:
            codes.append(code)
    codes.sort(key=lambda c: TEST_TYPE_ORDER.index(c))
    return ",".join(codes)


def is_job_solution_bundle(item: dict[str, Any]) -> bool:
    """True for pre-packaged Job-Focused Assessment bundles (out of ITS scope)."""
    slug = slug_from_link(item.get("link", ""))
    return slug.endswith(JOB_SOLUTION_LINK_SUFFIX)


def normalize_name(item: dict[str, Any]) -> str:
    slug = slug_from_link(item.get("link", ""))
    if slug in NAME_OVERRIDES_BY_SLUG:
        return NAME_OVERRIDES_BY_SLUG[slug]
    return (item.get("name") or "").strip()


def normalize_item(item: dict[str, Any]) -> dict[str, Any]:
    keys = item.get("keys") or []
    return {
        "entity_id": str(item["entity_id"]),
        "name": normalize_name(item),
        "url": item["link"].strip(),
        "test_type": derive_test_type(keys),
        "description": (item.get("description") or "").strip(),
        "duration": item.get("duration") or "",
        "job_levels": list(item.get("job_levels") or []),
        "remote": item.get("remote", ""),
        "adaptive": item.get("adaptive", ""),
        "languages": list(item.get("languages") or []),
        "keys": list(keys),
    }


def validate_url(url: str) -> bool:
    return bool(
        re.match(
            r"^https://www\.shl\.com/products/product-catalog/view/[a-z0-9\-()%]+/?$",
            url,
            re.IGNORECASE,
        )
    )


def dedupe_items(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Keep first occurrence by entity_id, then by url."""
    seen_ids: set[str] = set()
    seen_urls: set[str] = set()
    deduped: list[dict[str, Any]] = []
    dropped = 0
    for item in items:
        eid, url = item["entity_id"], item["url"]
        if eid in seen_ids or url in seen_urls:
            dropped += 1
            continue
        seen_ids.add(eid)
        seen_urls.add(url)
        deduped.append(item)
    return deduped, dropped


def build_catalog(raw_items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stats: dict[str, Any] = {
        "raw_total": len(raw_items),
        "excluded_job_solution_bundles": [],
        "excluded_bad_status": [],
        "excluded_invalid_url": [],
        "excluded_missing_name": [],
        "dedupe_dropped": 0,
        "retained": 0,
    }

    normalized: list[dict[str, Any]] = []
    for item in raw_items:
        if is_job_solution_bundle(item):
            stats["excluded_job_solution_bundles"].append(
                {"name": item.get("name"), "url": item.get("link")}
            )
            continue
        if item.get("status") not in (None, "ok"):
            stats["excluded_bad_status"].append(item.get("link"))
            continue
        if not item.get("link") or not validate_url(item["link"]):
            stats["excluded_invalid_url"].append(item.get("link"))
            continue
        if not normalize_name(item):
            stats["excluded_missing_name"].append(item.get("link"))
            continue
        normalized.append(normalize_item(item))

    deduped, dropped = dedupe_items(normalized)
    stats["dedupe_dropped"] = dropped
    stats["retained"] = len(deduped)
    return deduped, stats


def log_stats(stats: dict[str, Any]) -> None:
    log.info("Raw feed items: %d", stats["raw_total"])
    log.info(
        "Excluded Job-Focused bundles (-solution slugs): %d",
        len(stats["excluded_job_solution_bundles"]),
    )
    for row in stats["excluded_job_solution_bundles"]:
        log.info("  - %s (%s)", row["name"], row["url"])
    if stats["excluded_bad_status"]:
        log.info("Excluded non-ok status: %d", len(stats["excluded_bad_status"]))
    if stats["excluded_invalid_url"]:
        log.info("Excluded invalid URLs: %d", len(stats["excluded_invalid_url"]))
    if stats["excluded_missing_name"]:
        log.info("Excluded missing name: %d", len(stats["excluded_missing_name"]))
    if stats["dedupe_dropped"]:
        log.info("Dedupe dropped: %d", stats["dedupe_dropped"])
    log.info("Retained ITS catalog items: %d", stats["retained"])


# Ground-truth URLs from the 10 sample conversations (must survive ITS filter).
TRACE_URL_SLUGS = [
    "occupational-personality-questionnaire-opq32r",
    "opq-universal-competency-report-2-0",
    "opq-leadership-report",
    "smart-interview-live-coding",
    "linux-programming-general",
    "networking-and-implementation-new",
    "shl-verify-interactive-g",
    "svar-spoken-english-us-new",
    "contact-center-call-simulation-new",
    "entry-level-customer-serv-retail-and-contact-center",
    "customer-service-phone-simulation",
    "shl-verify-interactive-numerical-reasoning",
    "financial-accounting-new",
    "basic-statistics-new",
    "graduate-scenarios",
    "global-skills-assessment",
    "global-skills-development-report",
    "opq-mq-sales-report",
    "salestransformationreport2-0-individualcontributor",
    "dependability-and-safety-instrument-dsi",
    "safety-and-dependability-focus-8-0",
    "workplace-health-and-safety-new",
    "hipaa-security",
    "medical-terminology-new",
    "microsoft-word-365-essentials-new",
    "ms-excel-new",
    "ms-word-new",
    "microsoft-excel-365-new",
    "microsoft-word-365-new",
    "core-java-advanced-level-new",
    "spring-new",
    "restful-web-services-new",
    "sql-new",
    "amazon-web-services-aws-development-new",
    "docker-new",
]


def verify_trace_coverage(catalog: list[dict[str, Any]]) -> None:
    catalog_slugs = {slug_from_link(item["url"]) for item in catalog}
    missing = [s for s in TRACE_URL_SLUGS if s not in catalog_slugs]
    if missing:
        raise RuntimeError(f"Trace URLs missing from catalog after ITS filter: {missing}")
    log.info("All %d trace ground-truth URLs present in catalog.", len(TRACE_URL_SLUGS))


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        raw_text = fetch_feed()
    except (urllib.error.URLError, TimeoutError) as exc:
        if RAW_PATH.exists():
            log.warning("Fetch failed (%s); using existing %s", exc, RAW_PATH)
            raw_text = RAW_PATH.read_text(encoding="utf-8", errors="replace")
        else:
            log.error("Fetch failed and no cached raw catalog exists.")
            return 1

    # Persist repaired raw JSON for auditability.
    raw_items = load_raw_items(raw_text)
    RAW_PATH.write_text(
        json.dumps(raw_items, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log.info("Wrote repaired raw catalog (%d items) to %s", len(raw_items), RAW_PATH)

    catalog, stats = build_catalog(raw_items)
    log_stats(stats)
    verify_trace_coverage(catalog)

    CATALOG_PATH.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log.info("Wrote normalized catalog to %s", CATALOG_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
