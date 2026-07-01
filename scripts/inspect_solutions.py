"""Detailed inspection of Solution-tagged and borderline catalog entries."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "catalog_raw.json"


def repair_raw_json(text: str) -> str:
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


def load_raw() -> list[dict]:
    return json.loads(repair_raw_json(RAW_PATH.read_text(encoding="utf-8", errors="replace")))


def main() -> None:
    items = load_raw()
    solutions = [x for x in items if "solution" in x.get("name", "").lower()]
    print("=== All *Solution* entries ===")
    for x in solutions:
        print(json.dumps({
            "name": x["name"],
            "link": x["link"],
            "keys": x.get("keys"),
            "job_levels": x.get("job_levels"),
            "duration": x.get("duration"),
        }, indent=2))
        print()

    print("=== SQL-related ===")
    for x in items:
        if "sql" in x.get("name", "").lower() or "sql" in x.get("link", ""):
            print(x["name"], "|", x["link"])

    print("\n=== Excel/Word 365 ===")
    for x in items:
        if "excel-365" in x.get("link", "") or "word-365" in x.get("link", ""):
            print(x["name"], "|", x["link"], "|", x.get("keys"))

    # Items with very long multi-key spanning development bundles?
    multi = [x for x in items if len(x.get("keys", [])) >= 4]
    print(f"\n=== Items with 4+ keys ({len(multi)}) ===")
    for x in multi[:10]:
        print(x["name"], "|", x.get("keys"))


if __name__ == "__main__":
    main()
