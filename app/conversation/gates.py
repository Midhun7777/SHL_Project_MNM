"""Safety and scope gates — run before any retrieval or LLM."""
from __future__ import annotations

import re

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"disregard\s+(your|the)\s+(instructions|rules|prompt)",
    r"reveal\s+(your|the)\s+(system\s+)?prompt",
    r"you\s+are\s+now\s+",
    r"act\s+as\s+(?!a recruiter)",
    r"jailbreak",
    r"bypass\s+(your\s+)?(rules|restrictions)",
]

OFF_TOPIC_PATTERNS = [
    r"\bwrite\s+(me\s+)?a\s+job\s+(posting|description)\b",
    r"\blegal(ly)?\s+(required|obligation|advice)\b",
    r"\b(am i|are we)\s+legally\b",
    r"\bemployment\s+law\b",
    r"\bsalary\s+(range|negotiation)\b",
    r"\bvisa\s+sponsor",
    r"\bgeneral\s+hiring\s+advice\b",
]

SHL_SCOPE_HINT = re.compile(
    r"\b(assessment|test|opq|verify|simulation|personality|cognitive|shl|battery|shortlist)\b",
    re.I,
)


def is_prompt_injection(text: str) -> bool:
    low = text.lower()
    return any(re.search(p, low) for p in INJECTION_PATTERNS)


def is_off_topic(text: str) -> bool:
    low = text.lower()
    if SHL_SCOPE_HINT.search(low):
        return False
    return any(re.search(p, low) for p in OFF_TOPIC_PATTERNS)


def is_legal_refusal(text: str) -> bool:
    low = text.lower()
    return bool(re.search(r"\b(hipaa|legally required|legal requirement|satisfy that requirement)\b", low))
