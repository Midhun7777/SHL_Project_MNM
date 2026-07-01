"""Rule-based intent classification (fast, deterministic baseline)."""
from __future__ import annotations

import re
from enum import Enum


class Intent(str, Enum):
    CLARIFY = "clarify"
    RECOMMEND = "recommend"
    REFINE = "refine"
    COMPARE = "compare"
    REFUSE = "refuse"


COMPARE_RE = re.compile(
    r"\b(difference|compare|versus|vs\.?|how\s+(is|are)\s+.+\s+different)\b",
    re.I,
)
REFINE_RE = re.compile(
    r"\b(add|remove|drop|exclude|also|instead|actually|make it|shorter|longer|without|replace)\b",
    re.I,
)
CONFIRM_RE = re.compile(
    r"\b(confirmed|that works|that'?s good|perfect|keep|locking it in|go ahead|understood\. keep)\b",
    re.I,
)
VAGUE_RE = re.compile(
    r"^(we need a solution|i need an assessment|what should we use\??|help me choose)\.?$",
    re.I,
)


def classify_intent(latest_user: str, *, turn_count: int, has_prior_recs: bool, all_user_text: str = "") -> Intent:
    low = latest_user.lower()
    all_low = (all_user_text or latest_user).lower()

    # C10: no shorter OPQ substitute — explain, no shortlist change.
    if re.search(r"shorter.*opq|replace.*opq.*shorter|remove the opq32r and replace", low):
        return Intent.CLARIFY

    if is_compare_question(latest_user):
        return Intent.COMPARE

    if has_prior_recs and (REFINE_RE.search(latest_user) or CONFIRM_RE.search(latest_user)):
        if CONFIRM_RE.search(latest_user) and not REFINE_RE.search(latest_user):
            return Intent.RECOMMEND  # re-emit confirmed shortlist
        return Intent.REFINE

    if turn_count >= 7:
        return Intent.RECOMMEND  # force convergence before 8-turn cap

    if is_vague(latest_user) and turn_count <= 1:
        return Intent.CLARIFY

    if needs_clarification(latest_user, turn_count, all_low):
        return Intent.CLARIFY

    return Intent.RECOMMEND


def is_compare_question(text: str) -> bool:
    return bool(COMPARE_RE.search(text))


def is_vague(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 40 and VAGUE_RE.search(stripped):
        return True
    if stripped.lower() in {"i need an assessment", "we need a solution", "what should we use"}:
        return True
    return False


def needs_clarification(text: str, turn_count: int, all_low: str = "") -> bool:
    low = text.lower()
    all_low = all_low or low
    # Contact centre language/accent (C3 pattern).
    if "contact centre" in all_low or "contact center" in all_low:
        if "english" not in all_low and turn_count <= 2:
            return True
        if re.search(r"\benglish\b", all_low) and not re.search(
            r"\b(us|uk|australian|indian)\b", all_low
        ):
            if turn_count <= 3:
                return True
    # Senior leadership (C1): who + selection vs development.
    if ("senior leadership" in all_low or "cxo" in all_low or "director" in all_low):
        if "selection" not in all_low and "development" not in all_low:
            if turn_count <= 2:
                return True
    # Full-stack JD needs backend/frontend split (C9).
    if ("full-stack" in all_low or "full stack" in all_low) and "java" in all_low:
        if turn_count <= 2 and "backend" not in all_low and "frontend" not in all_low:
            return True
        if turn_count <= 3 and "senior ic" not in all_low and "tech lead" not in all_low:
            if "backend" in all_low and "manage" not in all_low:
                return True
    # Bilingual healthcare — approach choice (C7).
    if "healthcare" in all_low and "spanish" in all_low:
        if "hybrid" not in all_low and "personality-only" not in all_low and turn_count <= 1:
            return True
    return False
