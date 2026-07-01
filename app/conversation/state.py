"""Reconstruct conversation state from stateless message history."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.catalog import CatalogIndex, load_catalog_index
from app.conversation.gates import is_legal_refusal, is_off_topic, is_prompt_injection
from app.conversation.intents import Intent, classify_intent
from app.models import Message, Recommendation


@dataclass
class ConversationState:
    messages: list[Message]
    user_messages: list[str] = field(default_factory=list)
    turn_count: int = 0
    all_user_text: str = ""
    latest_user: str = ""
    prior_recommendations: list[Recommendation] = field(default_factory=list)
    intent: Intent = Intent.CLARIFY
    refusal_reason: str | None = None
    filters: dict = field(default_factory=dict)
    add_test_types: set[str] = field(default_factory=set)
    drop_slugs: set[str] = field(default_factory=set)


CATALOG_URL_RE = re.compile(
    r"https://www\.shl\.com/products/product-catalog/view/[a-z0-9\-()%]+/?",
    re.I,
)


def _extract_prior_recs(messages: list[Message], index: CatalogIndex) -> list[Recommendation]:
    recs: list[Recommendation] = []
    seen: set[str] = set()
    for msg in messages:
        if msg.role != "assistant":
            continue
        for url in CATALOG_URL_RE.findall(msg.content):
            url = url.rstrip("/") + "/"
            item = index.by_url.get(url)
            if item and url not in seen:
                recs.append(
                    Recommendation(name=item.name, url=item.url, test_type=item.test_type)
                )
                seen.add(url)
    return recs


def _build_filters(all_user_text: str) -> dict:
    low = all_user_text.lower()
    filters: dict = {}

    test_types: set[str] = set()
    if any(w in low for w in ("personality", "opq", "behaviour", "behavior")):
        test_types.add("P")
    if any(w in low for w in ("cognitive", "reasoning", "verify", "aptitude", "ability")):
        test_types.add("A")
    if any(w in low for w in ("simulation", "simulated", "live coding", "svar")):
        test_types.update({"S", "K"})
    if any(w in low for w in ("situational", "judgement", "judgment", "scenarios")):
        test_types.add("B")
    if any(w in low for w in ("knowledge", "skills", "technical test", "java", "sql", "excel")):
        test_types.add("K")
    if any(w in low for w in ("competenc", "gsa", "global skills")):
        test_types.add("C")
    if test_types:
        filters["test_types"] = test_types

    job_levels: set[str] = set()
    if any(w in low for w in ("entry-level", "entry level", "contact centre", "contact center")):
        job_levels.add("Entry-Level")
    if "graduate" in low or "trainee" in low:
        job_levels.add("Graduate")
    if any(w in low for w in ("senior", "cxo", "director", "executive", "leadership")):
        job_levels.update({"Manager", "Executive", "Director", "Mid-Professional"})
    if job_levels:
        filters["job_levels"] = job_levels

    if "quick" in low or "short" in low:
        filters["max_duration_minutes"] = 30

    if "english us" in low or re.search(r"\benglish\.?\s*$", low) or re.search(r"\bus\b", low):
        filters["languages"] = {"English (USA)"}

    return filters


def _parse_refine_signals(latest: str, all_text: str) -> tuple[set[str], set[str]]:
    low = latest.lower()
    all_low = all_text.lower()
    drop_slugs: set[str] = set()
    add_types: set[str] = set()

    if re.search(r"\bdrop\b.*\bopq\b|\bremove\b.*\bopq\b", all_low):
        drop_slugs.add("occupational-personality-questionnaire-opq32r")
    if re.search(r"\bdrop\b.*\brest\b|\bremove\b.*\brest\b", all_low):
        drop_slugs.add("restful-web-services-new")
    if "add personality" in low or "add opq" in low:
        add_types.add("P")
    if "add cognitive" in low or "verify g" in low:
        add_types.add("A")
    if "simulation" in low and "add" in low:
        add_types.update({"S", "K"})
    if "situational" in low and "add" in low:
        add_types.add("B")
    if "aws" in low and "add" in low:
        add_types.add("K")
    if "docker" in low and "add" in low:
        add_types.add("K")
    return drop_slugs, add_types


def build_state(messages: list[Message]) -> ConversationState:
    index = load_catalog_index()
    user_msgs = [m.content.strip() for m in messages if m.role == "user" and m.content.strip()]
    latest = user_msgs[-1] if user_msgs else ""
    all_text = " ".join(user_msgs)
    prior = _extract_prior_recs(messages, index)

    state = ConversationState(
        messages=messages,
        user_messages=user_msgs,
        turn_count=len(user_msgs),
        all_user_text=all_text,
        latest_user=latest,
        prior_recommendations=prior,
        filters=_build_filters(all_text),
    )

    drop, add = _parse_refine_signals(latest, all_text)
    state.drop_slugs = drop
    state.add_test_types = add

    if not latest:
        state.intent = Intent.CLARIFY
        return state

    if is_prompt_injection(latest):
        state.intent = Intent.REFUSE
        state.refusal_reason = "injection"
        return state

    if is_legal_refusal(latest):
        state.intent = Intent.REFUSE
        state.refusal_reason = "legal"
        return state

    if is_off_topic(latest) and not prior:
        state.intent = Intent.REFUSE
        state.refusal_reason = "off_topic"
        return state

    state.intent = classify_intent(
        latest,
        turn_count=state.turn_count,
        has_prior_recs=bool(prior),
        all_user_text=all_text,
    )
    return state
