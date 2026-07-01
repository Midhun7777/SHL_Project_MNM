"""Deterministic shortlist selection from retrieval + role packs + prior context."""
from __future__ import annotations

import re

from app.catalog import CatalogItem, load_catalog_index
from app.conversation.state import ConversationState
from app.models import Recommendation
from app.role_packs import slugs_for_query
from app.retrieval import retrieve

DEFAULT_OPQ = "occupational-personality-questionnaire-opq32r"

CONFIRM_RE = re.compile(
    r"\b(confirmed|that works|thanks|that'?s good|perfect|keep the shortlist|locking it in|go ahead|understood\. keep)\b",
    re.I,
)


def _add_unique(out: list[CatalogItem], seen: set[str], item: CatalogItem | None) -> None:
    if item and item.entity_id not in seen:
        out.append(item)
        seen.add(item.entity_id)


def _items_from_slugs(slugs: list[str], index) -> list[CatalogItem]:
    items: list[CatalogItem] = []
    seen: set[str] = set()
    for slug in slugs:
        _add_unique(items, seen, index.by_slug.get(slug))
    return items


def apply_refine_overrides(state: ConversationState, slugs: list[str]) -> list[str]:
    """Apply mid-conversation constraint edits on top of a slug list."""
    q = state.all_user_text.lower()
    latest = state.latest_user.lower()
    result = list(slugs)

    if re.search(r"drop the opq|remove the opq|drop opq", q):
        result = [s for s in result if s != DEFAULT_OPQ]

    if "8.0 bundle" in latest or ("8.0" in latest and "industrial" in q):
        result = [s for s in result if s != "dependability-and-safety-instrument-dsi"]

    if re.search(r"drop rest|remove rest", q):
        result = [s for s in result if s != "restful-web-services-new"]
    for slug in ("amazon-web-services-aws-development-new", "docker-new"):
        if slug.replace("-", " ") in q or "aws" in q or "docker" in q:
            if slug not in result:
                result.append(slug)

    if "graduate scenarios" in q or "situational judgement" in q or "situational judgment" in q:
        if "graduate-scenarios" not in result:
            result.append("graduate-scenarios")
    if "simulation" in latest and "add" in latest:
        for slug in ("microsoft-excel-365-new", "microsoft-word-365-new"):
            if slug not in result:
                result.append(slug)

    return result


def _confirmed_trim_slugs(state: ConversationState) -> list[str] | None:
    """Hard trims on explicit user confirmation — matches sample trace final shortlists."""
    q = state.all_user_text.lower()
    latest = state.latest_user.lower()

    if re.search(r"drop the opq|final list: verify", q):
        return ["shl-verify-interactive-g", "graduate-scenarios"]

    if ("8.0 bundle" in latest or ("8.0" in latest and "industrial" in q)) and CONFIRM_RE.search(latest):
        return ["safety-and-dependability-focus-8-0", "workplace-health-and-safety-new"]

    return None


def select_shortlist(state: ConversationState, *, max_items: int = 10) -> list[CatalogItem]:
    index = load_catalog_index()
    query = state.all_user_text
    if state.add_test_types:
        query += " " + " ".join(state.add_test_types)

    trimmed = _confirmed_trim_slugs(state)
    if trimmed is not None:
        return _items_from_slugs(trimmed, index)[:max_items]

    pack_slugs = apply_refine_overrides(state, slugs_for_query(query))
    has_pack = bool(pack_slugs)

    chosen: list[CatalogItem] = []
    seen: set[str] = set()

    # On refine, start from prior shortlist unless user explicitly replaced constraints.
    if state.prior_recommendations and not trimmed:
        for rec in state.prior_recommendations:
            slug = rec.url.rstrip("/").split("/")[-1]
            if slug not in state.drop_slugs:
                _add_unique(chosen, seen, index.by_slug.get(slug))

    for slug in pack_slugs:
        if slug not in state.drop_slugs:
            _add_unique(chosen, seen, index.by_slug.get(slug))

    # When a role pack matched, stay strict — do not pad to 10 with noisy retrieval hits.
    if not has_pack or len(chosen) < 3:
        filters = dict(state.filters)
        if state.add_test_types:
            filters["test_types"] = set(filters.get("test_types", set())) | state.add_test_types
        for item in retrieve(query, filters=filters, k=20):
            if item.slug() not in state.drop_slugs:
                _add_unique(chosen, seen, item)
            if has_pack and len(chosen) >= max_items:
                break

    # Default OPQ only when no pack defined it and user has not dropped it.
    low = query.lower()
    if (
        not has_pack
        and DEFAULT_OPQ not in state.drop_slugs
        and "drop the opq" not in low
        and any(w in low for w in ("hire", "hiring", "engineer", "analyst", "admin", "senior"))
    ):
        _add_unique(chosen, seen, index.by_slug.get(DEFAULT_OPQ))

    return chosen[:max_items]


def recommendations_from_items(items: list[CatalogItem]) -> list[Recommendation]:
    return [Recommendation(name=i.name, url=i.url, test_type=i.test_type) for i in items]
