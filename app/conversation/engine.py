"""
Conversation state machine: gate → intent → clarify | recommend | refine | compare | refuse.

All recommendations are copied from catalog rows — never LLM-generated URLs.
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from app.catalog import load_catalog_index
from app.conversation.intents import Intent
from app.conversation.selection import recommendations_from_items, select_shortlist
from app.conversation.state import ConversationState, build_state
from app.llm import chat_completion, parse_json_response
from app.models import ChatRequest, ChatResponse, Recommendation
from app.retrieval import get_retriever

log = logging.getLogger(__name__)

EXECUTOR = ThreadPoolExecutor(max_workers=2)
CHAT_TIMEOUT_SEC = 28.0

CONFIRM_RE = re.compile(
    r"\b(confirmed|that works|thanks|that'?s good|perfect|keep the shortlist|locking it in|go ahead|understood\. keep)\b",
    re.I,
)

# Known compare pairs → catalog slug fragments (compare uses catalog text only).
COMPARE_PAIRS: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    (("dsi", "dependability and safety"), ("8.0", "safety & dependability", "safety and dependability")),
    (("opq mq sales", "mq sales report"), ("opq32r", "opq ", "personality questionnaire")),
    (("contact center call simulation", "contact centre call simulation"), ("customer service phone",)),
]


def _validate_response(resp: ChatResponse) -> ChatResponse:
    """Ensure response matches Pydantic schema before returning."""
    return ChatResponse.model_validate(resp.model_dump())


def _select_with_llm(state: ConversationState, candidates: list, *, max_items: int = 10) -> list | None:
    if not candidates:
        return None
    catalog_block = "\n".join(
        f"- id={i.entity_id} | {i.name} | type={i.test_type} | duration={i.duration}"
        for i in candidates[:30]
    )
    system = (
        "You select SHL assessments for a recruiter. Choose ONLY from the candidate list. "
        "Use ONLY catalog facts in your reply. "
        'Return JSON: {"entity_ids": ["..."], "reply": "short message"}. '
        "Pick 1-10 items. Never invent IDs."
    )
    user = (
        f"Conversation:\n{state.all_user_text}\n\n"
        f"Latest user message:\n{state.latest_user}\n\n"
        f"Candidates:\n{catalog_block}"
    )
    for attempt in range(2):
        raw = chat_completion(system, user, json_mode=True)
        if not raw:
            continue
        parsed = parse_json_response(raw)
        if not parsed:
            continue
        id_set = {str(x) for x in parsed.get("entity_ids", [])}
        chosen = [c for c in candidates if c.entity_id in id_set]
        if chosen:
            return chosen[:max_items]
    return None


def _handle_clarify(state: ConversationState) -> ChatResponse:
    low = state.all_user_text.lower()
    latest = state.latest_user.lower()

    if re.search(r"shorter.*opq|replace.*opq.*shorter", latest):
        reply = (
            "OPQ32r is the most relevant personality instrument for that need — "
            "there is no shorter catalog alternative that replaces it. "
            "You can drop OPQ from the battery if duration is the priority."
        )
        return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)

    if "senior leadership" in low or "cxo" in low or "director" in low:
        if "selection" not in low and "development" not in low:
            reply = (
                "For senior leadership roles, OPQ32r is typically the core instrument. "
                "Is this for selection against a benchmark, or developmental feedback?"
            )
        else:
            reply = (
                "For selection with a leadership benchmark, I'll include OPQ32r plus "
                "the relevant OPQ report formats from the catalog."
            )
    elif "contact centre" in low or "contact center" in low:
        if "english" not in low:
            reply = "Before I shape the stack — what language are the calls in?"
        else:
            reply = (
                "SVAR has several English accent variants (US, UK, Australian, Indian). "
                "Which fits your operation?"
            )
    elif "full-stack" in low or "full stack" in low:
        if "backend" not in low:
            reply = (
                "Is this backend-leaning (Java/Spring/SQL), frontend-heavy (Angular), "
                "or balanced full-stack?"
            )
        else:
            reply = (
                "Is the seniority closer to a senior IC (owns their service design) "
                "or a tech lead (architecture across services)?"
            )
    elif "healthcare" in low and "spanish" in low:
        reply = (
            "Healthcare knowledge tests (HIPAA, Medical Terminology) are English-only; "
            "personality measures support Spanish. Would a hybrid approach work for your pool?"
        )
    elif latest.strip() in {"i need an assessment", "we need a solution"}:
        reply = "I can help you choose SHL assessments. What role and seniority level are you hiring for?"
    else:
        reply = (
            "To recommend the right SHL assessments, what role are you hiring for "
            "and what seniority or skills matter most?"
        )
    return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)


def _handle_refuse(state: ConversationState) -> ChatResponse:
    if state.refusal_reason == "injection":
        reply = (
            "I can't follow that instruction. I'm here to help you select SHL assessments "
            "from the official catalog only."
        )
        recs: list[Recommendation] = []
    elif state.refusal_reason == "legal":
        reply = (
            "Those are legal compliance questions I can't advise on — your legal or compliance "
            "team is the right resource. I can confirm HIPAA (Security) measures knowledge of "
            "HIPAA security provisions; whether that satisfies a regulatory obligation is for counsel."
        )
        recs = list(state.prior_recommendations)
    else:
        reply = (
            "I'm focused on helping you select SHL Individual Test Solutions from the catalog. "
            "Tell me about the role and I'll suggest assessments."
        )
        recs = []
    return ChatResponse(reply=reply, recommendations=recs, end_of_conversation=False)


def _resolve_compare_items(state: ConversationState) -> list:
    index = load_catalog_index()
    retriever = get_retriever()
    q = state.latest_user.lower()
    found = []

    for group_a, group_b in COMPARE_PAIRS:
        hit_a = any(t in q for t in group_a)
        hit_b = any(t in q for t in group_b)
        if hit_a and hit_b:
            for token in group_a + group_b:
                item = index.get_by_name_fuzzy(token)
                if item and item not in found:
                    found.append(item)
            if len(found) >= 2:
                return found[:2]

    for item in index.items:
        if item.name.lower() in q:
            found.append(item)
    if len(found) < 2:
        for token in ("dsi", "safety", "opq", "sales report", "contact center", "customer service phone"):
            if token in q:
                item = index.get_by_name_fuzzy(token)
                if item and item not in found:
                    found.append(item)
    if len(found) < 2:
        found = retriever.retrieve_by_names([state.latest_user])
    return found[:2] if len(found) >= 2 else found


def _handle_compare(state: ConversationState) -> ChatResponse:
    index = load_catalog_index()
    items = _resolve_compare_items(state)

    if len(items) < 2:
        return ChatResponse(
            reply="I couldn't find both products in the SHL catalog to compare. Could you name them exactly?",
            recommendations=[],
            end_of_conversation=False,
        )

    a, b = items[0], items[1]
    reply = (
        f"{a.name} ({a.test_type}, {a.duration or 'duration n/a'}): "
        f"{a.description[:350].strip()}…\n\n"
        f"{b.name} ({b.test_type}, {b.duration or 'duration n/a'}): "
        f"{b.description[:350].strip()}…\n\n"
        "Both are distinct catalog products — compare duration, languages, and keys above."
    )
    # Sample traces: compare turns carry explanation only, recommendations stay empty.
    return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)


def _handle_recommend(state: ConversationState, *, end: bool = False) -> ChatResponse:
    items = select_shortlist(state, max_items=10)
    if not items:
        return _handle_clarify(state)

    # Optional LLM re-rank — only reorder/shrink, never expand beyond selected items.
    llm_pick = _select_with_llm(state, items)
    if llm_pick:
        items = llm_pick

    recs = recommendations_from_items(items[:10])

    if state.intent == Intent.REFINE:
        reply = "Updated shortlist based on your latest constraints:"
    elif end or CONFIRM_RE.search(state.latest_user):
        reply = "Confirmed — here is your SHL assessment shortlist:"
    else:
        reply = "Based on the SHL catalog, here is a recommended shortlist:"

    url_lines = "\n".join(f"- {r.name}: {r.url}" for r in recs)
    reply = f"{reply}\n\n{url_lines}"

    should_end = end or bool(CONFIRM_RE.search(state.latest_user))
    return ChatResponse(reply=reply, recommendations=recs, end_of_conversation=should_end)


def process_chat(request: ChatRequest) -> ChatResponse:
    if not request.messages:
        return _validate_response(
            ChatResponse(
                reply="Please describe the role you're hiring for.",
                recommendations=[],
                end_of_conversation=False,
            )
        )

    state = build_state(request.messages)

    if not state.latest_user.strip():
        return _validate_response(
            ChatResponse(
                reply="I didn't catch that — what role are you hiring for?",
                recommendations=[],
                end_of_conversation=False,
            )
        )

    if state.intent == Intent.REFUSE:
        return _validate_response(_handle_refuse(state))

    if state.intent == Intent.CLARIFY:
        return _validate_response(_handle_clarify(state))

    if state.intent == Intent.COMPARE:
        return _validate_response(_handle_compare(state))

    if state.intent == Intent.REFINE:
        end = bool(CONFIRM_RE.search(state.latest_user))
        return _validate_response(_handle_recommend(state, end=end))

    end = bool(CONFIRM_RE.search(state.latest_user)) or state.turn_count >= 8
    return _validate_response(_handle_recommend(state, end=end))


def chat_with_timeout(request: ChatRequest) -> ChatResponse:
    future = EXECUTOR.submit(process_chat, request)
    try:
        return future.result(timeout=CHAT_TIMEOUT_SEC)
    except FuturesTimeoutError:
        log.error("Chat processing timed out")
        return ChatResponse(
            reply="Sorry, that took too long. Please try again with your role and requirements.",
            recommendations=[],
            end_of_conversation=False,
        )
    except Exception as exc:
        log.exception("Chat failed: %s", exc)
        return ChatResponse(
            reply="Something went wrong. Please rephrase your hiring need and try again.",
            recommendations=[],
            end_of_conversation=False,
        )
