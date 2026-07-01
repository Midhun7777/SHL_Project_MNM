"""Optional OpenAI client with safe fallbacks when unavailable."""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

log = logging.getLogger(__name__)


def llm_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def chat_completion(
    system: str,
    user: str,
    *,
    json_mode: bool = False,
    timeout: float = 20.0,
) -> str | None:
    if not llm_available():
        return None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=timeout)
        kwargs: dict[str, Any] = {
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or None
    except Exception as exc:
        log.warning("LLM call failed: %s", exc)
        return None


def parse_json_response(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None
