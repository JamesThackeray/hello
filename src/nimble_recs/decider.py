"""Deciders pick k options out of a list, given some context.

Both stages (domain, then question) use the same primitive, so swapping models
(Nimble, Qwen, a hosted API, or a dumb keyword baseline) means changing one class.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

import httpx

from .models import Choice, Option


class DecisionError(RuntimeError):
    pass


class Decider(Protocol):
    name: str

    def choose(self, *, task: str, context: str, options: list[Option], k: int) -> Choice: ...


def render_prompt(task: str, context: str, options: list[Option], k: int) -> str:
    lines = "\n".join(f"- {o.id}: {o.label}" for o in options)
    return (
        f"{task}\n\n"
        f"{context}\n\n"
        f"Options (id: description):\n{lines}\n\n"
        f"Return exactly {k} option id(s) from the list above, most relevant first, "
        f"and a one-sentence reason."
    )


def estimate_tokens(text: str) -> int:
    """Cheap heuristic (~4 chars/token for English). Good enough for budgeting."""
    return len(text) // 4 + 1


def clean_ids(raw: list[object], valid: set[str], k: int) -> list[str]:
    out: list[str] = []
    for item in raw:
        s = str(item).strip()
        if s in valid and s not in out:
            out.append(s)
    return out[:k]


# ------------------------------------------------------------------------ ollama


SYSTEM_PROMPT = (
    "You are the decision component of a personalised suggestion engine. "
    "Use the user's current query first and their recent history second. "
    "Only choose ids that appear in the options list. Answer in JSON."
)


class OllamaDecider:
    """Calls Ollama's /api/chat with a JSON schema so the model must return valid ids.

    The schema's `enum` constrains generation to real option ids; we still validate
    the result because not every model/runtime honours every schema keyword.
    """

    def __init__(self, host: str, model: str, timeout_s: float = 120.0,
                 client: httpx.Client | None = None, retries: int = 1):
        self.name = f"ollama:{model}"
        self.model = model
        self.retries = retries
        self._client = client or httpx.Client(base_url=host, timeout=timeout_s)

    def choose(self, *, task: str, context: str, options: list[Option], k: int) -> Choice:
        if not options:
            raise DecisionError("no options to choose from")
        k = min(k, len(options))
        ids = [o.id for o in options]
        schema = {
            "type": "object",
            "properties": {
                "choices": {
                    "type": "array",
                    "items": {"type": "string", "enum": ids},
                    "minItems": k,
                    "maxItems": k,
                },
                "reason": {"type": "string"},
            },
            "required": ["choices", "reason"],
        }
        body = {
            "model": self.model,
            "stream": False,
            "format": schema,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": render_prompt(task, context, options, k)},
            ],
        }

        last_err: Exception | None = None
        for _ in range(self.retries + 1):
            try:
                resp = self._client.post("/api/chat", json=body)
                resp.raise_for_status()
                content = resp.json()["message"]["content"]
                data = json.loads(content)
                chosen = clean_ids(data.get("choices", []), set(ids), k)
                if chosen:
                    return Choice(chosen, str(data.get("reason", "")).strip())
                last_err = DecisionError(f"model returned no valid ids: {content[:200]}")
            except httpx.HTTPStatusError as e:
                raise DecisionError(
                    f"Ollama returned {e.response.status_code}: {e.response.text[:300]}"
                ) from e
            except httpx.HTTPError as e:
                raise DecisionError(f"could not reach Ollama: {e}") from e
            except (KeyError, ValueError, TypeError) as e:
                last_err = e
        raise DecisionError(f"bad model output after retries: {last_err}")


# ----------------------------------------------------------------------- keyword


_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by can do for from how i in is it my of on or should "
    "the to what when where which who why with you your me we our this that".split()
)


def _norm(word: str) -> str:
    # Crude plural folding so "vets" matches "vet"; good enough for a baseline.
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def _tokens(text: str) -> set[str]:
    return {_norm(w) for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1}


class KeywordDecider:
    """Offline baseline: word overlap between context and option text.

    Useful as (1) a fallback when the LLM fails, (2) a no-Ollama demo mode, and
    (3) the baseline your model should beat in evals.
    """

    name = "keyword"

    def choose(self, *, task: str, context: str, options: list[Option], k: int) -> Choice:
        if not options:
            raise DecisionError("no options to choose from")
        ctx = _tokens(context)
        # Stable sort keeps original order on ties, so results are deterministic.
        scored = sorted(options, key=lambda o: -len(ctx & _tokens(f"{o.id} {o.label}")))
        chosen = [o.id for o in scored[: min(k, len(options))]]
        return Choice(chosen, "keyword overlap with your query and history")
