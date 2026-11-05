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


def _with_tag(model: str) -> str:
    """Ollama lists `nimble` as `nimble:latest`; normalise so the two compare equal."""
    return model if ":" in model.rsplit("/", 1)[-1] else f"{model}:latest"


def require_model(client: httpx.Client, model: str) -> None:
    """Raise DecisionError with an actionable message if Ollama is down or `model` isn't pulled."""
    try:
        resp = client.get("/api/tags")
        resp.raise_for_status()
        installed = [m.get("name", "") for m in resp.json().get("models", [])]
    except (httpx.HTTPError, ValueError, AttributeError) as e:
        raise DecisionError(f"could not reach Ollama: {e}") from e
    if _with_tag(model) not in {_with_tag(n) for n in installed}:
        raise DecisionError(
            f"model '{model}' is not installed in Ollama "
            f"(installed: {', '.join(installed) or 'none'}). "
            f"Run `ollama pull {model}`, or choose another with --model or MODEL in .env"
        )


class OllamaDecider:
    """Calls Ollama's /api/chat with a JSON schema so the model must return valid ids.

    The schema's `enum` constrains generation to real option ids; we still validate
    the result because not every model/runtime honours every schema keyword.

    Thinking is switched off (`think: false`): a decision only needs the JSON answer,
    and with thinking on, models such as gemma4 are slower and their constrained
    answer can contradict their own reasoning.
    """

    def __init__(self, host: str, model: str, timeout_s: float = 120.0,
                 client: httpx.Client | None = None, retries: int = 1,
                 num_ctx: int = 8192, keep_alive: str = "30m"):
        self.name = f"ollama:{model}"
        self.model = model
        self.retries = retries
        self.num_ctx = num_ctx          # Ollama defaults to 4096 and silently truncates past it
        self.keep_alive = keep_alive    # keep the model in memory between chat messages
        self._think_off = True          # cleared if the server rejects the `think` field
        self._client = client or httpx.Client(base_url=host, timeout=timeout_s)

    def prepare(self, load_timeout_s: float = 600.0) -> None:
        """Check the model is installed, then load it so the first query doesn't time out.

        Raises DecisionError with an actionable message if Ollama is down or the model is missing.
        """
        require_model(self._client, self.model)
        # An empty generate request just loads the model (with our context size) into memory.
        load = {"model": self.model, "keep_alive": self.keep_alive,
                "options": {"num_ctx": self.num_ctx}}
        try:
            self._client.post("/api/generate", json=load, timeout=load_timeout_s).raise_for_status()
        except httpx.HTTPStatusError as e:
            raise DecisionError(
                f"Ollama could not load {self.model}: {e.response.text[:300]}"
            ) from e
        except httpx.HTTPError as e:
            raise DecisionError(f"Ollama could not load {self.model}: {e}") from e

    def _post_chat(self, body: dict) -> httpx.Response:
        if self._think_off:
            resp = self._client.post("/api/chat", json={**body, "think": False})
            if resp.status_code != 400 or "think" not in resp.text.lower():
                return resp
            self._think_off = False  # this model/runtime can't toggle thinking; stop sending it
        return self._client.post("/api/chat", json=body)

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
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0, "num_ctx": self.num_ctx},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": render_prompt(task, context, options, k)},
            ],
        }

        last_err: Exception | None = None
        for _ in range(self.retries + 1):
            try:
                resp = self._post_chat(body)
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


# --------------------------------------------------------------------- systemone


MAX_CHOICES = 26    # /v1/systemone allows 2-26 options per `choice` question
MAX_QUESTIONS = 64  # ...and up to 64 questions per request

# Measured on nimble: every question's prompt holds the *whole* request (the state plus
# all questions' options), so it's the request that must fit the 8K context, not each
# question. Option lists come to about 2 characters per token, plus ~150 tokens of template.
CHARS_PER_TOKEN = 2
TEMPLATE_TOKENS = 150


class _ContextOverflow(DecisionError):
    """Ollama said the request doesn't fit the model's context window."""


def _split(items: list[str], size: int) -> list[list[str]]:
    """Split into as few near-equal chunks of at most `size` as possible (so none has 1 item)."""
    n = -(-len(items) // size)
    base, extra = divmod(len(items), n)
    out, start = [], 0
    for i in range(n):
        end = start + base + (1 if i < extra else 0)
        out.append(items[start:end])
        start = end
    return out


class SystemOneDecider:
    """Ollama decision models (such as nimble) through /v1/systemone.

    A decision model doesn't write text. For each question it scores the options it is
    given and returns a probability for each one. We ask a single `choice` question and
    rank the options by probability to get the top k.

    One question can hold at most 26 options, so longer lists (e.g. 50 questions in a
    domain) are split into chunks, one question per chunk. The top k of each chunk go
    through to another round, until one question can rank what is left. The chunks are
    packed into as few requests as fit `max_prompt_tokens`. Fewer, bigger requests are
    faster, but a request that overflows the context fails outright.
    """

    # The history only has to fit next to one question's worth of options (see Recommender).
    options_per_prompt = MAX_CHOICES

    def __init__(self, host: str, model: str, timeout_s: float = 120.0,
                 client: httpx.Client | None = None, keep_alive: str = "30m",
                 max_prompt_tokens: int = 6000):
        self.name = f"ollama:{model}"
        self.model = model
        self.keep_alive = keep_alive
        self.max_prompt_tokens = max_prompt_tokens
        self._client = client or httpx.Client(base_url=host, timeout=timeout_s)

    def prepare(self, load_timeout_s: float = 600.0) -> None:
        """Check the model is installed, then load it so the first query doesn't time out."""
        require_model(self._client, self.model)
        warm_up = {"q": {"type": "choice", "instructions": "Pick one.",
                         "criteria": {"yes": None, "no": None}}}
        self._ask("warm-up", warm_up, timeout=load_timeout_s)

    def _ask(self, state: str, questions: dict, timeout: float | None = None) -> dict:
        body = {"model": self.model, "state": state, "questions": questions,
                "keep_alive": self.keep_alive}
        extra = {"timeout": timeout} if timeout else {}
        try:
            resp = self._client.post("/v1/systemone", json=body, **extra)
            resp.raise_for_status()
            answers = resp.json()["answers"]
            if not isinstance(answers, dict):
                raise TypeError("answers is not an object")
            return answers
        except httpx.HTTPStatusError as e:
            msg = f"Ollama returned {e.response.status_code}: {e.response.text[:300]}"
            # 400 "request (N tokens) exceeds the available context size", or a
            # 500 "Context size has been exceeded" from the runner itself.
            if "context size" in e.response.text.lower():
                raise _ContextOverflow(msg) from e
            raise DecisionError(msg) from e
        except httpx.HTTPError as e:
            raise DecisionError(f"could not reach Ollama: {e}") from e
        except (KeyError, ValueError, TypeError) as e:
            raise DecisionError(f"unexpected /v1/systemone response: {e}") from e

    def _batches(self, state: str, questions: dict) -> list[dict]:
        """Pack questions, in order, into as few requests as fit the token budget."""
        fixed = TEMPLATE_TOKENS + len(state) // CHARS_PER_TOKEN
        batches: list[dict] = []
        batch: dict = {}
        used = fixed
        for name, q in questions.items():
            cost = len(json.dumps(q)) // CHARS_PER_TOKEN
            if batch and (used + cost > self.max_prompt_tokens or len(batch) == MAX_QUESTIONS):
                batches.append(batch)
                batch, used = {}, fixed
            batch[name] = q
            used += cost
        if batch:
            batches.append(batch)
        return batches

    def _ask_fitting(self, state: str, batch: dict) -> dict:
        """Ask a batch; if Ollama says it doesn't fit after all, split it in half and retry."""
        try:
            return self._ask(state, batch)
        except _ContextOverflow:
            if len(batch) == 1:
                raise
            items = list(batch.items())
            half = len(items) // 2
            return {**self._ask_fitting(state, dict(items[:half])),
                    **self._ask_fitting(state, dict(items[half:]))}

    def _ask_all(self, state: str, questions: dict) -> dict:
        answers: dict = {}
        for batch in self._batches(state, questions):
            answers.update(self._ask_fitting(state, batch))
        return answers

    @staticmethod
    def _rank(answer: object, chunk: list[str]) -> list[tuple[str, float]]:
        """Options in this chunk, most probable first (ties keep the original order)."""
        probs = answer.get("probabilities") if isinstance(answer, dict) else None
        if not isinstance(probs, dict):
            raise DecisionError(f"no probabilities in /v1/systemone answer: {str(answer)[:200]}")
        scored = [(oid, float(probs.get(oid, 0.0))) for oid in chunk]
        return sorted(scored, key=lambda pair: -pair[1])

    def choose(self, *, task: str, context: str, options: list[Option], k: int) -> Choice:
        if not options:
            raise DecisionError("no options to choose from")
        k = min(k, len(options))
        if len(options) == 1:
            return Choice([options[0].id], "only one option")
        labels = {o.id: o.label for o in options}
        pool = list(dict.fromkeys(o.id for o in options))
        while True:
            chunks = _split(pool, MAX_CHOICES)
            questions = {
                f"pick{i}": {"type": "choice", "instructions": task,
                             "criteria": {oid: labels[oid] for oid in chunk}}
                for i, chunk in enumerate(chunks)
            }
            answers = self._ask_all(context, questions)
            ranked = [self._rank(answers.get(f"pick{i}"), chunk) for i, chunk in enumerate(chunks)]
            if len(chunks) == 1:
                top = ranked[0][:k]
                reason = "model probability: " + ", ".join(
                    "<1%" if p < 0.005 else f"{p:.0%}" for _, p in top)
                return Choice([oid for oid, _ in top], reason)
            survivors = [oid for chunk_ranked in ranked for oid, _ in chunk_ranked[:k]]
            if len(survivors) >= len(pool):  # only when k is close to the chunk size
                raise DecisionError(f"can't narrow {len(pool)} options down to {k}")
            pool = survivors


def ollama_decider(host: str, model: str, timeout_s: float = 120.0, num_ctx: int = 8192,
                   max_prompt_tokens: int = 6000) -> OllamaDecider | SystemOneDecider:
    """Pick the right client for `model`: decision models such as nimble only work through
    /v1/systemone (they reject /api/chat), and Ollama marks them with the `decision` capability."""
    try:
        resp = httpx.post(f"{host}/api/show", json={"model": model}, timeout=10)
        caps = (resp.json().get("capabilities") or []) if resp.is_success else []
    except (httpx.HTTPError, ValueError, AttributeError):
        caps = []  # unreachable or not pulled: prepare() reports it with a clear message
    if "decision" in caps:
        return SystemOneDecider(host, model, timeout_s, max_prompt_tokens=max_prompt_tokens)
    return OllamaDecider(host, model, timeout_s, num_ctx=num_ctx)


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
