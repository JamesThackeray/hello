import json
from pathlib import Path

import httpx
import pytest

from nimble_recs.config import Settings
from nimble_recs.decider import DecisionError, KeywordDecider, OllamaDecider
from nimble_recs.models import Choice, Option
from nimble_recs.recommender import Recommender
from nimble_recs.store import MemoryStore, load_csv

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture
def store():
    return MemoryStore(*load_csv(DATA))


def settings(**kw):
    return Settings(**kw)


# ----------------------------------------------------------------------- data


def test_seed_data_is_consistent():
    domains, questions = load_csv(DATA)
    assert len(domains) == 20
    assert len({q.id for q in questions}) == len(questions)
    assert all(q.id.startswith(q.domain_id + "-") for q in questions)


# ------------------------------------------------------------------ pipeline


def test_two_stage_keyword_pipeline(store):
    rec = Recommender(store, KeywordDecider(), settings())
    s = rec.suggest("alice", "how do I open a bank account and build credit")
    assert [d.id for d in s.domains] == ["personal_finance"]
    assert len(s.questions) == 3
    assert all(q.domain_id == "personal_finance" for q in s.questions)
    assert not s.fallback_used
    assert len(store.suggestion_log) == 3


def test_history_is_per_user_and_clearable(store):
    rec = Recommender(store, KeywordDecider(), settings())
    rec.suggest("alice", "visa")
    rec.suggest("alice", "eVisa right to work")
    rec.suggest("bob", "dog")
    assert [h.query for h in store.recent_history("alice", 10)] == ["eVisa right to work", "visa"]
    assert store.clear_history("alice") == 2
    assert store.recent_history("alice", 10) == []
    assert len(store.recent_history("bob", 10)) == 1


class RecordingDecider:
    name = "recording"

    def __init__(self):
        self.calls = []

    def choose(self, *, task, context, options, k):
        self.calls.append((task, context, options, k))
        return Choice([o.id for o in options[:k]], "first k")


def test_history_reaches_the_model_and_stage2_only_sees_chosen_domain(store):
    d = RecordingDecider()
    rec = Recommender(store, d, settings())
    rec.suggest("u", "first query")
    rec.suggest("u", "second query")

    task1, ctx1, opts1, _ = d.calls[2]  # stage 1 of second suggest
    assert len(opts1) == 20
    assert "Current query: second query" in ctx1
    assert "1. first query" in ctx1

    _, _, opts2, k2 = d.calls[3]
    first_domain = opts1[0].id
    assert {o.id.rsplit("-", 1)[0] for o in opts2} == {first_domain}
    assert k2 == 3


def test_top_two_domains_widens_stage2(store):
    d = RecordingDecider()
    Recommender(store, d, settings(top_domains=2)).suggest("u", "x")
    _, _, opts2, _ = d.calls[1]
    assert len({o.id.rsplit("-", 1)[0] for o in opts2}) == 2


def test_history_trimmed_to_fit_budget(store):
    d = RecordingDecider()
    rec = Recommender(store, d, settings(max_prompt_tokens=700, history_window=50))
    for i in range(40):
        store.add_history("u", f"an older query with plenty of words number {i}")
    rec.suggest("u", "now")
    _, ctx, _, _ = d.calls[0]
    assert "Current query: now" in ctx
    assert "number 39" in ctx          # newest kept
    assert "number 0" not in ctx        # oldest dropped


class FailingDecider:
    name = "broken"

    def choose(self, **_):
        raise DecisionError("boom")


def test_falls_back_to_keyword_when_model_fails(store):
    s = Recommender(store, FailingDecider(), settings()).suggest("u", "puppy vet")
    assert s.fallback_used
    assert s.domains[0].id == "pets"
    assert any("broken failed" in n for n in s.notes)
    assert all(row[4] for row in store.suggestion_log)


# ------------------------------------------------------------- ollama client


def _ollama(handler):
    client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
    return OllamaDecider("http://test", "nimble", client=client)


OPTS = [Option("a", "apples"), Option("b", "bananas"), Option("c", "cherries")]


def test_ollama_request_shape_and_parsing():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        content = json.dumps({"choices": ["b", "b", "zzz", "a"], "reason": "fruit"})
        return httpx.Response(200, json={"message": {"content": content}})

    choice = _ollama(handler).choose(task="t", context="c", options=OPTS, k=2)
    assert choice.ids == ["b", "a"]          # deduped, invalid id dropped
    assert seen["model"] == "nimble"
    assert seen["stream"] is False
    assert seen["format"]["properties"]["choices"]["items"]["enum"] == ["a", "b", "c"]
    assert seen["format"]["properties"]["choices"]["maxItems"] == 2


def test_ollama_retries_on_bad_json_then_raises():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(200, json={"message": {"content": "not json"}})

    with pytest.raises(DecisionError):
        _ollama(handler).choose(task="t", context="c", options=OPTS, k=1)
    assert len(calls) == 2


def test_ollama_http_error_is_decision_error():
    def handler(req):
        return httpx.Response(404, json={"error": "model 'nimble' not found"})

    with pytest.raises(DecisionError, match="404"):
        _ollama(handler).choose(task="t", context="c", options=OPTS, k=1)
