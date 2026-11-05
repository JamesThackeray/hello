import json
from pathlib import Path

import httpx
import pytest

from nimble_recs.config import Settings
from nimble_recs.decider import (
    DecisionError,
    KeywordDecider,
    OllamaDecider,
    SystemOneDecider,
    ollama_decider,
)
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
    assert s.notes == ["broken failed (boom); used keyword"]  # once, not once per stage
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
    assert seen["think"] is False
    assert seen["options"]["num_ctx"] == 8192
    assert seen["keep_alive"] == "30m"


def test_ollama_drops_think_if_server_rejects_it():
    bodies = []

    def handler(req):
        body = json.loads(req.content)
        bodies.append(body)
        if "think" in body:
            return httpx.Response(400, json={"error": '"x" does not support thinking'})
        content = json.dumps({"choices": ["a"], "reason": "r"})
        return httpx.Response(200, json={"message": {"content": content}})

    d = _ollama(handler)
    assert d.choose(task="t", context="c", options=OPTS, k=1).ids == ["a"]
    assert d.choose(task="t", context="c", options=OPTS, k=1).ids == ["a"]
    assert ["think" in b for b in bodies] == [True, False, False]  # stops asking after one 400


def test_prepare_loads_installed_model():
    seen = []

    def handler(req):
        seen.append((req.method, req.url.path))
        if req.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "nimble:latest"}]})
        return httpx.Response(200, json={"done": True})

    _ollama(handler).prepare()  # "nimble" matches "nimble:latest"
    assert seen == [("GET", "/api/tags"), ("POST", "/api/generate")]


def test_prepare_explains_missing_model():
    def handler(req):
        return httpx.Response(200, json={"models": [{"name": "gemma4:12b"}]})

    with pytest.raises(DecisionError, match=r"not installed.*gemma4:12b.*ollama pull nimble"):
        _ollama(handler).prepare()


# ---------------------------------------------------------- /v1/systemone client


def _systemone(handler):
    client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
    return SystemOneDecider("http://test", "nimble", client=client)


def _answer_by_score(score):
    """Fake decision model: probability of each option comes from score(option_id)."""
    requests = []

    def handler(req):
        body = json.loads(req.content)
        requests.append(body)
        answers = {}
        for name, q in body["questions"].items():
            raw = {oid: score(oid) for oid in q["criteria"]}
            total = sum(raw.values())
            probs = {oid: v / total for oid, v in raw.items()}
            answers[name] = {"type": "choice", "choice": max(probs, key=probs.get),
                             "probabilities": probs, "confidence": 0.5}
        return httpx.Response(200, json={"model": "nimble", "answers": answers})

    return handler, requests


def test_systemone_request_shape_and_ranking():
    handler, requests = _answer_by_score({"a": 1, "b": 6, "c": 3}.get)
    choice = _systemone(handler).choose(task="Which fruit?", context="ctx", options=OPTS, k=2)
    assert choice.ids == ["b", "c"]
    assert choice.reason == "model probability: 60%, 30%"
    (body,) = requests
    assert body["model"] == "nimble"
    assert body["state"] == "ctx"
    assert body["questions"] == {"pick0": {"type": "choice", "instructions": "Which fruit?",
                                           "criteria": {"a": "apples", "b": "bananas",
                                                        "c": "cherries"}}}


def test_systemone_narrows_long_lists_in_rounds():
    opts = [Option(f"q{i:02d}", f"question {i}") for i in range(50)]
    handler, requests = _answer_by_score(lambda oid: int(oid[1:]) + 1)  # q49 best
    choice = _systemone(handler).choose(task="t", context="c", options=opts, k=3)
    assert choice.ids == ["q49", "q48", "q47"]
    first, final = requests
    assert [len(q["criteria"]) for q in first["questions"].values()] == [25, 25]
    assert sorted(final["questions"]["pick0"]["criteria"]) == [
        "q22", "q23", "q24", "q47", "q48", "q49"]


FIFTY = [Option(f"q{i:02d}", f"question {i}") for i in range(50)]


def _by_number(oid):
    return int(oid[1:]) + 1  # q49 scores best


def test_systemone_packs_chunks_into_requests_that_fit_the_budget():
    handler, requests = _answer_by_score(_by_number)
    d = _systemone(handler)
    d.max_prompt_tokens = 500  # room for one 25-option chunk per request, not two
    assert d.choose(task="t", context="c", options=FIFTY, k=3).ids == ["q49", "q48", "q47"]
    assert [len(r["questions"]) for r in requests] == [1, 1, 1]  # 2 chunks, then the final


def test_systemone_splits_a_request_that_overflows_the_context():
    inner, requests = _answer_by_score(_by_number)

    def handler(req):
        if len(json.loads(req.content)["questions"]) > 1:
            requests.append("rejected")
            return httpx.Response(400, json={
                "error": "request (8901 tokens) exceeds the available context size (8448 tokens)"})
        return inner(req)

    assert _systemone(handler).choose(task="t", context="c", options=FIFTY, k=3).ids == [
        "q49", "q48", "q47"]
    assert [r if r == "rejected" else len(r["questions"]) for r in requests] == [
        "rejected", 1, 1, 1]


def test_systemone_gives_up_when_one_question_cannot_fit():
    def handler(req):
        return httpx.Response(500, json={"error": "Context size has been exceeded."})

    with pytest.raises(DecisionError, match="Context size"):
        _systemone(handler).choose(task="t", context="c", options=OPTS, k=1)


class SplittingRecorder(RecordingDecider):
    options_per_prompt = 26


def test_history_kept_for_big_lists_when_decider_splits_prompts():
    from nimble_recs.models import Domain, Question

    qs = [Question(f"big-{i:03d}", "big",
                   f"What is the best way to handle situation number {i} when it comes up at home?")
          for i in range(400)]
    for decider, history_expected in ((RecordingDecider(), False), (SplittingRecorder(), True)):
        store = MemoryStore([Domain("big", "Big", "a big domain")], qs)
        store.add_history("u", "an earlier query")
        Recommender(store, decider, settings()).suggest("u", "now")
        _, ctx, opts, _ = decider.calls[1]  # stage 2
        assert len(opts) == 400
        assert ("an earlier query" in ctx) is history_expected


@pytest.mark.parametrize("show, expected", [
    (httpx.Response(200, json={"capabilities": ["decision"]}), SystemOneDecider),
    (httpx.Response(200, json={"capabilities": ["completion", "thinking"]}), OllamaDecider),
    (httpx.Response(404, json={"error": "model not found"}), OllamaDecider),
])
def test_ollama_decider_picks_endpoint_from_capabilities(monkeypatch, show, expected):
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: show)
    assert type(ollama_decider("http://test", "m")) is expected


def test_systemone_errors_become_decision_errors():
    def not_found(req):
        return httpx.Response(404, json={"error": "model 'nimble' not found"})

    with pytest.raises(DecisionError, match="404"):
        _systemone(not_found).choose(task="t", context="c", options=OPTS, k=1)

    def junk(req):
        return httpx.Response(200, json={"answers": {"pick0": {"choice": "a"}}})

    with pytest.raises(DecisionError, match="no probabilities"):
        _systemone(junk).choose(task="t", context="c", options=OPTS, k=1)


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
