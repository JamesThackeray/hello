"""The two-stage pipeline: history -> pick domain(s) -> pick questions in those domains."""

from __future__ import annotations

from .config import Settings
from .decider import Decider, DecisionError, KeywordDecider, estimate_tokens, render_prompt
from .models import Choice, HistoryItem, Option, Suggestion
from .store import Store

DOMAIN_TASK = (
    "Stage 1 of 2: pick the domain(s) this user most likely wants suggestions from right now."
)
QUESTION_TASK = (
    "Stage 2 of 2: pick the questions this user would most likely want to ask or explore next."
)


def build_context(query: str, history: list[HistoryItem]) -> str:
    """history is newest first and excludes the current query."""
    lines = [f"Current query: {query}"]
    if history:
        lines.append("Recent history (newest first):")
        lines += [f"{i}. {h.query}" for i, h in enumerate(history, 1)]
    else:
        lines.append("Recent history: (none)")
    return "\n".join(lines)


def fit_context(query: str, history: list[HistoryItem], task: str,
                options: list[Option], k: int, max_tokens: int) -> str:
    """Drop the oldest history until the whole prompt fits the token budget."""
    hist = list(history)
    while True:
        ctx = build_context(query, hist)
        if estimate_tokens(render_prompt(task, ctx, options, k)) <= max_tokens or not hist:
            return ctx
        hist.pop()


class Recommender:
    def __init__(self, store: Store, decider: Decider, settings: Settings,
                 fallback: Decider | None = None):
        self.store = store
        self.decider = decider
        self.settings = settings
        self.fallback = fallback if fallback is not None else KeywordDecider()

    def _decide(self, task: str, context: str, options: list[Option], k: int,
                notes: list[str]) -> tuple[Choice, bool]:
        try:
            return self.decider.choose(task=task, context=context, options=options, k=k), False
        except DecisionError as e:
            if self.fallback is None or self.fallback is self.decider:
                raise
            notes.append(f"{self.decider.name} failed ({e}); used {self.fallback.name}")
            return self.fallback.choose(task=task, context=context, options=options, k=k), True

    def suggest(self, user_id: str, query: str) -> Suggestion:
        s = self.settings
        previous = self.store.recent_history(user_id, s.history_window)
        history_id = self.store.add_history(user_id, query)
        notes: list[str] = []

        # ---- Stage 1: domain
        domains = self.store.list_domains()
        if not domains:
            raise RuntimeError("no domains loaded - run `nimble-recs seed` first")
        by_id = {d.id: d for d in domains}
        d_opts = [Option(d.id, f"{d.name} - {d.description}") for d in domains]
        d_k = min(s.top_domains, len(domains))
        ctx = fit_context(query, previous, DOMAIN_TASK, d_opts, d_k, s.max_prompt_tokens)
        d_choice, fb1 = self._decide(DOMAIN_TASK, ctx, d_opts, d_k, notes)
        chosen_domains = [by_id[i] for i in d_choice.ids]

        # ---- Stage 2: questions inside the chosen domain(s)
        questions = self.store.questions_for([d.id for d in chosen_domains])
        if not questions:
            notes.append("chosen domain has no questions")
            return Suggestion(query, chosen_domains, [], d_choice.reason, "", fb1, notes)
        q_by_id = {q.id: q for q in questions}
        q_opts = [Option(q.id, q.text) for q in questions]
        q_k = min(s.num_suggestions, len(questions))
        ctx = fit_context(query, previous, QUESTION_TASK, q_opts, q_k, s.max_prompt_tokens)
        q_choice, fb2 = self._decide(QUESTION_TASK, ctx, q_opts, q_k, notes)
        chosen_qs = [q_by_id[i] for i in q_choice.ids]

        self.store.log_suggestions(history_id, chosen_qs, fallback=fb1 or fb2)
        return Suggestion(query, chosen_domains, chosen_qs, d_choice.reason,
                          q_choice.reason, fb1 or fb2, notes)
