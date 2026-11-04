from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class Domain:
    id: str
    name: str
    description: str


@dataclass(frozen=True)
class Question:
    id: str
    domain_id: str
    text: str


@dataclass(frozen=True)
class HistoryItem:
    id: int
    query: str
    created_at: datetime


@dataclass(frozen=True)
class Option:
    """One thing the decider can choose: a domain or a question."""

    id: str
    label: str


@dataclass(frozen=True)
class Choice:
    ids: list[str]
    reason: str = ""


@dataclass
class Suggestion:
    query: str
    domains: list[Domain]
    questions: list[Question]
    domain_reason: str = ""
    question_reason: str = ""
    fallback_used: bool = False
    notes: list[str] = field(default_factory=list)
