"""Storage: Postgres for real use, in-memory for tests and quick demos."""

from __future__ import annotations

import csv
import itertools
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Protocol

from .models import Domain, HistoryItem, Question


class Store(Protocol):
    def list_domains(self) -> list[Domain]: ...
    def questions_for(self, domain_ids: list[str]) -> list[Question]: ...
    def add_history(self, user_id: str, query: str) -> int: ...
    def recent_history(self, user_id: str, limit: int) -> list[HistoryItem]:
        """Newest first."""
        ...
    def clear_history(self, user_id: str) -> int: ...
    def log_suggestions(
        self, history_id: int, questions: list[Question], fallback: bool
    ) -> None: ...


# --------------------------------------------------------------------------- CSV


def load_csv(data_dir: Path) -> tuple[list[Domain], list[Question]]:
    with open(data_dir / "domains.csv", newline="", encoding="utf-8") as f:
        domains = [Domain(r["id"], r["name"], r["description"]) for r in csv.DictReader(f)]
    with open(data_dir / "questions.csv", newline="", encoding="utf-8") as f:
        questions = [Question(r["id"], r["domain_id"], r["text"]) for r in csv.DictReader(f)]

    known = {d.id for d in domains}
    orphans = sorted({q.domain_id for q in questions} - known)
    if orphans:
        raise ValueError(f"questions.csv references unknown domain ids: {orphans}")
    return domains, questions


# ------------------------------------------------------------------------ memory


class MemoryStore:
    def __init__(self, domains: list[Domain], questions: list[Question]):
        self._domains = list(domains)
        self._questions = list(questions)
        self._history: dict[str, list[HistoryItem]] = {}
        self._ids = itertools.count(1)
        self.suggestion_log: list[tuple[int, str, str, int, bool]] = []

    def list_domains(self) -> list[Domain]:
        return list(self._domains)

    def questions_for(self, domain_ids: list[str]) -> list[Question]:
        wanted = set(domain_ids)
        return [q for q in self._questions if q.domain_id in wanted]

    def add_history(self, user_id: str, query: str) -> int:
        item = HistoryItem(next(self._ids), query, datetime.now(timezone.utc))
        self._history.setdefault(user_id, []).append(item)
        return item.id

    def recent_history(self, user_id: str, limit: int) -> list[HistoryItem]:
        return list(reversed(self._history.get(user_id, [])))[:limit]

    def clear_history(self, user_id: str) -> int:
        return len(self._history.pop(user_id, []))

    def log_suggestions(self, history_id: int, questions: list[Question], fallback: bool) -> None:
        for rank, q in enumerate(questions, 1):
            self.suggestion_log.append((history_id, q.domain_id, q.id, rank, fallback))


# ---------------------------------------------------------------------- postgres


class PostgresStore:
    def __init__(self, database_url: str):
        import psycopg  # imported lazily so tests / --memory don't need a DB driver

        # Fail fast with a clear message instead of hanging if the database isn't up.
        self._conn = psycopg.connect(database_url, autocommit=True, connect_timeout=10)

    def close(self) -> None:
        self._conn.close()

    def init_schema(self) -> None:
        sql = resources.files("nimble_recs").joinpath("schema.sql").read_text()
        self._conn.execute(sql)

    def seed(self, domains: list[Domain], questions: list[Question]) -> None:
        """Upsert domains/questions; removes questions/domains no longer in the CSVs."""
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO domains (id, name, description) VALUES (%s, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name,
                                                  description = EXCLUDED.description""",
                [(d.id, d.name, d.description) for d in domains],
            )
            cur.executemany(
                """INSERT INTO questions (id, domain_id, text) VALUES (%s, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET domain_id = EXCLUDED.domain_id,
                                                  text = EXCLUDED.text""",
                [(q.id, q.domain_id, q.text) for q in questions],
            )
            cur.execute("DELETE FROM questions WHERE NOT (id = ANY(%s))", ([q.id for q in questions],))
            cur.execute("DELETE FROM domains WHERE NOT (id = ANY(%s))", ([d.id for d in domains],))

    def list_domains(self) -> list[Domain]:
        rows = self._conn.execute("SELECT id, name, description FROM domains ORDER BY id").fetchall()
        return [Domain(*r) for r in rows]

    def questions_for(self, domain_ids: list[str]) -> list[Question]:
        rows = self._conn.execute(
            "SELECT id, domain_id, text FROM questions WHERE domain_id = ANY(%s) ORDER BY id",
            (domain_ids,),
        ).fetchall()
        return [Question(*r) for r in rows]

    def add_history(self, user_id: str, query: str) -> int:
        row = self._conn.execute(
            "INSERT INTO history (user_id, query) VALUES (%s, %s) RETURNING id", (user_id, query)
        ).fetchone()
        return row[0]

    def recent_history(self, user_id: str, limit: int) -> list[HistoryItem]:
        rows = self._conn.execute(
            """SELECT id, query, created_at FROM history WHERE user_id = %s
               ORDER BY created_at DESC, id DESC LIMIT %s""",
            (user_id, limit),
        ).fetchall()
        return [HistoryItem(*r) for r in rows]

    def clear_history(self, user_id: str) -> int:
        cur = self._conn.execute("DELETE FROM history WHERE user_id = %s", (user_id,))
        return cur.rowcount

    def log_suggestions(self, history_id: int, questions: list[Question], fallback: bool) -> None:
        with self._conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO suggestions (history_id, domain_id, question_id, rank, fallback)
                   VALUES (%s, %s, %s, %s, %s)""",
                [(history_id, q.domain_id, q.id, i, fallback) for i, q in enumerate(questions, 1)],
            )
