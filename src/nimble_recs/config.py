from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    database_url: str = "postgresql://recs:recs@localhost:5432/recs"
    ollama_host: str = "http://localhost:11434"
    model: str = "nimble"
    history_window: int = 15
    top_domains: int = 1
    num_suggestions: int = 3
    max_prompt_tokens: int = 6000
    ollama_timeout_s: float = 120.0

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        d = cls()
        return cls(
            database_url=os.getenv("DATABASE_URL", d.database_url),
            ollama_host=os.getenv("OLLAMA_HOST", d.ollama_host).rstrip("/"),
            model=os.getenv("MODEL", d.model),
            history_window=_int("HISTORY_WINDOW", d.history_window),
            top_domains=_int("TOP_DOMAINS", d.top_domains),
            num_suggestions=_int("NUM_SUGGESTIONS", d.num_suggestions),
            max_prompt_tokens=_int("MAX_PROMPT_TOKENS", d.max_prompt_tokens),
            ollama_timeout_s=float(os.getenv("OLLAMA_TIMEOUT_S", d.ollama_timeout_s)),
        )
