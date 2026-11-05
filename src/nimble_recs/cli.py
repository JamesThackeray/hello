"""Command line: `nimble-recs seed` and `nimble-recs chat`."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from .config import Settings
from .decider import DecisionError, KeywordDecider, ollama_decider
from .models import Suggestion
from .recommender import Recommender
from .store import MemoryStore, PostgresStore, load_csv

HELP = """\
Type anything to search - each message is saved to your history and used for suggestions.
Commands:
  /clear  (or "clear history")  delete your history
  /history                      show your recent history
  /domains                      list the domains
  /user <name>                  switch user (each user has separate history)
  /help                         show this help
  /quit                         exit
"""


def _print_suggestion(s: Suggestion) -> None:
    for note in s.notes:
        print(f"  ! {note}")
    print(f"  domain: {', '.join(d.name for d in s.domains)}")
    if s.domain_reason:
        print(f"    why: {s.domain_reason}")
    if not s.questions:
        print("  (no questions in that domain)")
        return
    for i, q in enumerate(s.questions, 1):
        print(f"  {i}. {q.text}")
    if s.question_reason:
        print(f"    why: {s.question_reason}")


def _make_store(args: argparse.Namespace, settings: Settings):
    if args.memory:
        return MemoryStore(*load_csv(Path(args.data_dir)))
    try:
        return PostgresStore(settings.database_url)
    except Exception as e:  # psycopg.OperationalError etc.
        sys.exit(
            f"Could not connect to Postgres at {settings.database_url}\n  {e}\n"
            "Start it with `docker compose up -d`, or run with --memory to skip the database."
        )


def cmd_seed(args: argparse.Namespace, settings: Settings) -> None:
    domains, questions = load_csv(Path(args.data_dir))
    store = _make_store(argparse.Namespace(memory=False, data_dir=args.data_dir), settings)
    store.init_schema()
    store.seed(domains, questions)
    print(f"Seeded {len(domains)} domains and {len(questions)} questions.")


def cmd_chat(args: argparse.Namespace, settings: Settings) -> None:
    if args.model:
        settings = replace(settings, model=args.model)
    store = _make_store(args, settings)
    if isinstance(store, PostgresStore):
        store.init_schema()

    if args.decider == "keyword":
        decider = KeywordDecider()
    else:
        decider = ollama_decider(settings.ollama_host, settings.model, settings.ollama_timeout_s,
                                 num_ctx=settings.ollama_num_ctx,
                                 max_prompt_tokens=settings.max_prompt_tokens)
        print(f"Loading {settings.model} in Ollama (the first load can take a minute)...",
              flush=True)
        try:
            decider.prepare()
        except DecisionError as e:
            print(f"  ! {e}\n  ! using the offline keyword baseline for this session")
            decider = KeywordDecider()
    rec = Recommender(store, decider, settings)

    user = args.user
    print(f"nimble-recs chat | decider={decider.name} | store="
          f"{'memory' if args.memory else 'postgres'} | user={user}")
    print("Type /help for commands.\n")

    while True:
        try:
            line = input(f"{user}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        cmd = line.lower()

        if cmd in ("/quit", "/exit", "quit", "exit"):
            break
        if cmd in ("/help", "help"):
            print(HELP)
        elif cmd in ("/clear", "clear history", "/clear history"):
            n = store.clear_history(user)
            print(f"  cleared {n} history item(s) for {user}")
        elif cmd == "/history":
            items = store.recent_history(user, settings.history_window)
            if not items:
                print("  (no history)")
            for h in items:
                print(f"  {h.created_at.astimezone():%Y-%m-%d %H:%M}  {h.query}")
        elif cmd == "/domains":
            for d in store.list_domains():
                print(f"  {d.id:<18} {d.name} - {d.description}")
        elif cmd.startswith("/user"):
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                user = parts[1].strip()
                print(f"  now chatting as {user}")
            else:
                print("  usage: /user <name>")
        elif cmd.startswith("/"):
            print("  unknown command - try /help")
        else:
            try:
                _print_suggestion(rec.suggest(user, line))
            except Exception as e:
                print(f"  error: {e}")
        print()


def main(argv: list[str] | None = None) -> None:
    settings = Settings.from_env()
    p = argparse.ArgumentParser(prog="nimble-recs", description=__doc__)
    p.add_argument("--data-dir", default="data", help="folder with domains.csv and questions.csv")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("seed", help="create tables and load the CSVs into Postgres")

    chat = sub.add_parser("chat", help="interactive chat that builds history and suggests questions")
    chat.add_argument("--user", default="default", help="user id (history is per user)")
    chat.add_argument("--memory", action="store_true", help="no Postgres: load CSVs into memory")
    chat.add_argument("--decider", choices=["ollama", "keyword"], default="ollama",
                      help="'keyword' runs fully offline without Ollama")
    chat.add_argument("--model", help=f"Ollama model (default from env: {settings.model})")

    args = p.parse_args(argv)
    {"seed": cmd_seed, "chat": cmd_chat}[args.command](args, settings)


if __name__ == "__main__":
    main()
