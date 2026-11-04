# nimble-recs

A chat CLI that gives **personalised suggestions** using a local **decision model** (Nimble, served by [Ollama](https://ollama.com)).

Everything you type is saved as search history. Each new message runs a **two-stage decision**:

```
you type a query
      │
      ▼
save to history (Postgres)
      │
      ▼
Stage 1: decision model picks the DOMAIN          ← sees: query + recent history + 20 domain descriptions
      │
      ▼
load that domain's questions
      │
      ▼
Stage 2: decision model picks the top 3 QUESTIONS ← sees: query + recent history + that domain's questions
      │
      ▼
print suggestions (+ the model's reason)
```

Each stage is a small choice, so the prompt stays well under Nimble's ~8K token context. History is trimmed (oldest first) if a prompt would get too big.

---

## Quick start

### 1. Prerequisites

- Python 3.10+
- [Docker](https://docs.docker.com/get-docker/) (for Postgres)
- [Ollama](https://ollama.com/download)

### 2. Install

```bash
git clone https://github.com/JamesThackeray/hello.git
cd hello
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
```

### 3. Start Postgres and load the data

```bash
docker compose up -d
nimble-recs seed
```

### 4. Get the model

```bash
ollama pull nimble
```

### 5. Chat

```bash
nimble-recs chat
```

Example output (the exact wording depends on the model):

```
default> how do I open a bank account when I move to the UK
  domain: Personal Finance
    why: The user is asking about opening a bank account.
  1. How do I open a UK bank account as a newcomer?
  2. How can I build a credit score from scratch?
  3. What is the cheapest way to send money abroad?
```

### Try it without Ollama or Docker

```bash
nimble-recs chat --memory --decider keyword
```

`--memory` loads the CSVs into memory instead of Postgres. History is lost when you quit. `--decider keyword` uses a simple word-overlap baseline instead of an LLM.

---

## Chat commands

| Command | What it does |
|---|---|
| *(any text)* | Search: saved to history, returns suggestions |
| `/clear` or `clear history` | Delete your history (and its logged suggestions) |
| `/history` | Show recent history |
| `/domains` | List domains |
| `/user <name>` | Switch user. Each user has their own history |
| `/help`, `/quit` | Show help, exit |

You can also start as a specific user: `nimble-recs chat --user james`.

---

## Your data

The seed data lives in two CSVs in `data/`:

**`data/domains.csv`**
```csv
id,name,description
travel,Travel,"Holidays, flights, trains, packing, travel insurance, city breaks and itineraries"
```

**`data/questions.csv`**
```csv
id,domain_id,text
travel-01,travel,"What are the cheapest ways to travel between UK cities?"
```

It ships with **20 domains × 10 example questions**. Replace them with your real 50 per domain, then re-run `nimble-recs seed`. Seeding is an upsert, and it removes rows that are no longer in the CSVs.

> **Tip:** Stage 1 only sees each domain's name and description, so a rich description ("flights, visas, commuting…") makes a big difference to accuracy.

### Database tables

| Table | Contents |
|---|---|
| `domains` | id, name, description |
| `questions` | id, domain_id, text |
| `history` | every query per user, with a timestamp |
| `suggestions` | what was suggested for each query (rank, and whether the fallback was used). Useful for evals |

---

## Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | `postgresql://recs:recs@localhost:5432/recs` | Postgres connection |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server |
| `MODEL` | `nimble` | Any Ollama model (can override with `--model`) |
| `HISTORY_WINDOW` | `15` | How many past queries the model sees |
| `TOP_DOMAINS` | `1` | Set to `2` so stage 2 picks from the top two domains. This is a safety net if stage 1 guesses wrong |
| `NUM_SUGGESTIONS` | `3` | Questions to suggest |
| `MAX_PROMPT_TOKENS` | `6000` | Prompt budget. History is trimmed to fit |

---

## Swapping the model

All model calls go through one interface in `src/nimble_recs/decider.py`:

```python
choose(task, context, options, k) -> Choice(ids, reason)
```

- **`OllamaDecider`** calls Ollama's `/api/chat` with a **JSON schema**. The `enum` of option ids means the model can only answer with real domain or question ids. Results are still validated, and the call is retried once on bad output.
- **`KeywordDecider`** is an offline word-overlap baseline. It is also the **automatic fallback**: if Ollama is down or returns junk, you still get suggestions, plus a `!` warning.

Try other models with no code changes:

```bash
nimble-recs chat --model qwen3:8b
```

```bash
nimble-recs chat --model llama3.1:8b
```

> **Note on Nimble:** Ollama's decision models are new, and the docs disagree on whether they're served through the normal `/api/chat` endpoint or a newer decision endpoint (`/v1/systemone`). This project uses `/api/chat`. If `nimble` returns an error there, run with another model for now (`--model qwen3:8b`), or add a new decider class for the decision endpoint. Nothing else needs to change.
>
> Hardware: Nimble is a 9B model (about 5.6 GB quantised, 9.5 GB full). Plan on 16 GB of RAM, or a GPU with 8 GB or more.

---

## Project layout

```
src/nimble_recs/
  cli.py          chat loop + `seed` command
  recommender.py  the two-stage pipeline + history trimming
  decider.py      OllamaDecider, KeywordDecider (baseline/fallback)
  store.py        PostgresStore, MemoryStore, CSV loader
  schema.sql      tables (applied automatically)
  config.py       .env settings
data/             domains.csv, questions.csv
tests/            pytest suite (no Ollama/Postgres needed)
```

## Tests

```bash
pytest
```

The tests use the in-memory store and a mocked Ollama server, so they run anywhere.

---

## Ideas for next steps

- **Evals:** write ~50 test queries labelled with the correct domain and question, then measure **domain accuracy** (stage 1) and **hit@3** (stage 2) for Nimble vs. `qwen3:8b` vs. the keyword baseline.
- **Feedback:** let the user pick a suggestion (`/pick 2`) and store it, so future rankings can learn from it.
- **Weight recent history more** or summarise long history into a short "user profile".
