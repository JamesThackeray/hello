-- Applied automatically by `nimble-recs seed` (idempotent).

CREATE TABLE IF NOT EXISTS domains (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS questions (
    id        TEXT PRIMARY KEY,
    domain_id TEXT NOT NULL REFERENCES domains(id) ON DELETE CASCADE,
    text      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS questions_domain_idx ON questions (domain_id);

-- Everything the user types into the chat CLI.
CREATE TABLE IF NOT EXISTS history (
    id         BIGSERIAL PRIMARY KEY,
    user_id    TEXT NOT NULL,
    query      TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS history_user_time_idx ON history (user_id, created_at DESC);

-- What we suggested for each query (useful later for evals / feedback).
CREATE TABLE IF NOT EXISTS suggestions (
    id          BIGSERIAL PRIMARY KEY,
    history_id  BIGINT NOT NULL REFERENCES history(id) ON DELETE CASCADE,
    domain_id   TEXT NOT NULL,
    question_id TEXT NOT NULL,
    rank        INT NOT NULL,
    fallback    BOOLEAN NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
