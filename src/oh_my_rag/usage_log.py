import sqlite3
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

from oh_my_rag.config import USAGE_DB_PATH, USAGE_LOG_ENABLED

SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class LlmCall:
    """One request to the LLM, successful or not."""

    kind: str                      # "expansion" | "answer" | caller-supplied
    model: str
    base_url: str
    streamed: bool
    messages_sent: int             # messages in the request payload
    prompt_chars: int              # always known, unlike tokens
    duration_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reply_chars: int | None = None
    finish_reason: str | None = None
    ok: bool = True
    error: str | None = None


class UsageLog:
    """Append-only record of LLM traffic, in its own SQLite file.

    Holds no connection: like FtsIndex, it opens one per thread on first use,
    because the expansion and answer calls can arrive from different threads.

    Logging must never break a query, so every write is guarded — a failure here
    warns on stderr and is otherwise swallowed.
    """

    def __init__(self, path: str = USAGE_DB_PATH, enabled: bool = USAGE_LOG_ENABLED):
        self.path = path
        self.enabled = enabled
        self._local = threading.local()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path)
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            self._ensure_schema(conn)
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS llm_calls (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                ts                TEXT    NOT NULL,
                kind              TEXT    NOT NULL,
                model             TEXT    NOT NULL,
                base_url          TEXT    NOT NULL,
                streamed          INTEGER NOT NULL,
                messages_sent     INTEGER NOT NULL,
                prompt_tokens     INTEGER,
                completion_tokens INTEGER,
                total_tokens      INTEGER,
                prompt_chars      INTEGER NOT NULL,
                reply_chars       INTEGER,
                duration_ms       REAL    NOT NULL,
                finish_reason     TEXT,
                ok                INTEGER NOT NULL,
                error             TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_llm_calls_ts   ON llm_calls(ts);
            CREATE INDEX IF NOT EXISTS idx_llm_calls_kind ON llm_calls(kind);
            CREATE TABLE IF NOT EXISTS usage_meta (
                key TEXT PRIMARY KEY, value TEXT NOT NULL
            );
            """
        )
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO usage_meta (key, value) VALUES ('schema_version', ?)",
                (SCHEMA_VERSION,),
            )

    def record(self, call: LlmCall) -> int | None:
        if not self.enabled:
            return None
        try:
            conn = self._conn()
            with conn:
                cur = conn.execute(
                    """
                    INSERT INTO llm_calls (
                        ts, kind, model, base_url, streamed, messages_sent,
                        prompt_tokens, completion_tokens, total_tokens,
                        prompt_chars, reply_chars, duration_ms, finish_reason,
                        ok, error
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                        call.kind, call.model, call.base_url,
                        int(call.streamed), call.messages_sent,
                        call.prompt_tokens, call.completion_tokens, call.total_tokens,
                        call.prompt_chars, call.reply_chars, call.duration_ms,
                        call.finish_reason, int(call.ok), call.error,
                    ),
                )
            return cur.lastrowid
        except Exception as exc:
            print(f"[usage] could not record LLM call: {exc}", file=sys.stderr)
            return None

    # --- reading -------------------------------------------------------------

    def summary(self) -> list[dict]:
        """Per-kind totals. `calls` counts every attempt; `failed` counts errors."""
        rows = self._conn().execute(
            """
            SELECT kind,
                   count(*)                        AS calls,
                   sum(ok = 0)                     AS failed,
                   sum(prompt_tokens)              AS prompt_tokens,
                   sum(completion_tokens)          AS completion_tokens,
                   sum(total_tokens)               AS total_tokens,
                   round(avg(duration_ms))         AS avg_ms,
                   round(avg(messages_sent), 2)    AS avg_messages
            FROM llm_calls GROUP BY kind ORDER BY calls DESC
            """
        ).fetchall()
        cols = ["kind", "calls", "failed", "prompt_tokens", "completion_tokens",
                "total_tokens", "avg_ms", "avg_messages"]
        return [dict(zip(cols, r)) for r in rows]

    def count(self) -> int:
        return self._conn().execute("SELECT count(*) FROM llm_calls").fetchone()[0]


_log: UsageLog | None = None


def get_usage_log() -> UsageLog:
    """Process-wide singleton, so every caller appends to the same file."""
    global _log
    if _log is None:
        _log = UsageLog()
    return _log


def _print_summary() -> None:
    log = get_usage_log()
    rows = log.summary()
    if not rows:
        print(f"No LLM calls recorded in {log.path}.")
        return

    cols = ["kind", "calls", "failed", "prompt_tokens", "completion_tokens",
            "total_tokens", "avg_ms", "avg_messages"]
    widths = {c: max(len(c), *(len(str(r[c] if r[c] is not None else "-")) for r in rows))
              for c in cols}
    print(f"{log.path}  ({log.count()} calls)\n")
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(str(r[c] if r[c] is not None else "-").ljust(widths[c]) for c in cols))


if __name__ == "__main__":
    _print_summary()
