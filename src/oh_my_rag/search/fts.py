import sys
import re
import sqlite3
import threading
from dataclasses import dataclass

from oh_my_rag.config import CANDIDATE_K, FTS_PATH, FTS_TOKENIZER
from oh_my_rag.search.types import SearchHit

# Bumped whenever the table layout changes in a way that needs a --rebuild.
SCHEMA_VERSION = "1"

# The tokenizer cannot be bound as a parameter — it goes into DDL as a literal —
# so it is checked against the shape FTS5 accepts rather than trusted blindly.
_TOKENIZER_RE = re.compile(r"^[a-z0-9_]+( [a-z0-9_]+)*$")


def build_match_expr(terms: list[str]) -> str:
    """Turn arbitrary terms into a safe FTS5 MATCH expression.

    Terms come from an LLM, and FTS5's query language treats `-`, `:`, `*`,
    `"`, `(`, `)`, `OR`, `NOT` and `NEAR` as syntax. Wrapping each term in
    double quotes makes it a phrase literal — safe, and the right meaning for a
    multi-word keyword. Terms with no alphanumeric character tokenize to
    nothing and are dropped rather than producing an empty MATCH.
    """
    usable = [t.strip() for t in terms if t and any(ch.isalnum() for ch in t)]
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in usable)


@dataclass(frozen=True)
class FtsRow:
    chunk_uid: str
    src: str
    text: str


class FtsIndex:
    """The lexical half of hybrid search: an FTS5 table keyed by the same
    chunk uid Chroma uses, so fusion can match a keyword hit to a vector hit.

    Holds no connection of its own. SQLite connections cannot cross threads, and
    the search path runs on a worker thread, so each thread opens its own on
    first use.
    """

    def __init__(self, path: str = FTS_PATH, tokenizer: str = FTS_TOKENIZER):
        if not _TOKENIZER_RE.match(tokenizer):
            raise ValueError(
                f"FTS_TOKENIZER {tokenizer!r} is not a bare FTS5 tokenizer spec "
                "(lowercase words separated by single spaces)."
            )
        self.path = path
        self.tokenizer = tokenizer
        self._local = threading.local()

    # --- connection ----------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path)
            # WAL is a property of the database file, set once and persistent.
            # It lets a search read while an MCP write_doc is committing.
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            self._ensure_schema(conn)
            self._local.conn = conn
        return conn

    def close(self) -> None:
        """Close this thread's connection, if it has one."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # --- schema --------------------------------------------------------------

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        try:
            conn.executescript(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    text,
                    chunk_uid UNINDEXED,
                    src       UNINDEXED,
                    tokenize = '{self.tokenizer}'
                );

                -- FTS5 cannot index an UNINDEXED column, so deleting a chunk by
                -- uid would scan the whole table. This maps uid -> rowid so a
                -- re-seed is a rowid hit instead.
                CREATE TABLE IF NOT EXISTS chunk_rows (
                    chunk_uid TEXT PRIMARY KEY,
                    fts_rowid INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS fts_meta (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
        except sqlite3.OperationalError as exc:
            if "no such module: fts5" in str(exc):
                raise RuntimeError(
                    "This Python's sqlite3 was built without FTS5, which hybrid "
                    "search requires. Check: python -c \"import sqlite3; "
                    "sqlite3.connect(':memory:').execute('create virtual table "
                    "t using fts5(x)')\""
                ) from exc
            raise

        self._check_meta(conn)

    def _check_meta(self, conn: sqlite3.Connection) -> None:
        """Refuse to write into an index built with different settings.

        A tokenizer mismatch does not raise on its own — it just silently ranks
        against differently-tokenized rows, which is exactly the class of quiet
        degradation the embedding invariants already guard against.
        """
        stored = dict(conn.execute("SELECT key, value FROM fts_meta").fetchall())
        wanted = {"schema_version": SCHEMA_VERSION, "tokenizer": self.tokenizer}

        for key, want in wanted.items():
            have = stored.get(key)
            if have is not None and have != want:
                raise RuntimeError(
                    f"FTS index at '{self.path}' was built with {key}={have!r}, "
                    f"but config says {want!r}. Run `uv run oh-my-rag --rebuild`."
                )

        if stored != wanted:
            with conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO fts_meta (key, value) VALUES (?, ?)",
                    sorted(wanted.items()),
                )

    # --- writes --------------------------------------------------------------

    def upsert(self, rows: list[FtsRow]) -> int:
        """Insert or replace rows, keyed by chunk_uid. Idempotent, like the
        Chroma upsert it runs beside."""
        if not rows:
            return 0

        conn = self._conn()
        with conn:
            for row in rows:
                existing = conn.execute(
                    "SELECT fts_rowid FROM chunk_rows WHERE chunk_uid = ?",
                    (row.chunk_uid,),
                ).fetchone()
                if existing is not None:
                    conn.execute(
                        "DELETE FROM chunks_fts WHERE rowid = ?", (existing[0],)
                    )

                cur = conn.execute(
                    "INSERT INTO chunks_fts (text, chunk_uid, src) VALUES (?, ?, ?)",
                    (row.text, row.chunk_uid, row.src),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO chunk_rows (chunk_uid, fts_rowid) "
                    "VALUES (?, ?)",
                    (row.chunk_uid, cur.lastrowid),
                )
        return len(rows)

    def drop(self) -> None:
        """Wipe the index. The FTS half of `--rebuild`."""
        conn = self._conn()
        with conn:
            conn.executescript(
                """
                DROP TABLE IF EXISTS chunks_fts;
                DROP TABLE IF EXISTS chunk_rows;
                DROP TABLE IF EXISTS fts_meta;
                """
            )
        self._ensure_schema(conn)

    # --- reads ---------------------------------------------------------------

    def search(
        self, terms: list[str], limit: int = CANDIDATE_K, branch=None
    ) -> list[SearchHit]:
        """Rank chunks against a set of terms, best first.

        Never raises on bad input: a malformed expression or a term set that
        tokenizes to nothing returns an empty list, so a bad keyword can only
        cost the lexical half of a search, never the whole query.
        """
        expr = build_match_expr(terms)
        if branch is not None:
            branch.detail = expr or "(no usable terms)"
        if not expr:
            return []

        try:
            rows = self._conn().execute(
                """
                SELECT chunk_uid, src, text, bm25(chunks_fts) AS score
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY score ASC
                LIMIT ?
                """,
                (expr, limit),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            print(f"[fts] query failed, skipping lexical branch: {exc}", file=sys.stderr)
            if branch is not None:
                branch.failed = True
                branch.detail = f"{expr}  -> {exc}"
            return []

        if branch is not None:
            # How much of the corpus the expression actually matched, before
            # LIMIT — the number that tells you whether the keywords were
            # selective or matched half the index.
            try:
                branch.candidates = self._conn().execute(
                    "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH ?", (expr,)
                ).fetchone()[0]
            except sqlite3.OperationalError:
                pass

        return [
            SearchHit(chunk_uid=uid, src=src, text=text, score=float(score), rank=rank)
            for rank, (uid, src, text, score) in enumerate(rows, start=1)
        ]

    def count(self) -> int:
        return self._conn().execute("SELECT count(*) FROM chunks_fts").fetchone()[0]
