# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                              # install deps
uv run python -m oh_my_rag.embeddings_model_loader   # one-time: download EMBEDDING_MODEL into the HF cache
uv run oh-my-rag                                     # seed/update BOTH indexes from data/incoming
uv run oh-my-rag --rebuild                           # drop both indexes first (needed after deleting/renaming
                                                     #   sources, or changing EMBEDDING_MODEL / FTS_TOKENIZER)
./scripts/start_llm_server.sh                        # mlx-lm OpenAI-compatible server on :11434 (Apple Silicon; needs `uv tool install mlx-lm`)
uv run oh-my-rag-chat                                # terminal chat loop; /exit or /quit to leave
uv run oh-my-rag-chat --debug                        # + expansion, per-retriever ranks, fused ordering
uv run oh-my-rag-chat --no-expand                    # skip the expansion LLM call
uv run oh-my-rag-mcp                                 # MCP server over stdio (--http --port N for HTTP)
uv run oh-my-rag-mcp-call                            # list MCP tools; `... <tool> key=value` to call one
```

There is no test suite, linter, or formatter configured — don't invent commands for them.

Entry points are declared in `pyproject.toml` `[project.scripts]`.

## Architecture

Two pipelines that never run in the same process, joined by **two** persisted stores that
are written together and must stay in step:

- Chroma collection `my_collection` at `./chromadb_base` (dense vectors)
- SQLite FTS5 database at `./fts_index.sqlite3` (BM25 keyword search)

**Indexing** (`__init__.main` → `seed_database.seed_database`): `documents_loader.load()`
walks `DATA_SOURCE_DIR` for `.txt`/`.md` → `build_chunks.build_chunks_from_docs()` splits
with tiktoken `cl100k_base` → `seed_database.index_chunks()` embeds and writes **both**
stores.

**Query** (`rag.Rag.answer_verbose`): `query_expansion.expand()` → `search.hybrid.HybridSearch`
runs the dense and lexical branches on a 2-thread pool → `search.fusion.reciprocal_rank_fusion()`
→ `Rag.build_prompt()` → `Rag.ask_llm()` → `llm_client.chat()`.

`runners/cli.py` and `mcp/server.py` are frontends over `Rag`; new ones should be too.

For a full walkthrough with line-level references see `HOW_IT_WORKS.md`; for the design
rationale, phase plan and measurements see `DESIGN.md`.

### The chunk uid joins the two stores

`build_chunks.Chunk.uid` is `f"{src.path}::{chunk_id}"`. It is Chroma's document id **and**
FTS5's `chunk_uid`, and RRF matches hits across the two by comparing these strings. Defined
once as a property — never rebuild it inline.

The path in it must stay **relative** (`data/incoming/x.md`). `mcp.server.write_doc` converts
back to a relative path before indexing for exactly this reason; an absolute path would
produce a different uid and silently duplicate the document on the next reseed.

Anything that writes an index goes through `seed_database.index_chunks()`, so seeding and
`write_doc` cannot drift apart.

### Embeddings — keep both sides symmetric

Both pipelines embed **client-side, with the same model**, via
`embeddings_model_loader.get_embedding_model()` (`EMBEDDING_MODEL`, currently
`Qwen/Qwen3-Embedding-0.6B`, 1024-dim, `local_files_only`). Two invariants that are easy to
break silently:

- **Never pass `documents=` to `upsert()` without `embeddings=`.** Chroma then falls back to
  its own bundled default model (ONNX all-MiniLM-L6-v2). At 384-dim that now trips a
  dimension mismatch against a 1024-dim collection — but it was silent back when
  `EMBEDDING_MODEL` was also 384-dim, and would be silent again for any 384-dim model. The
  collection's stored config is `{}` (no embedding function baked in); keep it that way.
- **Both sides pass `normalize_embeddings=True`.** The vector index uses the l2 space, which
  ranks identically to cosine only for normalized vectors — and both bge-* and
  Qwen3-Embedding are trained for cosine.

`config.EMBEDDING_QUERY_PREFIX` is the query-side instruction and is deliberately *not*
applied to documents. It belongs to `EMBEDDING_MODEL` — each family words it differently
(Qwen3-Embedding's is copied verbatim from its `config_sentence_transformers.json`, and has
no space after `Query:`; bge wants `"Represent this sentence for searching relevant
passages: "`). A mismatch degrades retrieval silently. Change both together, then
`--rebuild` — the dimension change makes it mandatory anyway.

### Full-text search

`search/fts.py` owns the FTS5 schema. `chunk_uid`/`src` are `UNINDEXED`, so a side table
`chunk_rows` maps uid → rowid (FTS5 cannot index an UNINDEXED column, and deleting by uid
would otherwise scan the table). `fts_meta` records the tokenizer and makes a mismatch an
error instead of quiet degradation.

- **`bm25()` returns a NEGATIVE score** — `ORDER BY score ASC` is best-first. Don't "fix" it.
- **Never interpolate LLM output into a MATCH expression.** `search.fts.build_match_expr()`
  quotes each term into a phrase literal; `-`, `:`, `*`, `"`, `OR`, `NOT`, `NEAR` are FTS5 syntax.
- Porter stemming merges inflections (`retrieve`/retrieving, `policy`/policies) and some
  derivations (`retrieval`, `connection`, `computation`) but **not** `accrual`, `deployment`,
  `runner`, `policing`. Prompt for base nouns.

### Reasoning-model output needs scrubbing

`llm_client.strip_control_tokens()` handles three `<think>` shapes, all seen in practice:
a matched pair, an **orphan close** (`reasoning</think>answer` — the opener never arrived,
so everything before the tag is scratchpad), and an orphan open (truncated mid-thought).
Tags are matched case-insensitively and tolerate whitespace inside the brackets, because a
4-bit model does not reproduce them reliably. Order matters: complete blocks first.

Always send `max_tokens`. mlx-lm defaults to **512** completion tokens when the request
omits it, which silently truncates answers — and the `<think>` block spends part of that
budget before the answer starts. `LLM_MAX_TOKENS` covers the answer path,
`EXPANSION_MAX_TOKENS` the expansion call.

**Reasoning shares the `max_tokens` budget.** This is the trap when swapping in a bigger
reasoning model: at `EXPANSION_MAX_TOKENS = 200`, qwen3.5:9b spent all 200 tokens thinking
and returned *empty* content, so every expansion fell back. Neither `think=False` nor
`/no_think` is honoured on Ollama's OpenAI endpoint — `reasoning_effort` is
(`EXPANSION_REASONING_EFFORT`, `LLM_REASONING_EFFORT`). `llm_client.chat()` sends the field
only when set, so backends that don't know it are unaffected.

### Backends

`LLM_BASE_URL` targets any OpenAI-compatible `/chat/completions`. Ollama and mlx-lm both
serve that on `:11434`, so switching is one constant — `LLM_MODEL` — set to `ollama list`
naming (`qwen3.5:9b`) or an mlx-community repo id. Only one can hold the port.

Ollama returns a reasoning model's scratchpad in a separate `message.reasoning` field, so
`content` arrives clean; mlx-lm inlines it as `<think>` tags. `strip_control_tokens` handles
the inline case and is a harmless no-op for the other.

### Observability and streaming

`observability.Trace` is **data, never formatting** — timings, per-branch ranked lists, the
FTS MATCH expression and its pre-LIMIT candidate count, the fused result. `runners/cli.py`
renders it; MCP or a web frontend could consume the same object. Keep rendering out of it.

Tracing is opt-in: `Rag.prepare(query, trace=True)`. `HybridSearch.search` passes `branch=`
to the lexical retriever **only while tracing**, so a retriever implementing the minimal
`search(terms, limit)` contract still works — do not make it a required positional.

`Rag.prepare()` / `Rag.stream_answer()` are split so a caller can show retrieval *before*
generation starts. `llm_client.chat_stream()` scrubs reasoning incrementally via
`StreamScrubber`: it never emits past a trailing `<` that could begin a tag, so a tag split
across deltas is safe. Its one gap is an orphan `</think>` arriving after text is already
printed — `probation` buffers the opening N characters to catch it, and `Rag` sets
`probation=0` when `LLM_REASONING_EFFORT == "none"` because then no scratchpad exists.
`delta.reasoning` is ignored, matching the non-streaming path.

### The usage log

`usage_log.UsageLog` appends every LLM call to `./llm_usage.sqlite3` — its own database, so
rotating it never touches the retrieval indexes. Recorded from `llm_client.chat()` and
`chat_stream()`, the single choke point every call passes through, so a new caller is logged
automatically; it only needs to pass `kind=`.

- **Logging must never break a query.** `record()` swallows everything and warns on stderr.
- **Failures are logged too**, before the exception propagates — a timeout or refused
  connection is exactly the traffic worth having a record of.
- **`chat_stream` sends `stream_options={"include_usage": True}`.** An SSE stream carries no
  usage block otherwise, and the usage chunk arrives last with no `choices`.
- Token counts are `NULL` when the server reports none; `prompt_chars` is always recorded.
- `messages_sent` is the request's message count, not conversation turns: **answering sends
  1, expansion sends 2** (system + user). Both are single-turn.

### Degradation is a feature, not an accident

- `query_expansion.expand()` **never raises** — every failure returns the original query with
  `expanded=False` and a `reason`. Keep it that way.
- `HybridSearch._settle()` isolates each branch: FTS dying leaves pure vector search, Chroma
  dying leaves pure keyword search.
- `FtsIndex.search()` swallows `sqlite3.OperationalError` and returns `[]`.
- Failures log a reason and set a flag rather than degrading silently — that visibility is
  what makes retrieval quality debuggable.

### MCP

`mcp/server.py` exposes `list_docs`, `read_doc`, `write_doc`, `search_docs` via `fastmcp` 3.x
(`Transport = Literal['stdio','http','sse','streamable-http']`; the server uses `'http'`).

- `_resolve()` confines every path to `DATA_SOURCE_DIR`; `Path.resolve()` collapses `..` and
  follows symlinks before the containment check. Don't loosen it.
- Model and store loading is **lazy** — an MCP host expects a fast stdio handshake.
- `search_docs` returns ranked chunks, not a generated answer: the calling host has its own model.
- **Never `print()` to stdout from library code.** Under the stdio server, stdout IS the
  JSON-RPC channel — a stray line is a `JSONRPCMessage` validation error and a dead request.
  Diagnostics in `query_expansion`, `search/`, and `documents_loader` all pass `file=sys.stderr`.
  CLI output in `runners/` and `seed_database` is fine on stdout; those never run under MCP.

## Other known gaps

- `--from-cache` is parsed and threaded into `seed_database(from_cache=...)` but the body
  ignores it; `chunks.pkl` (`CHUNKS_PATH`) is written on every seed and never read back.
- A plain reseed never removes chunks whose source file is gone — only `--rebuild` does. Both
  stores behave identically here on purpose.

## Configuration

Everything is module-level constants in `src/oh_my_rag/config.py` — no env vars, no `.env`, no
CLI overrides (only the seeding and runner flags). Changing `LLM_MODEL` means restarting the
mlx-lm server with the matching model; `scripts/start_llm_server.sh` reads `LLM_MODEL`/`LLM_PORT`
from the shell environment, which is *not* wired to `config.py`, so both must be kept in sync by hand.

`chromadb_base/`, `fts_index.sqlite3*`, `chunks.pkl`, and `data/incoming` are gitignored — the
indexes and the user's documents are local-only, and a fresh clone must be seeded before
`oh-my-rag-chat` will work (`Rag.__init__` touches the collection, which raises if nothing was
ever indexed).
