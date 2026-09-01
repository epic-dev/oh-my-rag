# oh-my-rag

A small, fully local RAG (Retrieval-Augmented Generation) pipeline: chunk your
documents, embed them locally, store them in ChromaDB, and answer questions
about them with a local LLM. No API keys, no cloud calls.

## How it works

Retrieval is **hybrid**: every question is searched two ways at once and the rankings are
fused, because a small local embedding model alone tends to blur exact terms.

```
INDEXING  (uv run oh-my-rag)
  data/incoming/*.txt|.md
    -> documents_loader.load()
    -> build_chunks.build_chunks_from_docs()   (token-based chunking, tiktoken)
    -> seed_database.index_chunks()            writes BOTH stores under one uid:
         - Chroma vectors   ./chromadb_base    (sentence-transformers, local)
         - SQLite FTS5      ./fts_index.sqlite3

QUERY  (Rag class)
  question
    -> query_expansion.expand()      one LLM call: keywords + 2 rephrasings
    -> concurrently, on two threads:
         - dense  : Chroma, question + rephrasings in one batched call
         - lexical: SQLite FTS5, bm25() over the keywords
    -> fusion.reciprocal_rank_fusion()   RRF over the ranked lists -> TOP_K
    -> rag.build_prompt() -> local LLM  -> answer

runners/cli.py:
  a thin terminal chat loop over Rag — one example of a "runner"; write your own
  (web endpoint, Slack bot, ...) against the same class.

mcp/server.py:
  the same corpus over MCP — list_docs, read_doc, write_doc, search_docs.
```

Every step degrades rather than fails: if query expansion misbehaves the original question
is used, and if either retriever dies the other still answers. See
[HOW_IT_WORKS.md](HOW_IT_WORKS.md) for the walkthrough with line-level code references, and
[DESIGN.md](DESIGN.md) for why it is built this way.


## Requirements

- Python >= 3.14
- [uv](https://docs.astral.sh/uv/)
- A local OpenAI-compatible LLM server. Either works, both serve `:11434`:
  - **[Ollama](https://ollama.com)** (any platform) — `ollama pull qwen3.5:9b`,
    then set `LLM_MODEL = "qwen3.5:9b"` (use the name from `ollama list`).
  - **[mlx-lm](https://github.com/ml-explore/mlx-examples)** (macOS + Apple
    Silicon) — `./scripts/start_llm_server.sh`, and set `LLM_MODEL` to the
    mlx-community repo id.

  Only `LLM_MODEL` changes between them; `LLM_BASE_URL` is the same.

## Quickstart

**1. Install dependencies**
```bash
uv sync
```

**2. Download the embedding model locally** (one-time; required before seeding or querying)
```bash
uv run python -m oh_my_rag.embeddings_model_loader
```

**3. Add your documents**

Drop `.txt` / `.md` files into `data/incoming/`.

**4. Seed the database** (chunk + embed + index into Chroma *and* SQLite FTS5)
```bash
uv run oh-my-rag
```
- `--rebuild` — wipe and reindex both stores (needed after deleting or renaming
  source files, or after changing `EMBEDDING_MODEL` or `FTS_TOKENIZER`)
- `--from-cache` — currently parsed but ignored; see *Known gaps* in
  [HOW_IT_WORKS.md](HOW_IT_WORKS.md)

**5. Start a local LLM server**
```bash
# Ollama (runs as a service; nothing to start by hand)
ollama pull qwen3.5:9b        # then set LLM_MODEL = "qwen3.5:9b"

# or mlx-lm on Apple Silicon
uv tool install mlx-lm        # once
./scripts/start_llm_server.sh
```
Either serves an OpenAI-compatible API on `http://localhost:11434/v1`. Only one can
hold the port at a time.

**6. Chat with your documents**
```bash
uv run oh-my-rag-chat
uv run oh-my-rag-chat --debug       # full trace: expansion, each retriever's ranking, the fusion arithmetic, timings
uv run oh-my-rag-chat --debug --show 10   # list 10 hits per retriever instead of 5
uv run oh-my-rag-chat --no-expand   # skip the expansion call, search the question as typed
uv run oh-my-rag-chat --no-stream   # wait for the whole answer instead of streaming it
```
Type a question, `/exit` or `/quit` to leave. The answer streams by default.

`--debug` shows where the time actually goes, which is usually not where you expect:

```
── expansion ───────────────────────────────── 3.82s
── retrieval ──────────────────── 219ms wall, 2 threads
  vector       weight=1.0  20 hits   "What does PACELC add that CAP leaves out?"
  fts          weight=1.0  20 hits   29 chunk(s) matched before LIMIT 20
── fusion ──────────────────────────── RRF k=60 → top 5
  1. cap-theorem-vs-pacelc.md::2    0.05537
       = vector@1 (0.01639) + fts@1 (0.01639) + vector_alt1@2 (0.01129)
── answer ──────────────────────────────── streaming
  first token 17.68s  ·  332 chars in 21.18s  ·  16 char/s
```

**7. (Optional) Serve the corpus over MCP**
```bash
uv run oh-my-rag-mcp                    # stdio, for an MCP host
uv run oh-my-rag-mcp --http --port 8000 # streamable HTTP
uv run oh-my-rag-mcp-call               # list the tools; then e.g.
uv run oh-my-rag-mcp-call search_docs query="what is X?" top_k=3
```
Register it with Claude Code:
```bash
claude mcp add oh-my-rag -- uv run oh-my-rag-mcp
```
Tools: `list_docs`, `read_doc`, `write_doc` (writes *and* indexes immediately),
`search_docs` (returns ranked passages, not a generated answer).

## LLM usage log

Every call to the LLM — expansion and answering, streamed or not, successful or failed — is
appended to its own SQLite file, `./llm_usage.sqlite3` (gitignored, separate from the
retrieval indexes so you can wipe it freely).

```bash
uv run python -m oh_my_rag.usage_log     # per-kind totals
sqlite3 llm_usage.sqlite3 "SELECT * FROM llm_calls ORDER BY id DESC LIMIT 10;"
```

```
kind       calls  failed  prompt_tokens  completion_tokens  avg_ms   avg_messages
expansion  3      1       150            118                2749.0   2.0
answer     3      1       4101           176                10254.0  1.0
```

Columns: `ts`, `kind`, `model`, `base_url`, `streamed`, `messages_sent`, `prompt_tokens`,
`completion_tokens`, `total_tokens`, `prompt_chars`, `reply_chars`, `duration_ms`,
`finish_reason`, `ok`, `error`.

Token counts are `NULL` when the server does not report them; `prompt_chars` is always
recorded, so a row is never useless. Set `USAGE_LOG_ENABLED = False` to turn it off.

## Configuration

All settings live in `src/oh_my_rag/config.py`:

| Setting | Purpose |
|---|---|
| `LLM_BASE_URL` / `LLM_MODEL` | Full chat-completions endpoint URL (e.g. `http://localhost:11434/v1/chat/completions`) and model id |
| `LLM_TIMEOUT` / `LLM_MAX_TOKENS` | Answer-call timeout, and the completion cap (mlx-lm defaults to only 512, which truncates answers) |
| `LLM_REASONING_EFFORT` | `"none"` disables a reasoning model's scratchpad on the answer path. On qwen3.5:9b that is 23s vs 88s for equivalent answers |
| `EMBEDDING_MODEL` | sentence-transformers model — must be downloaded locally first (`embeddings_model_loader.py`) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Token-based chunk window/overlap (tiktoken `cl100k_base`) |
| `CHROMADB_PATH` | Where the Chroma index is persisted |
| `DATA_SOURCE_DIR` | Source documents folder |
| `FTS_PATH` / `FTS_TOKENIZER` | Full-text index location and its FTS5 tokenizer |
| `TOP_K` / `CANDIDATE_K` | Chunks sent to the LLM, and depth pulled per retriever before fusion |
| `RRF_K` / `RRF_WEIGHT_*` | Fusion constant and the weight each ranked list carries |
| `EXPANSION_*` | Query expansion: on/off, model, timeout, and the caps on its output |
| `MCP_SERVER_NAME` / `MCP_HTTP_PORT` | MCP server identity and `--http` port |
| `USAGE_DB_PATH` / `USAGE_LOG_ENABLED` | Where LLM calls are logged, and whether to log at all |

Changing `LLM_MODEL` requires restarting the mlx-lm server with the new model.
Changing `EMBEDDING_MODEL` requires re-downloading it and running
`uv run oh-my-rag --rebuild`. Changing `FTS_TOKENIZER` also requires `--rebuild` —
the code refuses to write into a mismatched index rather than let it degrade quietly.
