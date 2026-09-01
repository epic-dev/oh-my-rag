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
- A local OpenAI-compatible LLM server. `scripts/start_llm_server.sh` uses
  [mlx-lm](https://github.com/ml-explore/mlx-examples) (macOS + Apple Silicon
  only) — point `LLM_BASE_URL` in `config.py` at anything else (e.g. Ollama)
  if you're not on Apple Silicon.

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

**5. Start a local LLM server** (Apple Silicon, via mlx-lm)
```bash
uv tool install mlx-lm   # once
./scripts/start_llm_server.sh
```
Serves an OpenAI-compatible API on `http://localhost:11434/v1`.

**6. Chat with your documents**
```bash
uv run oh-my-rag-chat
uv run oh-my-rag-chat --debug       # show the expansion, which retriever found what, and the fused ranking
uv run oh-my-rag-chat --no-expand   # skip the expansion call, search the question as typed
```
Type a question, `/exit` or `/quit` to leave.

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

## Configuration

All settings live in `src/oh_my_rag/config.py`:

| Setting | Purpose |
|---|---|
| `LLM_BASE_URL` / `LLM_MODEL` | Full chat-completions endpoint URL (e.g. `http://localhost:11434/v1/chat/completions`) and model id |
| `EMBEDDING_MODEL` | sentence-transformers model — must be downloaded locally first (`embeddings_model_loader.py`) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Token-based chunk window/overlap (tiktoken `cl100k_base`) |
| `CHROMADB_PATH` | Where the Chroma index is persisted |
| `DATA_SOURCE_DIR` | Source documents folder |
| `FTS_PATH` / `FTS_TOKENIZER` | Full-text index location and its FTS5 tokenizer |
| `TOP_K` / `CANDIDATE_K` | Chunks sent to the LLM, and depth pulled per retriever before fusion |
| `RRF_K` / `RRF_WEIGHT_*` | Fusion constant and the weight each ranked list carries |
| `EXPANSION_*` | Query expansion: on/off, model, timeout, and the caps on its output |
| `MCP_SERVER_NAME` / `MCP_HTTP_PORT` | MCP server identity and `--http` port |

Changing `LLM_MODEL` requires restarting the mlx-lm server with the new model.
Changing `EMBEDDING_MODEL` requires re-downloading it and running
`uv run oh-my-rag --rebuild`. Changing `FTS_TOKENIZER` also requires `--rebuild` —
the code refuses to write into a mismatched index rather than let it degrade quietly.
