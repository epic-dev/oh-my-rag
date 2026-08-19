# oh-my-rag

A small, fully local RAG (Retrieval-Augmented Generation) pipeline: chunk your
documents, embed them locally, store them in ChromaDB, and answer questions
about them with a local LLM. No API keys, no cloud calls.

## How it works

```
data/incoming/*.txt|.md
  -> documents_loader.load()
  -> build_chunks.build_chunks_from_docs()      (token-based chunking, tiktoken)
  -> build_embeddings.chunks_to_embeddings()    (sentence-transformers, local)
  -> ChromaDB collection, persisted at ./chromadb_base

rag.py (Rag class):
  question -> embed -> retrieve top-N chunks from Chroma -> build a prompt
  -> ask a local LLM (OpenAI-compatible /chat/completions) -> return answer

runners/cli.py:
  a thin terminal chat loop built on top of Rag — one example of a "runner";
  write your own (web endpoint, Slack bot, ...) against the same Rag class.
```


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

**4. Seed the database** (chunk + embed + index into Chroma)
```bash
uv run oh-my-rag
```
- `--rebuild` — wipe and reindex everything (e.g. after deleting source files)
- `--from-cache` — re-embed the last chunked set without re-reading
  `data/incoming` (e.g. after changing `EMBEDDING_MODEL`)

**5. Start a local LLM server** (Apple Silicon, via mlx-lm)
```bash
uv tool install mlx-lm   # once
./scripts/start_llm_server.sh
```
Serves an OpenAI-compatible API on `http://localhost:11434/v1`.

**6. Chat with your documents**
```bash
uv run oh-my-rag-chat
```
Type a question, `/exit` or `/quit` to leave.

## Configuration

All settings live in `src/oh_my_rag/config.py`:

| Setting | Purpose |
|---|---|
| `LLM_BASE_URL` / `LLM_MODEL` | Full chat-completions endpoint URL (e.g. `http://localhost:11434/v1/chat/completions`) and model id |
| `EMBEDDING_MODEL` | sentence-transformers model — must be downloaded locally first (`embeddings_model_loader.py`) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Token-based chunk window/overlap (tiktoken `cl100k_base`) |
| `CHROMADB_PATH` | Where the Chroma index is persisted |
| `DATA_SOURE_DIR` | Source documents folder |

Changing `LLM_MODEL` requires restarting the mlx-lm server with the new
model. Changing `EMBEDDING_MODEL` requires re-downloading it and re-seeding
(`uv run oh-my-rag --from-cache` reuses already-chunked text, skipping a full
re-read of `data/incoming`).
