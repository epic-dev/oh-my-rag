# How oh-my-rag works

oh-my-rag has two independent pipelines that share one Chroma index: an
**indexing** pipeline that turns your documents into searchable vectors, and
a **query** pipeline that answers a question using those vectors plus a
local LLM. They never run in the same process automatically — you seed the
index once (or whenever documents change), then query it as many times as
you want.

## Schema

```
INDEXING  (uv run oh-my-rag)
  data/incoming/*.txt|.md|.mdx
        │  documents_loader.load()
        ▼
  raw documents
        │  build_chunks.build_chunks_from_docs()
        ▼
  text chunks
        │  build_embeddings.chunks_to_embeddings()
        ▼
  vectors  ──────────────────►  Chroma collection (./chromadb_base)

QUERY  (Rag class)
  question
        │  Rag.retrieve()      — embed question, search Chroma
        ▼
  top-N chunks
        │  Rag.build_prompt()  — wrap question + chunks in a prompt
        ▼
  prompt
        │  Rag.ask_llm()       — POST to a local OpenAI-compatible LLM
        ▼
  answer
```
