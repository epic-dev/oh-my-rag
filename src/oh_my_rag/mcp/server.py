import argparse
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from oh_my_rag.config import (
    DATA_SOURCE_DIR,
    MCP_HTTP_PORT,
    MCP_SERVER_NAME,
    TOP_K,
)
from oh_my_rag.documents_loader import Document, supported_formats

mcp = FastMCP(
    MCP_SERVER_NAME,
    instructions=(
        "Hybrid (vector + full-text) search over a local document collection. "
        "search_docs returns ranked passages with their source files; read_doc "
        "and list_docs reach the underlying files; write_doc adds a document and "
        "indexes it immediately."
    ),
)

# --- lazy singletons ---------------------------------------------------------
# An MCP host spawns this over stdio and expects a fast handshake. Loading
# sentence-transformers at import time risks a startup timeout, so nothing heavy
# is touched until a tool that needs it is actually called.
_rag = None
_fts = None


def _get_rag():
    global _rag
    if _rag is None:
        from oh_my_rag.rag import Rag

        _rag = Rag()
    return _rag


def _get_fts():
    global _fts
    if _fts is None:
        from oh_my_rag.search.fts import FtsIndex

        _fts = FtsIndex()
    return _fts


# --- path safety -------------------------------------------------------------

def _data_root() -> Path:
    return Path(DATA_SOURCE_DIR).resolve()


def _resolve(path: str) -> Path:
    """Resolve a caller-supplied path inside the corpus, or refuse.

    resolve() collapses `..` and follows symlinks before the containment check,
    so neither traversal nor a symlink pointing outside can escape.
    """
    root = _data_root()
    candidate = (root / path).resolve()
    if candidate != root and not candidate.is_relative_to(root):
        raise ToolError(f"path escapes {DATA_SOURCE_DIR}: {path!r}")
    return candidate


# --- tools -------------------------------------------------------------------

@mcp.tool
def list_docs(subdir: str = "") -> list[dict]:
    """List indexable documents in the collection.

    Args:
        subdir: Optional folder beneath the corpus root to limit the listing to.
    """
    base = _resolve(subdir) if subdir else _data_root()
    if not base.is_dir():
        raise ToolError(f"not a directory: {subdir or DATA_SOURCE_DIR}")

    root = _data_root()
    return sorted(
        (
            {
                "path": str(p.relative_to(root)),
                "bytes": p.stat().st_size,
            }
            for p in base.rglob("*")
            if p.is_file() and p.suffix in supported_formats
        ),
        key=lambda d: d["path"],
    )


@mcp.tool
def read_doc(path: str) -> str:
    """Return the full text of one document.

    Args:
        path: Path relative to the corpus root, as returned by list_docs.
    """
    target = _resolve(path)
    if not target.is_file():
        raise ToolError(f"no such document: {path}")
    if target.suffix not in supported_formats:
        raise ToolError(
            f"unsupported type {target.suffix!r}; expected one of "
            f"{sorted(supported_formats)}"
        )
    return target.read_text(encoding="utf-8")


@mcp.tool
def write_doc(path: str, text: str) -> dict:
    """Write a document into the collection and index it immediately.

    The new text is searchable on the very next search_docs call — no reseed.

    Args:
        path: Path relative to the corpus root. Must end in .txt or .md.
        text: Full contents of the document.
    """
    target = _resolve(path)
    if target.suffix not in supported_formats:
        raise ToolError(
            f"unsupported type {target.suffix!r}; expected one of "
            f"{sorted(supported_formats)}"
        )

    from oh_my_rag.build_chunks import build_chunks_from_docs
    from oh_my_rag.seed_database import index_chunks

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")

    root = _data_root()
    # The uid must match what a full reseed would produce, and seeding walks
    # DATA_SOURCE_DIR relatively — so store the relative path, not the absolute.
    doc_path = str(Path(DATA_SOURCE_DIR) / target.relative_to(root))
    chunks = build_chunks_from_docs([Document(path=doc_path, text=text)])
    indexed = index_chunks(
        chunks, collection=_get_rag().search.vector.collection, fts=_get_fts()
    )

    return {"path": str(target.relative_to(root)), "chunks_indexed": indexed}


@mcp.tool
def search_docs(query: str, top_k: int = TOP_K, expand: bool = True) -> list[dict]:
    """Hybrid search: dense vector retrieval and BM25 full-text, fused by RRF.

    Returns ranked passages rather than a generated answer, so the calling model
    can reason over the evidence itself.

    Args:
        query: A natural-language question.
        top_k: How many passages to return.
        expand: Ask the local LLM to add keywords and alternate phrasings first.
            Falls back to the raw query if that call fails.
    """
    rag = _get_rag()
    expanded = rag.expand_query(query, expand_queries=expand)
    hits = rag.search_expanded(expanded, top_k=top_k)

    return [
        {
            "rank": i,
            "src": hit.src,
            "chunk_uid": hit.chunk_uid,
            "text": hit.text,
            "score": round(hit.score, 6),
            "matched_by": sorted(hit.sources),
        }
        for i, hit in enumerate(hits, start=1)
    ]


# --- entry point -------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run the oh-my-rag MCP server.")
    parser.add_argument(
        "--http",
        action="store_true",
        help="Serve over streamable HTTP instead of stdio.",
    )
    parser.add_argument(
        "--port", type=int, default=MCP_HTTP_PORT, help="Port for --http."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.http:
        # 'http' is the current name for streamable HTTP in fastmcp 3.x
        # (Transport = Literal['stdio', 'http', 'sse', 'streamable-http']).
        mcp.run(transport="http", port=args.port)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
