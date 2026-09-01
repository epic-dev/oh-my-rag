import argparse
import pickle

import chromadb
from chromadb.api.types import Metadata

from oh_my_rag.documents_loader import load
from oh_my_rag.build_chunks import Chunk
from oh_my_rag.build_chunks import build_chunks_from_docs
from oh_my_rag.config import CHROMADB_PATH, CHUNKS_PATH
from oh_my_rag.embeddings_model_loader import get_embedding_model
from oh_my_rag.search.fts import FtsIndex, FtsRow

COLLECTION_NAME = "my_collection"


def get_collection(rebuild: bool = False):
    client = chromadb.PersistentClient(path=CHROMADB_PATH)

    if rebuild:
        try:
            print(f"[Rebuild]: remove collection: {COLLECTION_NAME}")
            client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass

    return client.get_or_create_collection(name=COLLECTION_NAME)


def index_chunks(chunks: list[Chunk], collection=None, fts: FtsIndex | None = None) -> int:
    """Write chunks to both stores under the same uid.

    The single entry point for indexing, so seeding and the MCP `write_doc`
    tool cannot drift apart. Both writes are upserts keyed on the uid, which
    makes re-running this the repair for a partial failure.
    """
    if not chunks:
        return 0

    collection = collection if collection is not None else get_collection()
    fts = fts if fts is not None else FtsIndex()

    ids = [c.uid for c in chunks]
    texts = [c.text for c in chunks]
    metadatas: list[Metadata] = [
        {"src": c.src.path, "chunk_id": c.chunk_id} for c in chunks
    ]

    # Embed here rather than letting Chroma fall back to its bundled default
    # model: queries are embedded with EMBEDDING_MODEL, and both sides have to
    # live in the same vector space. Normalized, so the collection's l2 space
    # ranks the same as cosine (what bge-* models are trained for).
    embeddings = get_embedding_model().encode(
        texts,
        normalize_embeddings=True,
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    fts.upsert([
        FtsRow(chunk_uid=c.uid, src=c.src.path, text=c.text) for c in chunks
    ])

    return len(chunks)


def seed_database(rebuild: bool = False, from_cache: bool = False):
    fts = FtsIndex()
    collection = get_collection(rebuild=rebuild)
    if rebuild:
        print("[Rebuild]: drop FTS index")
        fts.drop()

    documents = load()
    chunks = build_chunks_from_docs(documents)
    documents_count = len(documents)

    if not chunks:
        print("No chunks to index.")
        return

    indexed = index_chunks(chunks, collection=collection, fts=fts)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(
        f"Indexed {indexed} chunks from {documents_count} documents "
        f"into '{CHROMADB_PATH}' (vector) and '{fts.path}' (full-text)."
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Build/update the oh-my-rag index.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Wipe the existing collection and FTS index before indexing "
             "(handles deleted/renamed source files).",
    )
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="Re-embed the chunks already saved in CHUNKS_PATH instead of re-loading "
             "and re-chunking data/incoming. Use after changing EMBEDDING_MODEL.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed_database(rebuild=args.rebuild, from_cache=args.from_cache)
