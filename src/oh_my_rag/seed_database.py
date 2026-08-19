import argparse
import pickle

import chromadb
from chromadb.api.types import Metadata

from oh_my_rag.documents_loader import load
from oh_my_rag.build_chunks import build_chunks_from_docs
from oh_my_rag.build_embeddings import chunks_to_embeddings
from oh_my_rag.config import CHROMADB_PATH, CHUNKS_PATH

COLLECTION_NAME = "my_collection"


def seed_database(rebuild: bool = False, from_cache: bool = False):
    client = chromadb.PersistentClient(path=CHROMADB_PATH)

    if rebuild:
        try:
            print(f"[Rebuild]: remove collection: {COLLECTION_NAME}")
            client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    if from_cache:
        # Skips re-loading/re-chunking data/incoming entirely, reusing the chunks
        # pickled below on the last full run. Use this after changing
        # EMBEDDING_MODEL in config.py, to re-embed the same chunks with the new
        # model and upsert them (same ids => overwrites the old vectors) without
        # redoing document loading/chunking:
        #   uv run python -m oh_my_rag.seed_database --from-cache
        print(f"Loading chunks from cache: {CHUNKS_PATH}")
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        documents_count = len({c["src"]["path"] for c in chunks})
    else:
        print("Loading documents...")
        documents = load()
        chunks = build_chunks_from_docs(documents)
        documents_count = len(documents)

    if not chunks:
        print("No chunks to index.")
        return

    embeddings = chunks_to_embeddings(chunks)

    ids = [f"{c['src']['path']}::{c['chunk_id']}" for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas: list[Metadata] = [
        {"src": c["src"]["path"], "chunk_id": c["chunk_id"]} for c in chunks
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Indexed {len(chunks)} chunks from {documents_count} documents into '{CHROMADB_PATH}'.")


def parse_args():
    parser = argparse.ArgumentParser(description="Build/update the oh-my-rag index.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Wipe the existing collection before indexing (handles deleted/renamed source files).",
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
