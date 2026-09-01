import tiktoken
from dataclasses import dataclass

from oh_my_rag.config import (CHUNK_SIZE, CHUNK_OVERLAP)
from oh_my_rag.documents_loader import Document

@dataclass
class Chunk:
    chunk_id: int
    text: str
    src: Document

    @property
    def uid(self) -> str:
        """Identity of this chunk in *both* stores — Chroma's id and FTS5's
        chunk_uid. Fusion matches hits across the two on this string, so it is
        defined here once and never rebuilt inline."""
        return f"{self.src.path}::{self.chunk_id}"

cl100k_base = tiktoken.get_encoding("cl100k_base")

def build_chunks_from_text(text) -> list[str]:
    tokens = cl100k_base.encode(text)
    chunks = []

    step = CHUNK_SIZE - CHUNK_OVERLAP
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i + CHUNK_SIZE]
        chunks.append(cl100k_base.decode(chunk_tokens))
        if i + CHUNK_SIZE >= len(tokens):
            break
        i += step

    return chunks

def build_chunks_from_docs(documents: list[Document]) -> list[Chunk]:
    chunks = []

    for doc in documents:
        _chunks = build_chunks_from_text(doc.text)
        for i, chunk in enumerate(_chunks):
            chunks.append(Chunk(
                chunk_id=i,
                text=chunk,
                src=doc,
            ))

    return chunks