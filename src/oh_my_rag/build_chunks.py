import tiktoken
from .config import (CHUNK_SIZE, CHUNK_OVERLAP)

cl100k_base = tiktoken.get_encoding("cl100k_base")

def build_chunks_from_text(text):
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

def build_chunks_from_docs(documents):
    chunks = []

    for doc in documents:
        _chunks = build_chunks_from_text(doc["text"])
        for i, chunk in enumerate(_chunks):
            chunks.append({
                "chunk_id": i,
                "text": chunk,
                "src": doc
            })

    return chunks