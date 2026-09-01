from dataclasses import dataclass


@dataclass(frozen=True)
class SearchHit:
    """One result from a single retriever.

    `rank` is 1-based *within that retriever's own list* and is the only field
    fusion reads. `score` is the retriever's native number — Chroma distance on
    the dense side, negative BM25 on the lexical side — kept for debugging and
    deliberately not comparable across retrievers.
    """

    chunk_uid: str
    src: str
    text: str
    score: float
    rank: int


@dataclass(frozen=True)
class FusedHit:
    """One chunk after fusion, with a record of which retrievers found it.

    `sources` maps a retriever's name to the 1-based rank it gave this chunk,
    which is what makes a fused result explainable in `--debug` output.
    """

    chunk_uid: str
    src: str
    text: str
    score: float
    sources: dict[str, int]
