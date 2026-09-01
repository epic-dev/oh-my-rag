from dataclasses import dataclass, field

from oh_my_rag.config import RRF_K, TOP_K
from oh_my_rag.search.types import FusedHit, SearchHit


@dataclass(frozen=True)
class RankedList:
    """One retriever's output, with the weight its votes carry in fusion."""

    name: str
    hits: list[SearchHit] = field(default_factory=list)
    weight: float = 1.0


def reciprocal_rank_fusion(
    ranked_lists: list[RankedList],
    *,
    k: int = RRF_K,
    top_k: int = TOP_K,
) -> list[FusedHit]:
    """Fuse ranked lists by reciprocal rank: score(d) = Σ wᵢ / (k + rankᵢ(d)).

    Only ranks are read. Chroma distances and BM25 scores are on incomparable
    scales, and normalizing them into one number is guesswork that shifts with
    every corpus — dropping them sidesteps that entirely, which matters most
    when the dense model is small and its scores are not very trustworthy.

    Chunks are deduped on `chunk_uid`, and ties break on uid so the ordering is
    stable across runs rather than dictionary-order luck.
    """
    scores: dict[str, float] = {}
    sources: dict[str, dict[str, int]] = {}
    seen: dict[str, SearchHit] = {}

    for ranked in ranked_lists:
        for hit in ranked.hits:
            scores[hit.chunk_uid] = scores.get(hit.chunk_uid, 0.0) + ranked.weight / (
                k + hit.rank
            )
            sources.setdefault(hit.chunk_uid, {})[ranked.name] = hit.rank
            # First writer wins: every list carries the same text for a uid.
            seen.setdefault(hit.chunk_uid, hit)

    order = sorted(scores, key=lambda uid: (-scores[uid], uid))

    return [
        FusedHit(
            chunk_uid=uid,
            src=seen[uid].src,
            text=seen[uid].text,
            score=scores[uid],
            sources=sources[uid],
        )
        for uid in order[:top_k]
    ]
