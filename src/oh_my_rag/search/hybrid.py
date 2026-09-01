import sys
from concurrent.futures import ThreadPoolExecutor

from oh_my_rag.config import (
    CANDIDATE_K,
    RRF_WEIGHT_ALTERNATE,
    RRF_WEIGHT_FTS,
    RRF_WEIGHT_VECTOR,
    SEARCH_TIMEOUT,
    TOP_K,
)
from oh_my_rag.search.fts import FtsIndex
from oh_my_rag.search.fusion import RankedList, reciprocal_rank_fusion
from oh_my_rag.search.types import FusedHit, SearchHit
from oh_my_rag.search.vector import VectorSearch


class HybridSearch:
    """Runs the dense and lexical retrievers concurrently and fuses the result.

    The two branches genuinely overlap rather than time-slice: SQLite, Chroma's
    HNSW and torch all release the GIL inside their own C/C++ work, so the
    lexical search is close to free in wall clock.

    Each branch is settled independently. If FTS5 fails you get pure vector
    search — today's behaviour — and if Chroma fails you still get keyword
    results. Neither branch can take the other down.
    """

    def __init__(
        self,
        vector: VectorSearch | None = None,
        fts: FtsIndex | None = None,
        top_k: int = TOP_K,
        candidate_k: int = CANDIDATE_K,
    ):
        self.vector = vector if vector is not None else VectorSearch()
        self.fts = fts if fts is not None else FtsIndex()
        self.top_k = top_k
        self.candidate_k = candidate_k
        # Long-lived: pool threads keep their SQLite connections between
        # queries, and per-query thread creation is pure overhead.
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ohmyrag-search")

    def close(self) -> None:
        self._pool.shutdown(wait=False)

    def _settle(self, future, branch: str, default):
        # Diagnostics go to stderr, never stdout: under the MCP stdio server
        # stdout IS the JSON-RPC channel and a stray line corrupts the stream.
        try:
            return future.result(timeout=SEARCH_TIMEOUT)
        except Exception as exc:
            print(f"[search] {branch} branch failed, continuing without it: {exc}", file=sys.stderr)
            return default

    def search(self, expanded) -> list[FusedHit]:
        queries = expanded.vector_queries()
        terms = expanded.fts_terms()

        f_vec = self._pool.submit(self.vector.search, queries, self.candidate_k)
        f_fts = self._pool.submit(self.fts.search, terms, self.candidate_k)

        vector_lists: list[list[SearchHit]] = self._settle(f_vec, "vector", [])
        fts_hits: list[SearchHit] = self._settle(f_fts, "fts", [])

        ranked = []
        for i, hits in enumerate(vector_lists):
            # The user's own words outrank the model's paraphrases of them.
            first = i == 0
            ranked.append(
                RankedList(
                    name="vector" if first else f"vector_alt{i}",
                    hits=hits,
                    weight=RRF_WEIGHT_VECTOR if first else RRF_WEIGHT_ALTERNATE,
                )
            )
        ranked.append(RankedList(name="fts", hits=fts_hits, weight=RRF_WEIGHT_FTS))

        return reciprocal_rank_fusion(ranked, top_k=self.top_k)
