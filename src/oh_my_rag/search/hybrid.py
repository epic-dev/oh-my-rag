import sys
import time
from concurrent.futures import ThreadPoolExecutor

from oh_my_rag.config import (
    CANDIDATE_K,
    RERANK_CANDIDATES,
    RERANK_ENABLED,
    RRF_WEIGHT_ALTERNATE,
    RRF_WEIGHT_FTS,
    RRF_WEIGHT_VECTOR,
    SEARCH_TIMEOUT,
    TOP_K,
)
from oh_my_rag.search.fts import FtsIndex
from oh_my_rag.search.rerank import Reranker
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
        reranker: Reranker | None = None,
    ):
        self.vector = vector if vector is not None else VectorSearch()
        self.fts = fts if fts is not None else FtsIndex()
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.reranker = reranker if reranker is not None else Reranker()
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

    def search(self, expanded, trace=None) -> list[FusedHit]:
        queries = expanded.vector_queries()
        terms = expanded.fts_terms()

        lex_branch = None
        if trace is not None:
            from oh_my_rag.observability import BranchTrace

            lex_branch = BranchTrace(name="fts", weight=RRF_WEIGHT_FTS)

        # Passed only while tracing, so a retriever implementing the minimal
        # search(terms, limit) contract still works.
        lex_kwargs = {"branch": lex_branch} if lex_branch is not None else {}

        started = time.perf_counter()
        f_vec = self._pool.submit(self.vector.search, queries, self.candidate_k)
        f_fts = self._pool.submit(self.fts.search, terms, self.candidate_k, **lex_kwargs)

        vec_started = time.perf_counter()
        vector_lists: list[list[SearchHit]] = self._settle(f_vec, "vector", [])
        vec_ms = (time.perf_counter() - vec_started) * 1000
        fts_hits: list[SearchHit] = self._settle(f_fts, "fts", [])
        wall_ms = (time.perf_counter() - started) * 1000

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

        # With reranking on, fusion hands over a deeper candidate list and the
        # cross-encoder makes the final cut. Its job is to reorder what
        # retrieval found, so it never sees more than retrieval returned.
        reranking = self.reranker is not None and self.reranker.enabled
        fuse_to = max(self.top_k, RERANK_CANDIDATES) if reranking else self.top_k
        fused = reciprocal_rank_fusion(ranked, top_k=fuse_to)

        if reranking:
            import time as _time

            fused_order = [h.chunk_uid for h in fused[: self.top_k]]
            started_rerank = _time.perf_counter()
            fused = self.reranker.rerank(expanded.original, fused, self.top_k)
            rerank_ms = (_time.perf_counter() - started_rerank) * 1000
            if trace is not None:
                trace.record("rerank", rerank_ms)
                trace.rerank_before = fused_order
        else:
            fused = fused[: self.top_k]

        if trace is not None:
            from oh_my_rag.observability import BranchTrace

            for i, rl in enumerate(ranked[:-1]):
                trace.branches.append(
                    BranchTrace(
                        name=rl.name,
                        weight=rl.weight,
                        # One batched call covers every variant, so the elapsed
                        # time belongs to the branch, not to any one phrasing.
                        ms=vec_ms if i == 0 else 0.0,
                        hits=rl.hits,
                        query=queries[i] if i < len(queries) else None,
                        failed=not rl.hits and not vector_lists,
                    )
                )
            if lex_branch is not None:
                lex_branch.hits = fts_hits
                trace.branches.append(lex_branch)
            trace.record("  retrieval (both branches, concurrent)", wall_ms)
            trace.fused = fused
            if not vector_lists:
                trace.warn("dense branch returned nothing")
            if not fts_hits:
                trace.warn("lexical branch returned nothing")

        return fused
