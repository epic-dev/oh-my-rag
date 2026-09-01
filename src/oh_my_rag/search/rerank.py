import sys
from dataclasses import replace

from oh_my_rag.config import (
    RERANK_BATCH_SIZE,
    RERANK_ENABLED,
    RERANK_MAX_TOKENS,
    RERANK_MODEL,
)
from oh_my_rag.search.types import FusedHit


class Reranker:
    """Cross-encoder rescoring of the fused candidate list.

    Retrieval compares a query vector against passage vectors computed
    independently, so the model never sees the two together. A cross-encoder
    does, which is why it reorders usefully — and why it is too slow to run over
    the whole corpus and belongs at the end, over a short candidate list.

    Loaded lazily and strictly from the local cache, like the embedding model:
    an MCP host spawning this over stdio must not wait for a download.
    """

    def __init__(
        self,
        model_name: str = RERANK_MODEL,
        enabled: bool = RERANK_ENABLED,
        batch_size: int = RERANK_BATCH_SIZE,
        max_tokens: int = RERANK_MAX_TOKENS,
    ):
        self.model_name = model_name
        self.enabled = enabled
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            try:
                self._model = CrossEncoder(self.model_name, local_files_only=True)
                # Left at its default this model would allocate a 40k-token
                # window per passage, which is the whole cost of reranking.
                self._model.max_seq_length = self.max_tokens
            except OSError as exc:
                raise RuntimeError(
                    f"Reranker '{self.model_name}' is not cached locally. Run "
                    "`uv run python -m oh_my_rag.embeddings_model_loader` to "
                    "download it once, or set RERANK_ENABLED = False."
                ) from exc
        return self._model

    def rerank(self, query: str, hits: list[FusedHit], top_k: int) -> list[FusedHit]:
        """Rescore and reorder, returning the best `top_k`.

        Never raises: a reranker failure leaves the RRF ordering intact, which
        is the same degradation contract the two retrievers already follow.
        """
        if not self.enabled or not hits:
            return hits[:top_k]
        if len(hits) == 1:
            return hits[:top_k]

        try:
            # The model's own task instruction is applied automatically from its
            # config (default_prompt_name = "query").
            scores = self.model.predict(
                [(query, hit.text) for hit in hits],
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        except Exception as exc:
            print(f"[rerank] failed, keeping the fused order: {exc}", file=sys.stderr)
            return hits[:top_k]

        rescored = [
            replace(hit, rerank_score=float(score))
            for hit, score in zip(hits, scores)
        ]
        # Tie-break on uid so the ordering is stable across runs, matching fusion.
        rescored.sort(key=lambda h: (-h.rerank_score, h.chunk_uid))
        return rescored[:top_k]
