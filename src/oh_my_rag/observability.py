import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from oh_my_rag.search.types import FusedHit, SearchHit


@dataclass
class BranchTrace:
    """One retriever's contribution to a single query."""

    name: str
    weight: float
    ms: float = 0.0
    hits: list[SearchHit] = field(default_factory=list)
    query: str | None = None      # the phrasing this dense list was built from
    detail: str | None = None     # the FTS MATCH expression, or a failure reason
    candidates: int | None = None  # rows matching before LIMIT (lexical only)
    failed: bool = False


@dataclass
class Trace:
    """A structured record of one question's journey through the pipeline.

    Deliberately data, not printing: the CLI renders it, MCP could return it,
    and a web frontend could chart it. Nothing here formats anything.
    """

    query: str
    timings: dict[str, float] = field(default_factory=dict)
    expansion_raw: str | None = None
    expansion_reason: str | None = None
    branches: list[BranchTrace] = field(default_factory=list)
    fused: list[FusedHit] = field(default_factory=list)
    # The top-K as fusion ordered it, before the cross-encoder reordered it.
    rerank_before: list[str] = field(default_factory=list)
    prompt_chars: int = 0
    answer_chars: int = 0
    warnings: list[str] = field(default_factory=list)

    @contextmanager
    def stage(self, name: str):
        """Time a stage. Nested stages are fine; each keeps its own key."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = (time.perf_counter() - start) * 1000

    def record(self, name: str, ms: float) -> None:
        self.timings[name] = ms

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    @property
    def total_ms(self) -> float:
        # Retrieval branches run concurrently, so summing every key would
        # double-count. Only top-level stages are summed.
        return sum(
            ms for key, ms in self.timings.items() if not key.startswith("  ")
        )
