from collections.abc import Iterator
from dataclasses import dataclass

from oh_my_rag.config import (
    CHROMADB_PATH,
    EXPANSION_ENABLED,
    EXPANSION_MODEL,
    LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_REASONING_EFFORT,
    TOP_K,
)
from oh_my_rag.llm_client import chat, chat_stream
from oh_my_rag.observability import Trace
from oh_my_rag.query_expansion import ExpandedQuery, expand
from oh_my_rag.search.fts import FtsIndex
from oh_my_rag.search.hybrid import HybridSearch
from oh_my_rag.search.types import FusedHit
from oh_my_rag.search.vector import VectorSearch
from oh_my_rag.seed_database import COLLECTION_NAME


@dataclass
class RagResult:
    """Everything one question produced, for --debug and MCP responses."""

    query: str
    expanded: ExpandedQuery
    contexts: list[FusedHit]
    prompt: str
    answer: str
    trace: Trace | None = None


class Rag:
    def __init__(
        self,
        chroma_path: str = CHROMADB_PATH,
        collection_name: str = COLLECTION_NAME,
        llm_base_url: str = LLM_BASE_URL,
        llm_model: str = LLM_MODEL,
        top_k: int = TOP_K,
        expand_queries: bool = EXPANSION_ENABLED,
        expansion_model: str = EXPANSION_MODEL,
    ):
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        # Held separately from llm_model on purpose: the two calls have different
        # jobs and can run on different models. Passing llm_model= used to change
        # only the answer path while expansion silently kept using the config
        # constant, which is the kind of surprise this attribute removes.
        self.expansion_model = expansion_model
        self.top_k = top_k
        self.expand_queries = expand_queries

        vector = VectorSearch(chroma_path=chroma_path, collection_name=collection_name)
        # Touched eagerly so an unseeded clone fails here, with a clear error,
        # rather than on the first question.
        vector.collection

        self.search = HybridSearch(vector=vector, fts=FtsIndex(), top_k=top_k)

    def expand_query(
        self,
        query: str,
        expand_queries: bool | None = None,
        trace: Trace | None = None,
    ) -> ExpandedQuery:
        enabled = self.expand_queries if expand_queries is None else expand_queries
        if trace is None:
            return expand(
                query,
                base_url=self.llm_base_url,
                model=self.expansion_model,
                enabled=enabled,
            )

        with trace.stage("expansion"):
            expanded = expand(
                query,
                base_url=self.llm_base_url,
                model=self.expansion_model,
                enabled=enabled,
            )
        trace.expansion_reason = expanded.reason
        return expanded

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        expand_queries: bool | None = None,
    ) -> list[FusedHit]:
        return self.search_expanded(
            self.expand_query(query, expand_queries), top_k=top_k
        )

    def search_expanded(
        self,
        expanded: ExpandedQuery,
        top_k: int | None = None,
        trace: Trace | None = None,
    ) -> list[FusedHit]:
        if top_k is None:
            return self.search.search(expanded, trace=trace)

        original = self.search.top_k
        self.search.top_k = top_k
        try:
            return self.search.search(expanded, trace=trace)
        finally:
            self.search.top_k = original

    def build_prompt(self, query: str, contexts: list[FusedHit]) -> str:
        # Delimited blocks: concatenating raw chunks ran the last sentence of one
        # document straight into the filename of the next.
        context_text = "\n".join(
            f'<doc src="{c.src}">\n{c.text}\n</doc>' for c in contexts
        )
        return f"""
<role>You are a secure, factual enterprise intelligence assistant. Answer the user prompt utilizing ONLY the provided ground-truth context</role>
<instructions>Be precise. Give short answers. If in doubt - say 'Not found'</instructions>
<ground-truth-context>
{context_text}
</ground-truth-context>
<query>{query}</query>
"""

    def ask_llm(self, prompt: str) -> str:
        return chat(
            [{"role": "user", "content": prompt}],
            base_url=self.llm_base_url,
            model=self.llm_model,
            max_tokens=LLM_MAX_TOKENS,
            reasoning_effort=LLM_REASONING_EFFORT,
            kind="answer",
        )

    def prepare(
        self,
        query: str,
        expand_queries: bool | None = None,
        trace: bool = False,
    ) -> RagResult:
        """Everything up to generation. `answer` is empty.

        Split out so a caller can show retrieval *before* streaming the answer,
        which is the whole point of an observable pipeline.
        """
        record = Trace(query=query) if trace else None
        expanded = self.expand_query(query, expand_queries, trace=record)
        contexts = self.search_expanded(expanded, trace=record)
        prompt = self.build_prompt(query, contexts)
        if record is not None:
            record.prompt_chars = len(prompt)
        return RagResult(
            query=query,
            expanded=expanded,
            contexts=contexts,
            prompt=prompt,
            answer="",
            trace=record,
        )

    def answer(self, query: str, expand_queries: bool | None = None) -> str:
        return self.answer_verbose(query, expand_queries=expand_queries).answer

    def answer_verbose(
        self,
        query: str,
        expand_queries: bool | None = None,
        trace: bool = False,
    ) -> RagResult:
        result = self.prepare(query, expand_queries, trace=trace)
        if result.trace is not None:
            with result.trace.stage("generation"):
                result.answer = self.ask_llm(result.prompt)
            result.trace.answer_chars = len(result.answer)
        else:
            result.answer = self.ask_llm(result.prompt)
        return result

    def stream_answer(self, prompt: str) -> Iterator[str]:
        """Yield the answer as it is generated."""
        return chat_stream(
            [{"role": "user", "content": prompt}],
            base_url=self.llm_base_url,
            model=self.llm_model,
            max_tokens=LLM_MAX_TOKENS,
            reasoning_effort=LLM_REASONING_EFFORT,
            # With reasoning off there is no scratchpad to guard against, so the
            # first token can go out immediately. Otherwise hold a window back
            # so a leading orphan </think> can still be caught.
            probation=0 if LLM_REASONING_EFFORT == "none" else 400,
            kind="answer",
        )

    def answer_stream(
        self, query: str, expand_queries: bool | None = None
    ) -> Iterator[str]:
        yield from self.stream_answer(self.prepare(query, expand_queries).prompt)
