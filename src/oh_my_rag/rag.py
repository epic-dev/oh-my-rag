from dataclasses import dataclass

from oh_my_rag.config import (
    CHROMADB_PATH,
    EXPANSION_ENABLED,
    LLM_BASE_URL,
    LLM_MODEL,
    TOP_K,
)
from oh_my_rag.llm_client import chat
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


class Rag:
    def __init__(
        self,
        chroma_path: str = CHROMADB_PATH,
        collection_name: str = COLLECTION_NAME,
        llm_base_url: str = LLM_BASE_URL,
        llm_model: str = LLM_MODEL,
        top_k: int = TOP_K,
        expand_queries: bool = EXPANSION_ENABLED,
    ):
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self.top_k = top_k
        self.expand_queries = expand_queries

        vector = VectorSearch(chroma_path=chroma_path, collection_name=collection_name)
        # Touched eagerly so an unseeded clone fails here, with a clear error,
        # rather than on the first question.
        vector.collection

        self.search = HybridSearch(vector=vector, fts=FtsIndex(), top_k=top_k)

    def expand_query(self, query: str, expand_queries: bool | None = None) -> ExpandedQuery:
        enabled = self.expand_queries if expand_queries is None else expand_queries
        return expand(
            query,
            base_url=self.llm_base_url,
            enabled=enabled,
        )

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
        self, expanded: ExpandedQuery, top_k: int | None = None
    ) -> list[FusedHit]:
        if top_k is None:
            return self.search.search(expanded)

        original = self.search.top_k
        self.search.top_k = top_k
        try:
            return self.search.search(expanded)
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
        )

    def answer(self, query: str, expand_queries: bool | None = None) -> str:
        return self.answer_verbose(query, expand_queries=expand_queries).answer

    def answer_verbose(
        self, query: str, expand_queries: bool | None = None
    ) -> RagResult:
        expanded = self.expand_query(query, expand_queries)
        contexts = self.search_expanded(expanded)
        prompt = self.build_prompt(query, contexts)
        return RagResult(
            query=query,
            expanded=expanded,
            contexts=contexts,
            prompt=prompt,
            answer=self.ask_llm(prompt),
        )
