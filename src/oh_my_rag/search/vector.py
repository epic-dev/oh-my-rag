import chromadb

from oh_my_rag.config import CANDIDATE_K, CHROMADB_PATH, EMBEDDING_QUERY_PREFIX
from oh_my_rag.embeddings_model_loader import get_embedding_model
from oh_my_rag.search.types import SearchHit


class VectorSearch:
    """The dense half of hybrid search.

    Several phrasings of one question are embedded in a single batch and sent
    as a single Chroma call, so asking three ways costs one round trip.
    """

    def __init__(
        self,
        chroma_path: str = CHROMADB_PATH,
        collection_name: str | None = None,
        collection=None,
    ):
        if collection_name is None:
            from oh_my_rag.seed_database import COLLECTION_NAME

            collection_name = COLLECTION_NAME
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self._collection = collection

    @property
    def collection(self):
        # Lazy so that constructing a Rag (or an MCP server) does not open the
        # store until something is actually searched.
        if self._collection is None:
            client = chromadb.PersistentClient(path=self.chroma_path)
            self._collection = client.get_collection(name=self.collection_name)
        return self._collection

    def search(self, queries: list[str], limit: int = CANDIDATE_K) -> list[list[SearchHit]]:
        """Return one ranked list per query, in the order the queries were given."""
        if not queries:
            return []

        # Same model and normalization as the indexing side; l2 on normalized
        # vectors ranks identically to cosine, which is what both bge-* and
        # Qwen3-Embedding are trained for.
        # The prefix is query-side only — documents are embedded raw.
        embeddings = get_embedding_model().encode(
            [EMBEDDING_QUERY_PREFIX + q for q in queries],
            normalize_embeddings=True,
        )

        results = self.collection.query(
            query_embeddings=embeddings,
            n_results=limit,
        )

        ids = results.get("ids") or []
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []
        distances = results.get("distances") or []

        ranked: list[list[SearchHit]] = []
        for i in range(len(queries)):
            row_ids = ids[i] if i < len(ids) else []
            row_docs = documents[i] if i < len(documents) else []
            row_metas = metadatas[i] if i < len(metadatas) else []
            row_dists = distances[i] if i < len(distances) else []

            hits = []
            for rank, uid in enumerate(row_ids, start=1):
                j = rank - 1
                meta = row_metas[j] if j < len(row_metas) else {}
                hits.append(
                    SearchHit(
                        chunk_uid=uid,
                        src=(meta or {}).get("src", ""),
                        text=row_docs[j] if j < len(row_docs) else "",
                        score=float(row_dists[j]) if j < len(row_dists) else 0.0,
                        rank=rank,
                    )
                )
            ranked.append(hits)

        return ranked
