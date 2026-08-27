import re

import chromadb
import requests
import json

from oh_my_rag.config import CHROMADB_PATH, LLM_BASE_URL, LLM_MODEL
from oh_my_rag.embeddings_model_loader import get_embedding_model
from oh_my_rag.seed_database import COLLECTION_NAME


class Rag:
    def __init__(
        self,
        chroma_path: str = CHROMADB_PATH,
        collection_name: str = COLLECTION_NAME,
        llm_base_url: str = LLM_BASE_URL,
        llm_model: str = LLM_MODEL,
    ):
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model

        self.embedding_model = get_embedding_model()
        client = chromadb.PersistentClient(path=chroma_path)
        self.collection = client.get_collection(name=collection_name)

    def retrieve(self, query: str, n_results: int = 3) -> list[dict]:
        query_embeddings = self.embedding_model.encode(
            [f"Represent this sentence for searching relevant passages: {query}"]
        )
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
        )

        documents = results.get("documents") or [[]]
        metadatas = results.get("metadatas") or [[]]
        return [
            {"src": (meta or {}).get("src", ""), "text": doc}
            for doc, meta in zip(documents[0] or [], metadatas[0] or [])
        ]

    def build_prompt(self, query: str, contexts: list[dict]) -> str:
        context_text = "".join(
            f"{c['src']}: {c['text']}"
            for c in contexts
        )
        return f"""
<role>You are a secure, factual enterprise intelligence assistant. Answer the user prompt utilizing ONLY the provided ground-truth context</role>
<instructions>Be precise. Give short answers. If in doubt - say 'Not found'</instructions>
<ground-truth-context>{context_text}</ground-truth-context>
<query>{query}</query>
"""

    def ask_llm(self, prompt: str) -> str:
        response = requests.post(
            self.llm_base_url,
            json={
                "model": self.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
        )
        response.raise_for_status()

        content = response.json()["choices"][0]["message"]["content"]
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        for token in ("</s>", "<|im_end|>"):
            content = content.replace(token, "")
        return content.strip()

    def answer(self, query: str) -> str:
        contexts = self.retrieve(query)
        prompt = self.build_prompt(query, contexts)
        return self.ask_llm(prompt)
