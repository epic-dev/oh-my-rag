import os

from oh_my_rag.config import EMBEDDING_MODEL

_model = None


def get_embedding_model():
    """Return a process-wide cached SentenceTransformer, loaded strictly from the
    local Hugging Face cache (no network call, no update check on every use).

    If the model has never been downloaded, run this module directly first:
        uv run python -m oh_my_rag.embeddings_model_loader
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        try:
            _model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
        except OSError as exc:
            raise RuntimeError(
                f"Embedding model '{EMBEDDING_MODEL}' is not cached locally. "
                "Run `uv run python -m oh_my_rag.embeddings_model_loader` to download it once."
            ) from exc
    return _model


def sync_model():
    """Check the Hugging Face Hub for updates to EMBEDDING_MODEL and download
    them if the local cache is missing or stale."""
    os.environ.pop("HF_HUB_OFFLINE", None)
    from sentence_transformers import SentenceTransformer

    print(f"Checking '{EMBEDDING_MODEL}' for updates...")
    SentenceTransformer(EMBEDDING_MODEL)
    print("Local model cache is up to date.")


if __name__ == "__main__":
    sync_model()
