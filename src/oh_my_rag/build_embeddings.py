import numpy as np

from oh_my_rag.embeddings_model_loader import get_embedding_model

def chunks_to_embeddings(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = get_embedding_model().encode(texts)

    return np.array(embeddings)