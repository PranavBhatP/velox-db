"""Lazy-loaded sentence-transformers embedder."""

from functools import lru_cache

MODEL_NAME = "all-MiniLM-L6-v2"
EXPECTED_DIM = 384


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def embed(text: str) -> list[float]:
    model = _get_model()
    vector = model.encode(text, convert_to_numpy=True)
    return vector.tolist()


def embedding_dim() -> int:
    return EXPECTED_DIM
