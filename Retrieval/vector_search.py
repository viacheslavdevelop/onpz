import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from chunks_io import load_chunks

MODEL_NAME = "sergeyzh/rubert-tiny-turbo"
INDEX_PATH = Path(__file__).parent.parent / "Chunks" / "faiss_index.bin"
IDS_PATH = Path(__file__).parent.parent / "Chunks" / "faiss_chunk_ids.json"


def load_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def encode(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vectors.astype("float32")


def build_index(vectors: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def save_index(index: faiss.Index, chunk_ids: list[str]) -> None:
    faiss.write_index(index, str(INDEX_PATH))
    IDS_PATH.write_text(json.dumps(chunk_ids, ensure_ascii=False), encoding="utf-8")


def load_index() -> tuple[faiss.Index, list[str]]:
    index = faiss.read_index(str(INDEX_PATH))
    chunk_ids = json.loads(IDS_PATH.read_text(encoding="utf-8"))
    return index, chunk_ids


def search(model: SentenceTransformer, index: faiss.Index, chunk_ids: list[str], query: str, top_k: int = 5) -> list[tuple[str, float]]:
    query_vector = encode(model, [query])
    scores, positions = index.search(query_vector, top_k)
    return [(chunk_ids[pos], float(score)) for pos, score in zip(positions[0], scores[0])]


def build_and_save() -> None:
    chunks = load_chunks()
    chunk_ids = [c["chunk_id"] for c in chunks]
    model = load_model()
    vectors = encode(model, [c["text"] for c in chunks])
    index = build_index(vectors)
    save_index(index, chunk_ids)


if __name__ == "__main__":
    build_and_save()
