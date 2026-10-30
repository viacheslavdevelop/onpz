import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from chunks_io import load_chunks

INDEX_PATH = Path(__file__).parent.parent / "Chunks" / "bm25_index.pkl"

TOKEN_RE = re.compile(r"[a-zа-яё0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def build_index(chunks: list[dict]) -> BM25Okapi:
    tokenized = [tokenize(c["text"]) for c in chunks]
    return BM25Okapi(tokenized)


def save_index(index: BM25Okapi, chunk_ids: list[str], path: Path = INDEX_PATH) -> None:
    with path.open("wb") as f:
        pickle.dump({"index": index, "chunk_ids": chunk_ids}, f)


def load_index(path: Path = INDEX_PATH) -> tuple[BM25Okapi, list[str]]:
    with path.open("rb") as f:
        data = pickle.load(f)
    return data["index"], data["chunk_ids"]


def search(index: BM25Okapi, chunk_ids: list[str], query: str, top_k: int = 5) -> list[tuple[str, float]]:
    scores = index.get_scores(tokenize(query))
    ranked = sorted(zip(chunk_ids, scores), key=lambda pair: pair[1], reverse=True)
    return ranked[:top_k]


def build_and_save() -> None:
    chunks = load_chunks()
    chunk_ids = [c["chunk_id"] for c in chunks]
    index = build_index(chunks)
    save_index(index, chunk_ids)


if __name__ == "__main__":
    build_and_save()
