"""Кэш ресурсов поиска: модели, индексы и чанки загружаются один раз на процесс.

Используется веб-интерфейсом и сервисным слоем вместо прямых вызовов
``hybrid_search``, чтобы не перечитывать индексы и не загружать модель
эмбеддингов при каждом вопросе.
"""

from dataclasses import dataclass
from functools import lru_cache

import bm25_search
import vector_search
from chunks_io import load_chunks
from hybrid_retrieval import (
    ALPHA_DEFAULT,
    CANDIDATE_POOL_SIZE,
    build_list_groups,
    combine_scores,
)

ALLOWED_METHODS = ("hybrid", "bm25", "vector")


@dataclass
class RetrievalRuntime:
    """Загруженные один раз ресурсы поиска и методы поиска поверх них."""

    chunks: list[dict]
    chunk_lookup: dict[str, dict]
    list_groups: dict
    bm25_index: object
    bm25_chunk_ids: list[str]
    vector_model: object
    vector_index: object
    vector_chunk_ids: list[str]

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        alpha: float = ALPHA_DEFAULT,
        method: str = "hybrid",
        pool_size: int = CANDIDATE_POOL_SIZE,
    ) -> list[tuple[str, float]]:
        """Найти top_k фрагментов выбранным методом без повторной загрузки ресурсов."""
        if top_k < 1:
            raise ValueError("top_k должен быть не меньше 1.")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha должен быть в диапазоне от 0.0 до 1.0.")
        if method not in ALLOWED_METHODS:
            raise ValueError(
                f"Неизвестный метод поиска: {method!r}. "
                f"Допустимые значения: {', '.join(ALLOWED_METHODS)}."
            )
        if pool_size < 1:
            raise ValueError("pool_size должен быть не меньше 1.")

        candidate_pool = max(pool_size, top_k)
        bm25_raw = dict(
            bm25_search.search(self.bm25_index, self.bm25_chunk_ids, query, candidate_pool)
        )
        vector_raw = dict(
            vector_search.search(
                self.vector_model, self.vector_index, self.vector_chunk_ids, query, candidate_pool
            )
        )

        if method == "bm25":
            return sorted(bm25_raw.items(), key=lambda pair: pair[1], reverse=True)[:top_k]
        if method == "vector":
            return sorted(vector_raw.items(), key=lambda pair: pair[1], reverse=True)[:top_k]
        return combine_scores(bm25_raw, vector_raw, alpha=alpha, top_k=top_k)


def load_runtime() -> RetrievalRuntime:
    """Загрузить чанки, индексы и модель эмбеддингов с диска."""
    chunks = load_chunks()
    chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}
    list_groups = build_list_groups(chunks)
    bm25_index, bm25_chunk_ids = bm25_search.load_index()
    vector_model = vector_search.load_model()
    vector_index, vector_chunk_ids = vector_search.load_index()

    return RetrievalRuntime(
        chunks=chunks,
        chunk_lookup=chunk_lookup,
        list_groups=list_groups,
        bm25_index=bm25_index,
        bm25_chunk_ids=bm25_chunk_ids,
        vector_model=vector_model,
        vector_index=vector_index,
        vector_chunk_ids=vector_chunk_ids,
    )


@lru_cache(maxsize=1)
def get_runtime() -> RetrievalRuntime:
    """Вернуть общий на процесс экземпляр ресурсов поиска."""
    return load_runtime()
