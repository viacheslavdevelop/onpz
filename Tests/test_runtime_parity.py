"""Интеграционный тест: кэш runtime даёт тот же результат, что hybrid_search."""

import unittest
from pathlib import Path

from Tests import _bootstrap  # noqa: F401

import runtime
from hybrid_retrieval import hybrid_search

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_FILES = (
    BASE_DIR / "Chunks" / "bm25_index.pkl",
    BASE_DIR / "Chunks" / "faiss_index.bin",
    BASE_DIR / "Chunks" / "faiss_chunk_ids.json",
    BASE_DIR / "Chunks" / "chunks_v2_by_fact.jsonl",
)

QUERIES = (
    "Какое давление поддерживается в отбензинивающей колонне К-1?",
    "Опишите последовательность обработки сырой нефти от поступления на установку.",
    "Какие параметры у атмосферно-вакуумной трубчатки?",
)


@unittest.skipUnless(
    all(path.exists() for path in INDEX_FILES),
    "Поисковые индексы отсутствуют в Chunks/",
)
class RuntimeParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.retrieval_runtime = runtime.get_runtime()

    def test_runtime_is_cached(self) -> None:
        self.assertIs(runtime.get_runtime(), self.retrieval_runtime)

    def test_search_matches_hybrid_search(self) -> None:
        for query in QUERIES:
            with self.subTest(query=query):
                expected = hybrid_search(query, top_k=5)
                actual = self.retrieval_runtime.search(query, top_k=5)

                self.assertEqual([chunk_id for chunk_id, _ in actual], [chunk_id for chunk_id, _ in expected])
                for (_, actual_score), (_, expected_score) in zip(actual, expected):
                    self.assertAlmostEqual(actual_score, expected_score, delta=1e-6)

    def test_invalid_parameters_raise(self) -> None:
        with self.assertRaises(ValueError):
            self.retrieval_runtime.search("вопрос", top_k=0)
        with self.assertRaises(ValueError):
            self.retrieval_runtime.search("вопрос", alpha=1.5)
        with self.assertRaises(ValueError):
            self.retrieval_runtime.search("вопрос", method="unknown")

    def test_alternate_methods_return_top_k(self) -> None:
        for method in ("bm25", "vector"):
            with self.subTest(method=method):
                results = self.retrieval_runtime.search("давление в колонне", top_k=3, method=method)
                self.assertEqual(len(results), 3)


if __name__ == "__main__":
    unittest.main()
