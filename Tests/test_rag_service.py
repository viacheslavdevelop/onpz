"""Тесты сервисного слоя RAG без сети и без записи артефактов."""

import unittest
from unittest.mock import patch

from Tests import _bootstrap  # noqa: F401

import rag_service
from query_expansion import expand_query


def make_chunk(
    chunk_id: str,
    text: str,
    *,
    chunk_type: str = "text",
    section: str = "3. Режим работы колонн К-1 и К-2",
    subsection: str | None = "3.1. Давление",
    list_intro: str | None = None,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "document": "03_reglament_at_rezhim_k1_k2.md",
        "section": section,
        "subsection": subsection,
        "type": chunk_type,
        "list_intro": list_intro,
        "text": text,
    }


class StubRuntime:
    """Минимальная замена RetrievalRuntime для тестов."""

    def __init__(self, chunks: list[dict], scores: dict[str, float] | None = None) -> None:
        self.chunks = chunks
        self.chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}
        self.list_groups: dict = {}
        self.scores = scores or {chunk["chunk_id"]: 1.0 for chunk in chunks}
        self.last_query = None

    def search(self, query: str, *, top_k: int = 5, alpha: float = 0.7, method: str = "hybrid"):
        self.last_query = query
        results = [
            (chunk["chunk_id"], self.scores[chunk["chunk_id"]])
            for chunk in self.chunks
            if chunk["chunk_id"] in self.scores
        ]
        return results[:top_k]


def fake_chat(text: str) -> dict:
    return {
        "text": text,
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        "latency_s": 0.5,
    }


class RagServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            make_chunk("c1", "Давление в колонне К-1 составляет 0,15 МПа."),
            make_chunk(
                "c2",
                "Температура верха колонны 120 °C, мощность $8{,}4\\text{ млн т/год}$.",
            ),
        ]
        self.runtime = StubRuntime(self.chunks)

    def _ask(self, text: str, **kwargs) -> rag_service.RAGAnswer:
        with patch.object(rag_service, "get_runtime", return_value=self.runtime), patch.object(
            rag_service.llm_client, "chat", return_value=fake_chat(text)
        ):
            return rag_service.ask("Какое давление в колонне К-1?", **kwargs)

    def test_ask_returns_sources_matching_context(self) -> None:
        result = self._ask("Давление 0,15 МПа [1].")

        self.assertEqual([source.number for source in result.sources], [1, 2])
        self.assertIn("[1]", result.answer)
        self.assertIn("c1", result.context)
        self.assertIn("c2", result.context)
        self.assertFalse(result.context_truncated)
        self.assertEqual(result.omitted_sources, 0)
        self.assertEqual(result.usage, {"prompt_tokens": 100, "completion_tokens": 20})
        self.assertEqual(result.settings["top_k"], 5)

    def test_citation_outside_context_is_removed(self) -> None:
        result = self._ask("Давление 0,15 МПа [1] [9].")
        self.assertIn("[1]", result.answer)
        self.assertNotIn("[9]", result.answer)

    def test_refusal_is_detected(self) -> None:
        result = self._ask("Информация не найдена в регламенте.")
        self.assertTrue(result.refused)

    def test_missing_text_is_not_refusal(self) -> None:
        result = self._ask("")
        self.assertEqual(result.answer, "")
        self.assertFalse(result.refused)

    def test_retrieve_does_not_call_llm(self) -> None:
        with patch.object(rag_service, "get_runtime", return_value=self.runtime), patch.object(
            rag_service.llm_client, "chat", side_effect=AssertionError("LLM не должна вызываться")
        ):
            view = rag_service.retrieve("Какое давление в колонне К-1?")

        self.assertEqual(len(view.sources), 2)
        self.assertIn("c1", view.context)

    def test_empty_question_raises(self) -> None:
        with patch.object(rag_service, "get_runtime", return_value=self.runtime), patch.object(
            rag_service.llm_client, "chat", side_effect=AssertionError("LLM не должна вызываться")
        ):
            with self.assertRaises(rag_service.RAGError):
                rag_service.retrieve("   ")

    def test_context_truncation_is_reported(self) -> None:
        big_chunks = [
            make_chunk(f"big{i}", "я" * 5000, subsection=None) for i in range(1, 4)
        ]
        runtime = StubRuntime(big_chunks)

        with patch.object(rag_service, "get_runtime", return_value=runtime):
            view = rag_service.retrieve("Большой вопрос", top_k=3)

        self.assertTrue(view.context_truncated)
        self.assertEqual(view.omitted_sources, 2)
        self.assertEqual(len(view.sources), 1)
        self.assertEqual(view.sources[0].number, 1)

    def test_expanded_siblings_are_attached_to_source(self) -> None:
        intro = "При превышении давления необходимо:"
        primary = make_chunk(
            "item1", "1. Снизить подачу сырья.", chunk_type="list_item", list_intro=intro
        )
        sibling = make_chunk(
            "item2", "2. Открыть сброс на факел.", chunk_type="list_item", list_intro=intro
        )
        runtime = StubRuntime([primary])
        key = (primary["document"], primary["section"], primary["subsection"], intro)
        runtime.list_groups = {key: [primary, sibling]}

        with patch.object(rag_service, "get_runtime", return_value=runtime):
            view = rag_service.retrieve("Что делать при превышении давления?")

        self.assertEqual(view.sources[0].expanded_chunk_ids, ["item2"])
        self.assertIn("Открыть сброс", view.sources[0].expanded_text)
        self.assertEqual(len(view.sources), 1)

    def test_source_text_is_humanized(self) -> None:
        result = self._ask("Ответ [1].")
        c2_source = result.sources[1]
        self.assertIn("8,4 млн т/год", c2_source.text)
        self.assertNotIn("$", c2_source.text)

    def test_search_query_respects_abbreviation_flag(self) -> None:
        with patch.object(rag_service, "get_runtime", return_value=self.runtime):
            rag_service.retrieve("Какие параметры у АВТ?", expand_abbrev=False)
            self.assertEqual(self.runtime.last_query, "Какие параметры у АВТ?")

            rag_service.retrieve("Какие параметры у АВТ?", expand_abbrev=True)
            self.assertEqual(self.runtime.last_query, expand_query("Какие параметры у АВТ?"))
            self.assertNotEqual(self.runtime.last_query, "Какие параметры у АВТ?")

    def test_expand_query_keeps_plain_question(self) -> None:
        question = "Какое давление в колонне К-1?"
        self.assertEqual(expand_query(question), question)


if __name__ == "__main__":
    unittest.main()
