"""Тесты рендеринга UI-компонентов без Gradio и сетевых вызовов."""

import unittest

from Tests import _bootstrap  # noqa: F401

import ui_components


class FormatMetaTest(unittest.TestCase):
    def _meta(self, omitted: int) -> dict:
        return {
            "latency_s": 1.0,
            "retrieval_latency_s": 0.1,
            "generation_latency_s": 0.9,
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            "settings": {"top_k": 10, "alpha": 0.7, "method": "hybrid", "model": "test"},
            "context_truncated": True,
            "omitted_sources": omitted,
            "refused": False,
        }

    def test_truncation_caption_uses_singular_for_one(self) -> None:
        text = ui_components.format_meta(self._meta(1), 9, "ответ")
        self.assertIn("1 не поместился", text)
        self.assertNotIn("1 не поместились", text)

    def test_truncation_caption_uses_plural_for_many(self) -> None:
        text = ui_components.format_meta(self._meta(2), 8, "ответ")
        self.assertIn("2 не поместились", text)


class AssistantContentTest(unittest.TestCase):
    def test_refusal_is_rendered_as_quote(self) -> None:
        content = ui_components.build_assistant_content(
            "Информация не найдена в регламенте.", True, []
        )
        self.assertTrue(content.startswith("> Информация не найдена в регламенте."))

    def test_table_row_is_wrapped_in_fence(self) -> None:
        source = {
            "number": 1,
            "document": "doc.md",
            "section": "Раздел",
            "subsection": None,
            "chunk_id": "doc__001",
            "type": "table_row",
            "score": 0.5,
            "text": "| a | `b` |",
            "expanded_text": "",
        }
        _label, body = ui_components.render_source_card(source)
        self.assertIn("```\n| a | `b` |\n```", body)
