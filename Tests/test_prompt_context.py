"""Тесты соответствия контекста LLM и списка источников."""

import unittest

from Tests import _bootstrap  # noqa: F401

import prompt


def make_chunk(
    chunk_id: str,
    text: str,
    *,
    document: str = "03_reglament_at_rezhim_k1_k2.md",
    section: str = "3. Режим работы колонн К-1 и К-2",
    subsection: str | None = "3.1. Давление",
    chunk_type: str = "text",
) -> dict:
    return {
        "chunk_id": chunk_id,
        "document": document,
        "section": section,
        "subsection": subsection,
        "type": chunk_type,
        "text": text,
    }


class BuildContextBlocksTest(unittest.TestCase):
    def test_six_chunks_produce_six_numbered_blocks(self) -> None:
        chunks = {f"c{i}": make_chunk(f"c{i}", f"Текст фрагмента {i}.") for i in range(1, 7)}
        results = [(f"c{i}", 1.0 - i / 10) for i in range(1, 7)]

        blocks = prompt.build_context_blocks(results, chunks)

        self.assertEqual([block.number for block in blocks], list(range(1, 7)))
        rendered = prompt.render_context(blocks)
        for chunk_id in chunks:
            self.assertIn(chunk_id, rendered)
        sources = prompt.format_sources_from_blocks(blocks)
        self.assertEqual(len(sources.splitlines()), 7)
        self.assertTrue(sources.startswith("**Источники:**"))

    def test_oversized_chunks_are_dropped_and_sources_follow_blocks(self) -> None:
        chunks = {
            f"big{i}": make_chunk(f"big{i}", "я" * 5000) for i in range(1, 4)
        }
        results = [(f"big{i}", 1.0) for i in range(1, 4)]

        blocks = prompt.build_context_blocks(results, chunks)
        sources = prompt.format_sources_from_blocks(blocks)

        self.assertLess(len(blocks), 3)
        self.assertEqual([block.number for block in blocks], list(range(1, len(blocks) + 1)))
        for block in blocks:
            self.assertIn(block.chunk_id, sources)
        dropped = {f"big{i}" for i in range(1, 4)} - {block.chunk_id for block in blocks}
        for chunk_id in dropped:
            self.assertNotIn(chunk_id, sources)
        self.assertLessEqual(len(prompt.render_context(blocks)), prompt.MAX_CONTEXT_CHARS)

    def test_missing_chunk_does_not_break_numbering(self) -> None:
        chunks = {
            "a": make_chunk("a", "Первый фрагмент."),
            "b": make_chunk("b", "Второй фрагмент."),
        }
        results = [("a", 1.0), ("missing", 0.9), ("b", 0.8)]

        blocks = prompt.build_context_blocks(results, chunks)
        sources = prompt.format_sources_from_blocks(blocks)

        self.assertEqual([block.chunk_id for block in blocks], ["a", "b"])
        self.assertEqual([block.number for block in blocks], [1, 2])
        self.assertNotIn("missing", sources)
        self.assertEqual(len(sources.splitlines()), 3)

    def test_build_context_is_render_of_blocks(self) -> None:
        chunks = {f"c{i}": make_chunk(f"c{i}", f"Текст {i}.") for i in range(1, 4)}
        results = [(f"c{i}", 1.0) for i in range(1, 4)]

        self.assertEqual(
            prompt.build_context(results, chunks),
            prompt.render_context(prompt.build_context_blocks(results, chunks)),
        )

    def test_extra_chunks_are_in_text_but_not_extra_sources(self) -> None:
        chunk = make_chunk("item1", "Первый пункт процедуры.", chunk_type="list_item")
        extra = make_chunk("item2", "Второй пункт процедуры.", chunk_type="list_item")
        chunks = {"item1": chunk}
        results = [("item1", 1.0)]

        blocks = prompt.build_context_blocks(results, chunks, {"item1": [extra]})
        sources = prompt.format_sources_from_blocks(blocks)

        self.assertEqual(len(blocks), 1)
        self.assertIn("Первый пункт процедуры.", blocks[0].text)
        self.assertIn("Второй пункт процедуры.", blocks[0].text)
        self.assertEqual(blocks[0].extra_chunks, (extra,))
        self.assertEqual(len(sources.splitlines()), 2)


class CitationHelpersTest(unittest.TestCase):
    def test_sanitize_keeps_text_without_citations(self) -> None:
        answer = "Давление составляет 0,15 МПа, температура 120 °C."
        self.assertEqual(prompt.sanitize_citations(answer, "[1] chunk_id=a\nтекст"), answer)

    def test_sanitize_removes_citation_outside_context(self) -> None:
        context = "[1] chunk_id=a\nтекст\n\n[2] chunk_id=b\nтекст"
        cleaned = prompt.sanitize_citations("Ответ [1] [9].", context)
        self.assertIn("[1]", cleaned)
        self.assertNotIn("[9]", cleaned)

    def test_is_refusal_ignores_case_and_trailing_period(self) -> None:
        self.assertTrue(prompt.is_refusal(prompt.REFUSAL_PHRASE))
        self.assertTrue(prompt.is_refusal(prompt.REFUSAL_PHRASE.rstrip(".")))
        self.assertTrue(prompt.is_refusal(prompt.REFUSAL_PHRASE.lower()))
        self.assertFalse(prompt.is_refusal("Давление составляет 0,15 МПа [1]."))


if __name__ == "__main__":
    unittest.main()
