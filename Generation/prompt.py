"""Промпт и сборка контекста для генерации ответа по регламенту."""

import re

REFUSAL_PHRASE = "Информация не найдена в регламенте."
MAX_CONTEXT_CHARS = 9000

INLINE_MATH_PATTERN = re.compile(r"\$([^$]+)\$")
TEXT_COMMAND_PATTERN = re.compile(r"\\(?:text|mathrm|mathbf|mathit)\{([^{}]*)\}")
SUBSCRIPT_PATTERN = re.compile(r"_\{([^{}]*)\}")
SUPERSCRIPT_PATTERN = re.compile(r"\^\{([^{}]*)\}")
SHORT_SUBSCRIPT_PATTERN = re.compile(r"_([A-Za-z0-9])")
SHORT_SUPERSCRIPT_PATTERN = re.compile(r"\^([0-9])")
CONTEXT_SOURCE_PATTERN = re.compile(r"^\[(\d+)\]", re.MULTILINE)
CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
SUPERSCRIPTS = {"2": "²", "3": "³"}

SYSTEM_PROMPT = """Ты — инженер-технолог нефтеперерабатывающего завода, работающий с технологическим регламентом комплекса ЭЛОУ-АВТ.

Правила ответа:
1. Отвечай только по блокам контекста, которые приведены в сообщении пользователя. Внешние знания и предположения запрещены.
2. Если в контексте нет ответа, ответь ровно: «Информация не найдена в регламенте.» — и больше ничего не добавляй. Не смешивай отказ с содержательным ответом.
3. После каждого утверждения указывай номер источника в квадратных скобках: [1], [2]. Используй только номера, которые есть в контексте.
4. Числа, единицы измерения, температуры и давления приводи дословно, как в контексте. Не округляй, не пересчитывай и не переводи в другие единицы.
5. Если данные в разных блоках противоречат друг другу, укажи оба значения со ссылками на источники.
6. Отвечай на русском языке в кратком техническом стиле, без вступлений, извинений и общих рассуждений.
7. Не пересказывай вопрос и не добавляй разделы «Введение» и «Заключение».
8. Не используй LaTeX и служебную разметку; записывай числа, единицы и формулы обычным текстом.
9. Не пересказывай контекст целиком и не добавляй пояснений, без которых ответ остаётся полным.
10. Ответ должен содержать не более 150 слов. Не копируй длинные предложения из контекста, формулируй шаги кратко.
"""


def _humanize_expression(expression: str) -> str:
    """Преобразовать одну inline-формулу LaTeX в обычную текстовую запись."""
    value = TEXT_COMMAND_PATTERN.sub(r"\1", expression)
    value = value.replace(r"\%", "%")
    value = value.replace(r"\,", " ").replace(r"\;", " ").replace(r"\ ", " ")
    value = value.replace(r"^\circ", "°")
    value = value.replace(r"\ge", "≥").replace(r"\le", "≤").replace(r"\to", "→")
    value = value.replace("{,}", ",").replace("--", "–")
    value = SUBSCRIPT_PATTERN.sub(r"\1", value)
    value = SUPERSCRIPT_PATTERN.sub(r"\1", value)
    value = SHORT_SUBSCRIPT_PATTERN.sub(r"\1", value)
    value = SHORT_SUPERSCRIPT_PATTERN.sub(lambda match: SUPERSCRIPTS.get(match.group(1), match.group(1)), value)
    value = value.replace("{", "").replace("}", "").replace("\\", "")
    return " ".join(value.split())


def humanize_math(text: str) -> str:
    """Заменить inline-формулы LaTeX на читаемый текст."""
    return INLINE_MATH_PATTERN.sub(lambda match: _humanize_expression(match.group(1)), text)


def _chunk_location(chunk: dict) -> str:
    parts = [chunk.get("section"), chunk.get("subsection")]
    return " / ".join(part for part in parts if part)


def _format_chunk_header(number: int, chunk: dict) -> str:
    fields = [
        f"chunk_id={chunk.get('chunk_id', '')}",
        f"document={chunk.get('document', '')}",
    ]
    if chunk.get("section"):
        fields.append(f"section={chunk['section']}")
    if chunk.get("subsection"):
        fields.append(f"subsection={chunk['subsection']}")
    return f"[{number}] " + " | ".join(fields)


def _format_chunk_body(chunk: dict, extra_chunks: list[dict] | None = None) -> str:
    text = humanize_math(chunk.get("text", "").strip())
    if not extra_chunks:
        return text

    additions = [humanize_math(extra.get("text", "").strip()) for extra in extra_chunks]
    additions = [addition for addition in additions if addition]
    if not additions:
        return text
    return text + "\n\n" + "\n".join(additions)


def _format_block(number: int, chunk: dict, extra_chunks: list[dict] | None = None) -> str:
    return f"{_format_chunk_header(number, chunk)}\n{_format_chunk_body(chunk, extra_chunks)}"


def build_context(
    results: list[tuple[str, float]],
    chunk_lookup: dict,
    expanded_chunks: dict[str, list[dict]] | None = None,
) -> str:
    """Собрать пронумерованный контекст, обрезая его по границе блока."""
    blocks: list[str] = []
    current_length = 0
    number = 0

    for chunk_id, _score in results:
        chunk = chunk_lookup.get(chunk_id)
        if chunk is None:
            continue

        number += 1
        extras = (expanded_chunks or {}).get(chunk_id)
        block = _format_block(number, chunk, extras)
        separator_length = 2 if blocks else 0
        if blocks and current_length + separator_length + len(block) > MAX_CONTEXT_CHARS:
            break

        blocks.append(block)
        current_length += separator_length + len(block)

    return "\n\n".join(blocks)


def build_messages(query: str, context: str) -> list[dict]:
    """Собрать сообщения для чат-модели."""
    user_content = f"Контекст:\n{context}\n\nВопрос: {query}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def format_sources(results: list[tuple[str, float]], chunk_lookup: dict) -> str:
    """Сформировать блок источников кодом, а не моделью."""
    lines = ["**Источники:**"]
    for number, (chunk_id, _score) in enumerate(results, start=1):
        chunk = chunk_lookup.get(chunk_id, {})
        document = chunk.get("document", "")
        location = humanize_math(_chunk_location(chunk))
        description = f"{document}, {location}" if location else document
        lines.append(f"{number}. `{chunk_id}` — {description}")
    return "\n".join(lines)


def sanitize_citations(answer: str, context: str) -> str:
    """Удалить из ответа ссылки на источники, которых нет в контексте."""
    source_numbers = [int(number) for number in CONTEXT_SOURCE_PATTERN.findall(context)]
    if not source_numbers:
        return answer

    max_source = max(source_numbers)

    def replace(match: re.Match) -> str:
        numbers = [number.strip() for number in match.group(1).split(",")]
        kept = [
            number
            for number in numbers
            if number.isdigit() and 1 <= int(number) <= max_source
        ]
        return f"[{', '.join(kept)}]" if kept else ""

    cleaned = CITATION_PATTERN.sub(replace, answer)
    cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r",\s*([.;])", r"\1", cleaned)
    cleaned = re.sub(r"^\s*[.,;]\s*", "", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def is_refusal(answer: str) -> bool:
    """Проверить, что ответ начинается с фразы отказа."""
    normalized = re.sub(r"[^\w\s]", "", answer.lower())
    normalized = " ".join(normalized.split())
    return normalized.startswith("информация не найдена")
