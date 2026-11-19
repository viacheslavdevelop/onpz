"""Рендеринг источников, статуса LLM и метаданных для веб-интерфейса."""

import re

MAX_SOURCE_CARDS = 10
MAX_LABEL_LENGTH = 90
SCORE_DIGITS = 3

LLM_START_COMMAND = (
    'llama-server -m "Models/T-lite-it-2.1-Q4_K_M.gguf" -c 8192 -ngl 99 '
    "-fa on --jinja --host 127.0.0.1 --port 8080"
)

METHOD_TITLES = {
    "hybrid": "гибридный",
    "bm25": "BM25",
    "vector": "векторный",
}


def truncate(text: str, limit: int = MAX_LABEL_LENGTH) -> str:
    """Обрезать длинную строку для заголовка аккордеона."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def source_location(source: dict) -> str:
    """Раздел или подраздел источника."""
    return source.get("section") or source.get("subsection") or ""


def format_llm_status(status: dict) -> str:
    """Краткий статус LLM для левой панели."""
    if status.get("ok"):
        model = status.get("model") or "модель не указана"
        return f"🟢 **LLM доступна** · модель `{model}`"

    error = status.get("error") or "причина неизвестна"
    return (
        f"🔴 **LLM недоступна** · {error}\n\n"
        "Запустите llama-server:\n\n"
        f"```powershell\n{LLM_START_COMMAND}\n```"
    )


def compact_sources_line(sources: list[dict]) -> str:
    """Компактная строка источников для сообщения чата."""
    if not sources:
        return "**Источники:** не найдены."

    parts = []
    for source in sources:
        location = source_location(source)
        description = f"{source.get('document', '')}"
        if location:
            description += f" — {location}"
        parts.append(f"[{source.get('number', '?')}] {description}")
    return "**Источники:** " + "; ".join(parts)


def _fenced_code(text: str) -> str:
    """Обернуть текст в fence длиннее любой серии backticks внутри текста."""
    runs = [len(match.group(0)) for match in re.finditer(r"`+", text)]
    fence_length = max(3, max(runs, default=0) + 1)
    fence = "`" * fence_length
    return f"{fence}\n{text.strip()}\n{fence}"


def render_source_card(source: dict) -> tuple[str, str]:
    """Вернуть заголовок и Markdown-тело карточки источника."""
    location = source_location(source)
    description = source.get("document", "")
    if location:
        description = f"{description} — {location}"
    label = f"{source.get('number', '?')}. {truncate(description)}"

    score = float(source.get("score", 0.0))
    caption = (
        f"`{source.get('chunk_id', '')}` · {source.get('type', '')} · "
        f"score={score:.{SCORE_DIGITS}f} · балл гибридного ранжирования"
    )

    text = source.get("text", "").strip()
    if source.get("type") == "table_row":
        body_text = _fenced_code(text) if text else "_Текст фрагмента пуст._"
    else:
        body_text = text or "_Текст фрагмента пуст._"

    body = f"{caption}\n\n{body_text}"
    expanded_text = source.get("expanded_text", "").strip()
    if expanded_text:
        body += (
            "\n\n_Дополнено соседними пунктами процедуры:_\n\n"
            f"{expanded_text}"
        )
    return label, body


def format_meta(meta: dict, sources_count: int, answer: str = "") -> str:
    """Метаданные последнего ответа."""
    settings = meta.get("settings", {})
    usage = meta.get("usage", {})
    lines = [
        (
            f"⏱ {meta.get('latency_s', 0.0):.1f} с "
            f"(поиск {meta.get('retrieval_latency_s', 0.0):.1f} / "
            f"генерация {meta.get('generation_latency_s', 0.0):.1f}) · "
            f"tokens {usage.get('prompt_tokens', 0)} in / "
            f"{usage.get('completion_tokens', 0)} out"
        ),
        (
            f"top_k={settings.get('top_k', '—')} · alpha={settings.get('alpha', '—')} · "
            f"метод={settings.get('method', '—')} · модель={settings.get('model', '—')}"
        ),
        f"Источников в контексте: {sources_count}",
    ]

    if meta.get("context_truncated"):
        omitted = int(meta.get("omitted_sources", 0))
        total = sources_count + omitted
        verb = (
            "не поместился"
            if omitted % 10 == 1 and omitted % 100 != 11
            else "не поместились"
        )
        lines.append(
            f"В контекст вошли {sources_count} из {total} источников; "
            f"{omitted} {verb} из-за лимита 9000 символов."
        )

    if meta.get("refused"):
        lines.append("_Модель не нашла ответа в переданном контексте._")
        lines.append("_Найденные фрагменты не были использованы в ответе._")
    elif answer == "":
        lines.append(
            "⚠️ Модель вернула пустой ответ. Попробуйте переформулировать вопрос."
        )

    return "\n\n".join(lines)


def build_assistant_content(answer: str, refused: bool, sources: list[dict]) -> str:
    """Собрать Markdown сообщения ассистента для чата."""
    sources_line = compact_sources_line(sources)
    if refused:
        return (
            f"> {answer}\n\n"
            "_Модель не нашла ответа в переданном контексте._\n\n"
            f"{sources_line}"
        )
    if not answer:
        return (
            "⚠️ Модель вернула пустой ответ. Попробуйте переформулировать вопрос.\n\n"
            f"{sources_line}"
        )
    return f"{answer}\n\n{sources_line}"


def to_chatbot_messages(history: list[dict]) -> list[dict]:
    """Оставить в истории только поля, понятные gr.Chatbot."""
    return [
        {"role": message["role"], "content": message.get("content", "")}
        for message in history
        if message.get("role") in {"user", "assistant"}
    ]


def method_title(method: str) -> str:
    """Человекочитаемое название метода поиска."""
    return METHOD_TITLES.get(method, method)
