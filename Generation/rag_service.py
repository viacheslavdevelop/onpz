"""Сервисный слой RAG: поиск, сборка контекста и генерация ответа.

Модуль не зависит от UI-фреймворка, не пишет артефакты в ``Eval/``,
``Chunks/`` и ``Processed data/`` и не хранит глобального состояния, кроме
кэша ресурсов поиска из ``Retrieval/runtime.py``.
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(BASE_DIR / "Retrieval"))

import llm_client
import prompt as prompt_builder
from hybrid_retrieval import ALPHA_DEFAULT, group_siblings_by_source
from query_expansion import expand_query
from runtime import get_runtime

DEFAULT_TOP_K = 5
DEFAULT_TEMPERATURE = 0.1


class RAGError(RuntimeError):
    """Ошибка входных данных или пайплайна, понятная пользователю."""


@dataclass
class SourceCard:
    """Источник, реально переданный в контекст LLM."""

    number: int
    chunk_id: str
    score: float
    document: str
    section: str | None
    subsection: str | None
    type: str
    text: str
    expanded_chunk_ids: list[str] = field(default_factory=list)
    expanded_text: str = ""


@dataclass
class RetrievalView:
    """Результат поиска и подготовки контекста без обращения к LLM."""

    question: str
    search_query: str
    results: list[tuple[str, float]]
    sources: list[SourceCard]
    context: str
    context_truncated: bool
    omitted_sources: int
    latency_s: float


@dataclass
class RAGAnswer:
    """Полный результат пайплайна: ответ LLM и источники контекста."""

    question: str
    answer: str
    refused: bool
    sources: list[SourceCard]
    context: str
    context_truncated: bool
    omitted_sources: int
    latency_s: float
    retrieval_latency_s: float
    generation_latency_s: float
    usage: dict
    settings: dict


def _source_from_block(block, score: float) -> SourceCard:
    chunk = block.chunk
    extra_texts = [
        prompt_builder.humanize_math(extra.get("text", "").strip())
        for extra in block.extra_chunks
        if extra.get("text", "").strip()
    ]
    return SourceCard(
        number=block.number,
        chunk_id=block.chunk_id,
        score=score,
        document=chunk.get("document", ""),
        section=chunk.get("section"),
        subsection=chunk.get("subsection"),
        type=chunk.get("type", ""),
        text=prompt_builder.humanize_math(chunk.get("text", "")),
        expanded_chunk_ids=[extra["chunk_id"] for extra in block.extra_chunks],
        expanded_text="\n\n".join(extra_texts),
    )


def normalize_usage(usage: dict | None) -> dict:
    """Привести usage к паре целых счётчиков токенов."""
    usage = usage or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
    }


def retrieve(
    question: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    alpha: float = ALPHA_DEFAULT,
    expand_abbrev: bool = False,
    method: str = "hybrid",
    runtime=None,
) -> RetrievalView:
    """Найти источники и собрать контекст, не обращаясь к LLM."""
    started_at = time.perf_counter()
    question = question.strip()
    if not question:
        raise RAGError("Введите вопрос.")

    runtime = runtime or get_runtime()
    search_query = expand_query(question) if expand_abbrev else question
    results = runtime.search(search_query, top_k=top_k, alpha=alpha, method=method)

    expanded_by_source = group_siblings_by_source(
        results, runtime.chunk_lookup, runtime.list_groups
    )
    blocks = prompt_builder.build_context_blocks(
        results, runtime.chunk_lookup, expanded_by_source
    )
    scores = dict(results)
    sources = [_source_from_block(block, scores[block.chunk_id]) for block in blocks]
    context = prompt_builder.render_context(blocks)

    return RetrievalView(
        question=question,
        search_query=search_query,
        results=results,
        sources=sources,
        context=context,
        context_truncated=len(blocks) < len(results),
        omitted_sources=len(results) - len(blocks),
        latency_s=time.perf_counter() - started_at,
    )


def ask(
    question: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    alpha: float = ALPHA_DEFAULT,
    expand_abbrev: bool = False,
    think: bool = False,
    method: str = "hybrid",
    runtime=None,
) -> RAGAnswer:
    """Ответить на вопрос по регламенту с опорой на найденный контекст."""
    started_at = time.perf_counter()
    view = retrieve(
        question,
        top_k=top_k,
        alpha=alpha,
        expand_abbrev=expand_abbrev,
        method=method,
        runtime=runtime,
    )

    messages = prompt_builder.build_messages(view.question, view.context)
    response = llm_client.chat(messages, temperature=DEFAULT_TEMPERATURE, think=think)
    answer = prompt_builder.sanitize_citations(response["text"].strip(), view.context)

    return RAGAnswer(
        question=view.question,
        answer=answer,
        refused=prompt_builder.is_refusal(answer),
        sources=view.sources,
        context=view.context,
        context_truncated=view.context_truncated,
        omitted_sources=view.omitted_sources,
        latency_s=time.perf_counter() - started_at,
        retrieval_latency_s=view.latency_s,
        generation_latency_s=response["latency_s"],
        usage=normalize_usage(response.get("usage")),
        settings={
            "top_k": top_k,
            "alpha": alpha,
            "expand_abbrev": expand_abbrev,
            "think": think,
            "method": method,
            "model": llm_client.MODEL,
        },
    )
