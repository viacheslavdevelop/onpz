"""CLI генерации ответов по технологическому регламенту ЭЛОУ-АВТ."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(BASE_DIR / "Retrieval"))

import llm_client
import prompt as prompt_builder
from hybrid_retrieval import (
    ALPHA_DEFAULT,
    build_chunk_lookup,
    build_list_groups,
    expand_results,
    format_expanded_procedure,
    group_siblings_by_source,
    hybrid_search,
)
from query_expansion import expand_query

EVAL_DIR = BASE_DIR / "Eval"
TEST_QUERIES_PATH = EVAL_DIR / "test_queries.json"

DEFAULT_TOP_K = 5
DEFAULT_TAG = "baseline"
DEFAULT_TEMPERATURE = 0.1


def load_test_queries() -> list[dict]:
    return json.loads(TEST_QUERIES_PATH.read_text(encoding="utf-8"))


def artifact_paths(tag: str) -> tuple[Path, Path]:
    if tag == DEFAULT_TAG:
        return EVAL_DIR / "answers.jsonl", EVAL_DIR / "generation_results.md"
    return EVAL_DIR / f"answers_{tag}.jsonl", EVAL_DIR / f"generation_results_{tag}.md"


def build_expanded_by_source(
    results: list[tuple[str, float]],
    chunk_lookup: dict,
    list_groups: dict,
) -> dict[str, list[dict]]:
    """Привязать соседей по list_item-группе к первому найденному источнику группы."""
    return group_siblings_by_source(results, chunk_lookup, list_groups)


def collect_expanded_chunks(
    results: list[tuple[str, float]],
    chunk_lookup: dict,
    list_groups: dict,
) -> list[dict]:
    chunk_ids = [chunk_id for chunk_id, _score in results]
    if not any(chunk_lookup.get(chunk_id, {}).get("type") == "list_item" for chunk_id in chunk_ids):
        return []
    expanded = expand_results(chunk_ids, chunk_lookup, list_groups)
    return [chunk for chunk in expanded if chunk.get("type") == "list_item"]


def format_context_summary(chunks: list[dict]) -> str:
    return "\n".join(
        f"{chunk['rank']}. `{chunk['chunk_id']}` (score={chunk['score']:.3f})"
        for chunk in chunks
    )


def format_expanded_summary(expanded_chunks: list[dict]) -> str:
    if not expanded_chunks:
        return ""
    chunk_ids = ", ".join(f"`{chunk['chunk_id']}`" for chunk in expanded_chunks)
    return f"**Развёрнутый контекст процедуры:** {chunk_ids}"


def format_usage(usage: dict) -> str:
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    return f"tokens: {prompt_tokens} in / {completion_tokens} out"


def normalize_usage(usage: dict) -> dict:
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }


def build_record(
    query: dict,
    *,
    tag: str,
    alpha: float,
    top_k: int,
    expand_abbrev: bool,
    think: bool,
    results: list[tuple[str, float]],
    answer: str,
    refused: bool,
    prompt_chars: int,
    usage: dict,
    latency_s: float,
) -> dict:
    return {
        "query_id": query["query_id"],
        "category": query["category"],
        "query_text": query["query_text"],
        "tag": tag,
        "retrieval": {
            "method": "hybrid",
            "alpha": alpha,
            "top_k": top_k,
            "expand_abbrev": expand_abbrev,
            "chunks": [
                {"rank": rank, "chunk_id": chunk_id, "score": round(score, 3)}
                for rank, (chunk_id, score) in enumerate(results, start=1)
            ],
        },
        "llm": {
            "model": llm_client.MODEL,
            "temperature": DEFAULT_TEMPERATURE,
            "think": think,
            "seed": llm_client.SEED,
        },
        "prompt_chars": prompt_chars,
        "answer": answer,
        "refused": refused,
        "usage": normalize_usage(usage),
        "latency_s": round(latency_s, 2),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def build_error_record(
    query: dict,
    *,
    tag: str,
    alpha: float,
    top_k: int,
    expand_abbrev: bool,
    think: bool,
    error: str,
) -> dict:
    return {
        "query_id": query["query_id"],
        "category": query["category"],
        "query_text": query["query_text"],
        "tag": tag,
        "retrieval": {
            "method": "hybrid",
            "alpha": alpha,
            "top_k": top_k,
            "expand_abbrev": expand_abbrev,
            "chunks": [],
        },
        "llm": {
            "model": llm_client.MODEL,
            "temperature": DEFAULT_TEMPERATURE,
            "think": think,
            "seed": llm_client.SEED,
        },
        "prompt_chars": 0,
        "answer": "",
        "refused": False,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        "latency_s": 0.0,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "error": error,
    }


def generate_answer(
    query: dict,
    *,
    chunk_lookup: dict,
    list_groups: dict,
    top_k: int,
    alpha: float,
    tag: str,
    think: bool,
    expand_abbrev: bool,
) -> tuple[dict, str, str, list[dict]]:
    retrieval_query = expand_query(query["query_text"]) if expand_abbrev else query["query_text"]
    results = hybrid_search(retrieval_query, top_k=top_k, alpha=alpha)

    expanded = collect_expanded_chunks(results, chunk_lookup, list_groups)
    expanded_by_source = build_expanded_by_source(results, chunk_lookup, list_groups)
    blocks = prompt_builder.build_context_blocks(results, chunk_lookup, expanded_by_source)
    context = prompt_builder.render_context(blocks)
    messages = prompt_builder.build_messages(query["query_text"], context)

    response = llm_client.chat(messages, temperature=DEFAULT_TEMPERATURE, think=think)
    answer = prompt_builder.sanitize_citations(response["text"].strip(), context)
    if prompt_builder.is_refusal(answer):
        answer = prompt_builder.REFUSAL_PHRASE
    if not answer:
        print(f"Предупреждение: пустой ответ модели на запрос {query['query_id']}", file=sys.stderr)

    record = build_record(
        query,
        tag=tag,
        alpha=alpha,
        top_k=top_k,
        expand_abbrev=expand_abbrev,
        think=think,
        results=results,
        answer=answer,
        refused=prompt_builder.is_refusal(answer),
        prompt_chars=sum(len(message["content"]) for message in messages),
        usage=response["usage"],
        latency_s=response["latency_s"],
    )
    sources = prompt_builder.format_sources(results, chunk_lookup)
    return record, context, sources, expanded


def run_single(
    query_text: str,
    *,
    top_k: int,
    alpha: float,
    tag: str,
    think: bool,
    show_context: bool,
    expand_abbrev: bool,
) -> int:
    try:
        llm_client.healthcheck()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 2

    chunk_lookup = build_chunk_lookup()
    list_groups = build_list_groups(list(chunk_lookup.values()))
    query = {
        "query_id": "query",
        "category": "single",
        "query_text": query_text,
    }

    try:
        record, context, sources, expanded = generate_answer(
            query,
            chunk_lookup=chunk_lookup,
            list_groups=list_groups,
            top_k=top_k,
            alpha=alpha,
            tag=tag,
            think=think,
            expand_abbrev=expand_abbrev,
        )
    except Exception as exc:
        print(f"Ошибка генерации: {exc}", file=sys.stderr)
        return 1

    if show_context:
        print("КОНТЕКСТ:")
        print(context)
        print()

    print(f"ВОПРОС: {query_text}")
    print("ОТВЕТ:")
    print(record["answer"])
    print()
    print(sources)
    if expanded:
        print()
        print(prompt_builder.humanize_math(format_expanded_procedure(expanded)))
    print()
    print(
        f"[latency: {record['latency_s']:.1f} s | {format_usage(record['usage'])} | tag={tag}]"
    )
    return 0


def format_markdown_section(record: dict, sources: str, expanded_summary: str) -> str:
    lines = [f"## {record['query_id']} [{record['category']}] {record['query_text']}", ""]

    if record.get("error"):
        lines.append(f"**Ошибка:** {record['error']}")
        lines.append("")
        return "\n".join(lines)

    lines.append(
        f"**Контекст (hybrid, alpha={record['retrieval']['alpha']}, "
        f"top_k={record['retrieval']['top_k']}):**"
    )
    lines.append(format_context_summary(record["retrieval"]["chunks"]))
    if expanded_summary:
        lines.append("")
        lines.append(expanded_summary)
    lines.append("")
    lines.append("**Ответ:**")
    lines.append(record["answer"] or "_(пустой ответ)_")
    lines.append("")
    lines.append(sources)
    lines.append("")
    lines.append(
        f"**Метрики:** latency {record['latency_s']:.2f} s | "
        f"{format_usage(record['usage'])} | prompt {record['prompt_chars']} chars | "
        f"tag={record['tag']}"
    )
    lines.append("")
    return "\n".join(lines)


def write_jsonl(path: Path, records: list[dict]) -> None:
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown(path: Path, tag: str, top_k: int, alpha: float, sections: list[str]) -> None:
    header = [
        f"# Результаты генерации (tag={tag}, top_k={top_k}, alpha={alpha})",
        "",
        f"Всего запросов: {len(sections)}",
        "",
    ]
    path.write_text("\n".join(header + sections), encoding="utf-8")


def run_all(
    *,
    top_k: int,
    alpha: float,
    tag: str,
    think: bool,
    expand_abbrev: bool,
) -> int:
    try:
        llm_client.healthcheck()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 2

    chunk_lookup = build_chunk_lookup()
    list_groups = build_list_groups(list(chunk_lookup.values()))
    queries = load_test_queries()

    records: list[dict] = []
    sections: list[str] = []
    ok_count = 0
    error_count = 0

    for index, query in enumerate(queries, start=1):
        print(f"[{index}/{len(queries)}] {query['query_id']}", file=sys.stderr)
        try:
            record, _context, sources, expanded = generate_answer(
                query,
                chunk_lookup=chunk_lookup,
                list_groups=list_groups,
                top_k=top_k,
                alpha=alpha,
                tag=tag,
                think=think,
                expand_abbrev=expand_abbrev,
            )
            sections.append(
                format_markdown_section(
                    record,
                    sources=sources,
                    expanded_summary=format_expanded_summary(expanded),
                )
            )
            ok_count += 1
        except Exception as exc:
            error_count += 1
            record = build_error_record(
                query,
                tag=tag,
                alpha=alpha,
                top_k=top_k,
                expand_abbrev=expand_abbrev,
                think=think,
                error=str(exc),
            )
            sections.append(format_markdown_section(record, sources="", expanded_summary=""))
            print(f"Ошибка на запросе {query['query_id']}: {exc}", file=sys.stderr)
        records.append(record)

    answers_path, results_path = artifact_paths(tag)
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(answers_path, records)
    write_markdown(results_path, tag, top_k, alpha, sections)

    print(f"Сохранено: {answers_path}")
    print(f"Сохранено: {results_path}")
    print(f"Итог: {ok_count} ok / {error_count} error")
    return 1 if error_count else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Генерация ответов по регламенту ЭЛОУ-АВТ")
    parser.add_argument("--query", help="один вопрос к системе")
    parser.add_argument("--run_all", action="store_true", help="прогнать все тестовые запросы")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="число источников")
    parser.add_argument("--alpha", type=float, default=ALPHA_DEFAULT, help="вес BM25 в гибридном поиске")
    parser.add_argument("--tag", default=DEFAULT_TAG, help="метка прогона для имён артефактов")
    parser.add_argument("--think", action="store_true", help="включить рассуждения модели")
    parser.add_argument("--show_context", action="store_true", help="печатать контекст перед ответом")
    parser.add_argument(
        "--expand_abbrev",
        action="store_true",
        help="раскрывать аббревиатуры в поисковом запросе",
    )
    args = parser.parse_args()

    if args.run_all and args.query:
        parser.error("укажите только один режим: --query или --run_all")
    if not args.run_all and not args.query:
        parser.error("укажите --query или --run_all")

    if args.run_all:
        return run_all(
            top_k=args.top_k,
            alpha=args.alpha,
            tag=args.tag,
            think=args.think,
            expand_abbrev=args.expand_abbrev,
        )

    return run_single(
        args.query,
        top_k=args.top_k,
        alpha=args.alpha,
        tag=args.tag,
        think=args.think,
        show_context=args.show_context,
        expand_abbrev=args.expand_abbrev,
    )


if __name__ == "__main__":
    sys.exit(main())
