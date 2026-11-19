"""Гибридный поиск: объединение BM25 и векторного поиска (неделя 5).

Подход:
    1. Для запроса получаем top-N кандидатов отдельно от BM25 и от векторного
       поиска (N существенно больше итогового top_k, по умолчанию 20),
       чтобы собрать достаточно широкий пул кандидатов для переранжирования.
    2. Скоры каждого метода приводятся к шкале [0, 1] через min-max
       нормализацию в рамках текущего пула кандидатов (а не глобально) —
       это снимает проблему разных по природе шкал: BM25-скор не ограничен
       сверху, косинусный/IP-скор FAISS лежит в [-1, 1].
    3. Итоговый скор: hybrid_score = alpha * bm25_norm + (1 - alpha) * vector_norm.
       Если чанк найден только одним из методов, скор отсутствующего метода
       считается 0 (а не исключается из рассмотрения) — это соответствует
       идее "гибрид расширяет охват", а не "пересечение методов".

Подбор alpha (см. Eval/alpha_tuning.md и частичную разметку gold_chunk_ids
в Eval/test_queries.json): по результатам сравнения Recall@5 на размеченных
запросах категорий factual и synonym_abbreviation выбрано ALPHA_DEFAULT.
"""

import argparse
import json
from pathlib import Path

import bm25_search
import vector_search
from chunks_io import load_chunks

EVAL_DIR = Path(__file__).parent.parent / "Eval"
TEST_QUERIES_PATH = EVAL_DIR / "test_queries.json"
HYBRID_RESULTS_PATH = EVAL_DIR / "hybrid_results.md"

# Зафиксировано по итогам подбора на размеченных запросах (Eval/alpha_tuning.md):
# при alpha=0.7 достигнут полный Recall@5 (13/13) на 13 размеченных запросах
# категорий factual и synonym_abbreviation, при этом в отличие от alpha=1.0
# (чистый BM25) сохраняется вклад векторного метода для устойчивости на
# нетипичных формулировках запросов, не покрытых разметкой.
ALPHA_DEFAULT = 0.7
CANDIDATE_POOL_SIZE = 20


def build_chunk_lookup() -> dict:
    return {c["chunk_id"]: c for c in load_chunks()}


def list_group_key(chunk: dict) -> tuple | None:
    if chunk.get("type") != "list_item" or not chunk.get("list_intro"):
        return None
    return (chunk["document"], chunk.get("section"), chunk.get("subsection"), chunk["list_intro"])


def part_index(chunk: dict) -> int:
    part = chunk.get("part")
    return int(part.split("/")[0]) if part else 0


def build_list_groups(chunks: list[dict]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for chunk in chunks:
        key = list_group_key(chunk)
        if key is not None:
            groups.setdefault(key, []).append(chunk)
    for group in groups.values():
        group.sort(key=part_index)
    return groups


def expand_list_siblings(chunk_id: str, chunk_lookup: dict, list_groups: dict) -> list[dict]:
    chunk = chunk_lookup.get(chunk_id)
    if chunk is None:
        return []
    key = list_group_key(chunk)
    return list_groups[key] if key is not None else [chunk]


def expand_results(chunk_ids: list[str], chunk_lookup: dict, list_groups: dict) -> list[dict]:
    seen: set[str] = set()
    expanded: list[dict] = []
    for chunk_id in chunk_ids:
        for sibling in expand_list_siblings(chunk_id, chunk_lookup, list_groups):
            if sibling["chunk_id"] not in seen:
                seen.add(sibling["chunk_id"])
                expanded.append(sibling)
    return expanded


def format_expanded_procedure(expanded_chunks: list[dict]) -> str:
    lines = ["**Развёрнутый контекст процедуры (list_item-группы):**"]
    current_key = None
    for chunk in expanded_chunks:
        key = list_group_key(chunk)
        if key is not None and key != current_key:
            lines.append(f"_{chunk.get('subsection') or chunk.get('section')}: {chunk['list_intro']}_")
        current_key = key
        text = chunk["text"].replace("\n", " ").strip()
        lines.append(f"- `{chunk['chunk_id']}` {text}")
    return "\n".join(lines)


def chunk_summary(chunk: dict) -> str:
    location = chunk.get("subsection") or chunk.get("section") or chunk.get("document") or ""
    snippet = chunk.get("text", "").replace("\n", " ").strip()
    if len(snippet) > 150:
        snippet = snippet[:150] + "..."
    return f"{location} | {snippet}"


def group_siblings_by_source(
    results: list[tuple[str, float]],
    chunk_lookup: dict,
    list_groups: dict,
) -> dict[str, list[dict]]:
    """Для каждого источника вернуть соседей его list_item-группы.

    Соседи привязываются к первому найденному источнику группы и не включают
    чанки, которые уже попали в результаты поиска.
    """
    sources_by_group: dict[tuple, list[str]] = {}
    for chunk_id, _score in results:
        key = list_group_key(chunk_lookup.get(chunk_id, {}))
        if key is not None:
            sources_by_group.setdefault(key, []).append(chunk_id)

    expanded_by_source: dict[str, list[dict]] = {}
    for key, source_ids in sources_by_group.items():
        primary_source = source_ids[0]
        source_id_set = set(source_ids)
        siblings = list_groups.get(key, [])
        expanded_by_source[primary_source] = [
            sibling for sibling in siblings if sibling["chunk_id"] not in source_id_set
        ]
    return expanded_by_source


def min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Нормализует словарь {chunk_id: score} в диапазон [0, 1].

    Если все скоры совпадают (включая пустой словарь или единственный
    элемент), возвращает 1.0 для всех присутствующих ключей, чтобы не
    делить на ноль и не терять сигнал о самом факте присутствия чанка
    в кандидатах метода.
    """
    if not scores:
        return {}
    values = scores.values()
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return {chunk_id: 1.0 for chunk_id in scores}
    return {chunk_id: (score - lo) / (hi - lo) for chunk_id, score in scores.items()}


def hybrid_search(
    query: str,
    top_k: int = 5,
    alpha: float = ALPHA_DEFAULT,
    pool_size: int = CANDIDATE_POOL_SIZE,
) -> list[tuple[str, float]]:
    """Гибридный поиск: взвешенное объединение нормализованных BM25 и vector скоров."""
    bm25_index, bm25_chunk_ids = bm25_search.load_index()
    bm25_raw = dict(bm25_search.search(bm25_index, bm25_chunk_ids, query, pool_size))

    vector_model = vector_search.load_model()
    vector_index, vector_chunk_ids = vector_search.load_index()
    vector_raw = dict(vector_search.search(vector_model, vector_index, vector_chunk_ids, query, pool_size))

    return combine_scores(bm25_raw, vector_raw, alpha=alpha, top_k=top_k)


def combine_scores(
    bm25_raw: dict[str, float],
    vector_raw: dict[str, float],
    alpha: float,
    top_k: int,
) -> list[tuple[str, float]]:
    bm25_norm = min_max_normalize(bm25_raw)
    vector_norm = min_max_normalize(vector_raw)

    candidate_ids = set(bm25_norm) | set(vector_norm)
    scored = [
        (chunk_id, alpha * bm25_norm.get(chunk_id, 0.0) + (1 - alpha) * vector_norm.get(chunk_id, 0.0))
        for chunk_id in candidate_ids
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]


def run_cli_query(query: str, top_k: int, alpha: float) -> None:
    chunk_lookup = build_chunk_lookup()
    list_groups = build_list_groups(load_chunks())

    results = hybrid_search(query, top_k=top_k, alpha=alpha)

    print(f"\n[Hybrid alpha={alpha}]")
    for rank, (chunk_id, score) in enumerate(results, start=1):
        chunk = chunk_lookup.get(chunk_id, {})
        print(f"{rank}. {chunk_id} (score={score:.3f}) {chunk_summary(chunk)}")

    chunk_ids = [chunk_id for chunk_id, _ in results]
    if any(chunk_lookup.get(cid, {}).get("type") == "list_item" for cid in chunk_ids):
        expanded = expand_results(chunk_ids, chunk_lookup, list_groups)
        print(f"\n{format_expanded_procedure(expanded)}")


def load_test_queries() -> list[dict]:
    return json.loads(TEST_QUERIES_PATH.read_text(encoding="utf-8"))


def format_markdown_results(results: list[tuple[str, float]], chunk_lookup: dict) -> str:
    lines = ["**Hybrid:**"]
    for rank, (chunk_id, score) in enumerate(results, start=1):
        chunk = chunk_lookup.get(chunk_id, {})
        lines.append(f"{rank}. `{chunk_id}` (score={score:.3f}) — {chunk_summary(chunk)}")
    return "\n".join(lines)


def run_all_test_queries(top_k: int = 5, alpha: float = ALPHA_DEFAULT) -> None:
    chunk_lookup = build_chunk_lookup()
    list_groups = build_list_groups(load_chunks())
    queries = load_test_queries()

    bm25_index, bm25_chunk_ids = bm25_search.load_index()
    vector_model = vector_search.load_model()
    vector_index, vector_chunk_ids = vector_search.load_index()

    sections = [f"# Результаты гибридного поиска (неделя 5, alpha={alpha})", ""]
    for q in queries:
        query_id = q["query_id"]
        query_text = q["query_text"]
        category = q["category"]
        note = q.get("note", "")

        bm25_raw = dict(bm25_search.search(bm25_index, bm25_chunk_ids, query_text, CANDIDATE_POOL_SIZE))
        vector_raw = dict(
            vector_search.search(vector_model, vector_index, vector_chunk_ids, query_text, CANDIDATE_POOL_SIZE)
        )
        hybrid_results = combine_scores(bm25_raw, vector_raw, alpha=alpha, top_k=top_k)

        sections.append(f"## {query_id} [{category}] {query_text}")
        if note:
            sections.append(f"_Примечание: {note}_")
        sections.append("")
        sections.append(format_markdown_results(hybrid_results, chunk_lookup))

        if category == "procedural":
            chunk_ids = [chunk_id for chunk_id, _ in hybrid_results]
            expanded = expand_results(chunk_ids, chunk_lookup, list_groups)
            sections.append("")
            sections.append(format_expanded_procedure(expanded))

        sections.append("")

    HYBRID_RESULTS_PATH.write_text("\n".join(sections), encoding="utf-8")
    print(f"Сохранено: {HYBRID_RESULTS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=ALPHA_DEFAULT)
    parser.add_argument("--run_all", action="store_true")
    args = parser.parse_args()

    if args.run_all:
        run_all_test_queries(args.top_k, args.alpha)
    elif args.query:
        run_cli_query(args.query, args.top_k, args.alpha)
    else:
        parser.error("укажите --query или --run_all")


if __name__ == "__main__":
    main()
