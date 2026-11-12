"""Подбор веса alpha для гибридного поиска (неделя 5).

Прогоняет BM25, Vector и Hybrid (при нескольких alpha) по размеченным
запросам (с непустым gold_chunk_ids) из Eval/test_queries.json и считает
Recall@5 — долю запросов, для которых хотя бы один gold_chunk_id попал
в топ-5 результатов метода.

Результат печатается в консоль и сохраняется в Eval/alpha_tuning.md.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import bm25_search
import vector_search
from hybrid_retrieval import combine_scores, CANDIDATE_POOL_SIZE

EVAL_DIR = Path(__file__).parent.parent / "Eval"
TEST_QUERIES_PATH = EVAL_DIR / "test_queries.json"
REPORT_PATH = EVAL_DIR / "alpha_tuning.md"

ALPHAS = [0.0, 0.3, 0.5, 0.7, 1.0]
TOP_K = 5


def recall_at_k(results: list[tuple[str, float]], gold_ids: list[str]) -> int:
    top_ids = {chunk_id for chunk_id, _ in results}
    return 1 if top_ids & set(gold_ids) else 0


def main() -> None:
    queries = json.loads(TEST_QUERIES_PATH.read_text(encoding="utf-8"))
    labeled = [q for q in queries if q.get("gold_chunk_ids")]
    print(f"Размечено запросов (с gold_chunk_ids): {len(labeled)} из {len(queries)}")

    bm25_index, bm25_chunk_ids = bm25_search.load_index()
    vector_model = vector_search.load_model()
    vector_index, vector_chunk_ids = vector_search.load_index()

    per_query_rows = []
    totals = {alpha: 0 for alpha in ALPHAS}

    for q in labeled:
        query_text = q["query_text"]
        gold_ids = q["gold_chunk_ids"]

        bm25_raw = dict(bm25_search.search(bm25_index, bm25_chunk_ids, query_text, CANDIDATE_POOL_SIZE))
        vector_raw = dict(
            vector_search.search(vector_model, vector_index, vector_chunk_ids, query_text, CANDIDATE_POOL_SIZE)
        )

        row = {"query_id": q["query_id"], "category": q["category"]}
        for alpha in ALPHAS:
            results = combine_scores(bm25_raw, vector_raw, alpha=alpha, top_k=TOP_K)
            hit = recall_at_k(results, gold_ids)
            row[f"alpha={alpha}"] = hit
            totals[alpha] += hit
        per_query_rows.append(row)

    n = len(labeled)
    lines = ["# Подбор alpha для гибридного поиска (неделя 5)", ""]
    lines.append(f"Размечено запросов: {n} из {len(queries)} (все категории, кроме trap).")
    lines.append("")
    lines.append(f"alpha=0.0 эквивалентен чистому vector-поиску, alpha=1.0 — чистому BM25.")
    lines.append("")
    lines.append("| query_id | category | " + " | ".join(f"alpha={a}" for a in ALPHAS) + " |")
    lines.append("|---|---|" + "---|" * len(ALPHAS))
    for row in per_query_rows:
        cells = " | ".join(str(row[f"alpha={a}"]) for a in ALPHAS)
        lines.append(f"| {row['query_id']} | {row['category']} | {cells} |")
    lines.append("")
    lines.append("## Recall@5 по alpha")
    lines.append("")
    lines.append("| alpha | Recall@5 |")
    lines.append("|---|---|")
    for alpha in ALPHAS:
        lines.append(f"| {alpha} | {totals[alpha]}/{n} = {totals[alpha] / n:.3f} |")

    best_alpha = max(ALPHAS, key=lambda a: totals[a])
    lines.append("")
    lines.append(f"**Лучший alpha по Recall@5 на размеченных запросах: {best_alpha}** (возможна ничья — см. таблицу).")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nСохранено: {REPORT_PATH}")
    for alpha in ALPHAS:
        print(f"alpha={alpha}: Recall@5 = {totals[alpha]}/{n} = {totals[alpha] / n:.3f}")


if __name__ == "__main__":
    main()
