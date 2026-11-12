"""Расчёт метрик поиска и сводка результатов генерации (неделя 7)."""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(BASE_DIR / "Retrieval"))

import bm25_search
import vector_search
from hybrid_retrieval import ALPHA_DEFAULT, CANDIDATE_POOL_SIZE, combine_scores
from query_expansion import expand_query

EVAL_DIR = BASE_DIR / "Eval"
TEST_QUERIES_PATH = EVAL_DIR / "test_queries.json"
ANSWERS_PATH = EVAL_DIR / "answers.jsonl"
MANUAL_SCORES_PATH = EVAL_DIR / "manual_scores.json"
METRICS_JSON_PATH = EVAL_DIR / "metrics.json"
METRICS_MD_PATH = EVAL_DIR / "metrics.md"

TOP_K = 5
DIGITS = 3
TRAP_CATEGORY = "trap"


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def first_gold_rank(results: list[tuple[str, float]], gold_ids: list[str]) -> int | None:
    gold_set = set(gold_ids)
    for rank, (chunk_id, _score) in enumerate(results, start=1):
        if chunk_id in gold_set:
            return rank
    return None


def rounded(value: float | None) -> float | None:
    return None if value is None else round(value, DIGITS)


def evaluate_retrieval(
    queries: list[dict],
    *,
    bm25_index,
    bm25_chunk_ids: list[str],
    vector_model,
    vector_index,
    vector_chunk_ids: list[str],
    expand_abbrev: bool = False,
) -> dict:
    """Посчитать Recall@5 и MRR@5 для BM25, vector и hybrid на размеченных запросах."""
    per_method = {
        "bm25": {"hits": 0, "reciprocal_ranks": []},
        "vector": {"hits": 0, "reciprocal_ranks": []},
        "hybrid": {"hits": 0, "reciprocal_ranks": []},
    }

    for query in queries:
        query_text = expand_query(query["query_text"]) if expand_abbrev else query["query_text"]
        gold_ids = query["gold_chunk_ids"]

        bm25_raw = bm25_search.search(bm25_index, bm25_chunk_ids, query_text, CANDIDATE_POOL_SIZE)
        vector_raw = vector_search.search(
            vector_model, vector_index, vector_chunk_ids, query_text, CANDIDATE_POOL_SIZE
        )
        hybrid_results = combine_scores(
            dict(bm25_raw), dict(vector_raw), alpha=ALPHA_DEFAULT, top_k=TOP_K
        )

        method_results = {
            "bm25": bm25_raw[:TOP_K],
            "vector": vector_raw[:TOP_K],
            "hybrid": hybrid_results,
        }
        for method, results in method_results.items():
            rank = first_gold_rank(results, gold_ids)
            if rank is not None:
                per_method[method]["hits"] += 1
                per_method[method]["reciprocal_ranks"].append(1.0 / rank)
            else:
                per_method[method]["reciprocal_ranks"].append(0.0)

    n_queries = len(queries)
    summary = {}
    for method, values in per_method.items():
        recall = values["hits"] / n_queries if n_queries else None
        mrr = (
            sum(values["reciprocal_ranks"]) / len(values["reciprocal_ranks"])
            if values["reciprocal_ranks"]
            else None
        )
        summary[method] = {
            "recall_at_5": rounded(recall),
            "mrr_at_5": rounded(mrr),
            "n_queries": n_queries,
        }
    return summary


def scores_by_query(manual_scores: list[dict] | None) -> dict[str, dict]:
    if not manual_scores:
        return {}
    return {record["query_id"]: record for record in manual_scores}


def has_complete_scores(manual_scores: list[dict] | None, query_ids: set[str]) -> bool:
    records = scores_by_query(manual_scores)
    if not records:
        return False
    for query_id in query_ids:
        record = records.get(query_id)
        if record is None:
            return False
        if record.get("answer_accuracy") is None or record.get("hallucination") is None:
            return False
    return True


def generation_summary(
    answers: list[dict] | None,
    manual_scores: list[dict] | None,
) -> dict:
    if not answers:
        return {
            "refusal_rate": None,
            "answer_accuracy_mean": None,
            "hallucination_rate": None,
            "n_answers": 0,
            "trap_refusal_correctness": None,
        }

    refused = [record.get("refused") is True for record in answers]
    trap_answers = [record for record in answers if record.get("category") == TRAP_CATEGORY]
    trap_refused = [record.get("refused") is True for record in trap_answers]

    query_ids = {record["query_id"] for record in answers}
    complete = has_complete_scores(manual_scores, query_ids)
    accuracy_mean = None
    hallucination_rate = None
    if complete:
        scores = scores_by_query(manual_scores)
        accuracy_mean = sum(
            scores[record["query_id"]]["answer_accuracy"] for record in answers
        ) / len(answers)
        hallucination_rate = sum(
            1 for record in answers if scores[record["query_id"]]["hallucination"] is True
        ) / len(answers)

    return {
        "refusal_rate": rounded(sum(refused) / len(refused)),
        "answer_accuracy_mean": rounded(accuracy_mean),
        "hallucination_rate": rounded(hallucination_rate),
        "n_answers": len(answers),
        "trap_refusal_correctness": rounded(sum(trap_refused) / len(trap_refused))
        if trap_refused
        else None,
    }


def top_k_comparison() -> dict:
    variants = [
        ("5", "baseline", EVAL_DIR / "answers.jsonl", EVAL_DIR / "manual_scores.json"),
        ("3", "top3", EVAL_DIR / "answers_top3.jsonl", EVAL_DIR / "manual_scores_top3.json"),
        ("10", "top10", EVAL_DIR / "answers_top10.jsonl", EVAL_DIR / "manual_scores_top10.json"),
    ]
    comparison = {}
    for top_k, tag, answers_path, scores_path in variants:
        answers = load_jsonl(answers_path)
        if not answers:
            continue
        summary = generation_summary(answers, load_json(scores_path))
        comparison[top_k] = {
            "tag": tag,
            "answer_accuracy_mean": summary["answer_accuracy_mean"],
            "hallucination_rate": summary["hallucination_rate"],
            "refusal_rate": summary["refusal_rate"],
        }
    return comparison


def format_value(value, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}{suffix}"
    return f"{value}{suffix}"


def format_percent_value(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.1f}"


def format_rate(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.3f} ({value * 100:.1f}%)"


def build_analysis(
    answers: list[dict] | None,
    manual_scores: list[dict] | None,
    queries: list[dict],
) -> list[str]:
    if not answers:
        return ["Ответы не найдены: сначала выполните `python Generation/generate.py --run_all`."]

    scores = scores_by_query(manual_scores)
    complete = has_complete_scores(manual_scores, {record["query_id"] for record in answers})
    lines = []

    if complete:
        for record in sorted(answers, key=lambda item: item["query_id"]):
            score = scores[record["query_id"]]
            if score.get("answer_accuracy") == 2 and score.get("hallucination") is not True:
                continue
            comment = score.get("comment") or "без комментария"
            lines.append(
                f"- **{record['query_id']}** [{record['category']}]: "
                f"answer_accuracy={score.get('answer_accuracy')}, "
                f"hallucination={score.get('hallucination')}. {comment}"
            )
        if not lines:
            lines.append("Ошибок, отмеченных при ручной оценке, нет.")
        return lines

    gold_by_query = {query["query_id"]: query.get("gold_chunk_ids", []) for query in queries}
    for record in sorted(answers, key=lambda item: item["query_id"]):
        issues = []
        if record.get("error"):
            issues.append(f"ошибка генерации: {record['error']}")
        if not record.get("answer"):
            issues.append("пустой ответ")
        if record.get("category") != TRAP_CATEGORY and record.get("refused"):
            issues.append("необоснованный отказ")
        gold_ids = gold_by_query.get(record["query_id"], [])
        if gold_ids:
            retrieved = {
                chunk["chunk_id"] for chunk in record.get("retrieval", {}).get("chunks", [])
            }
            if not retrieved & set(gold_ids):
                issues.append("retrieval miss: ни один gold-чанк не попал в top-k")
        if issues:
            lines.append(f"- **{record['query_id']}** [{record['category']}]: " + "; ".join(issues))

    if not lines:
        lines.append("Автоматических аномалий не обнаружено.")
    lines.append("")
    lines.append(
        "Answer Accuracy и Hallucination Rate не рассчитаны: заполните "
        "`Eval/manual_scores.json` и запустите скрипт повторно."
    )
    return lines


def build_conclusions(
    retrieval: dict,
    generation: dict,
    top_k: dict,
    abbrev: dict | None,
) -> list[str]:
    hybrid = retrieval["hybrid"]
    bm25 = retrieval["bm25"]
    vector = retrieval["vector"]

    conclusions = [
        f"Гибридный поиск при alpha={ALPHA_DEFAULT} показал Recall@5 = {hybrid['recall_at_5']:.3f} "
        f"и MRR@5 = {hybrid['mrr_at_5']:.3f} на {hybrid['n_queries']} размеченных запросах; "
        f"BM25 — {bm25['recall_at_5']:.3f} / {bm25['mrr_at_5']:.3f}, "
        f"vector — {vector['recall_at_5']:.3f} / {vector['mrr_at_5']:.3f}."
    ]

    if hybrid["recall_at_5"] >= bm25["recall_at_5"] and hybrid["recall_at_5"] >= vector["recall_at_5"]:
        conclusions.append("Гибрид не уступает каждому из методов по Recall@5 на размеченной выборке.")
    else:
        conclusions.append(
            "Гибрид уступает одному из базовых методов по Recall@5; alpha не подгонялся, "
            "результат зафиксирован как есть."
        )

    if generation["trap_refusal_correctness"] is not None:
        conclusions.append(
            "Доля отказов на trap-запросах: "
            f"{generation['trap_refusal_correctness']:.3f}. "
            "Отказы проверяют устойчивость модели к вопросам вне регламента."
        )

    if generation["answer_accuracy_mean"] is None:
        conclusions.append(
            "Метрики качества генерации ожидают ручной оценки; выводы по точности ответов "
            "и галлюцинациям будут добавлены после заполнения manual_scores.json."
        )
    else:
        accuracy_percent = generation["answer_accuracy_mean"] / 2 * 100
        conclusions.append(
            f"Answer Accuracy = {generation['answer_accuracy_mean']:.3f} "
            f"({accuracy_percent:.1f}%), Hallucination Rate = {generation['hallucination_rate']:.3f}."
        )

    if abbrev and abbrev.get("expanded"):
        baseline = abbrev["baseline"]
        expanded = abbrev["expanded"]
        conclusions.append(
            "Раскрытие аббревиатур изменило Recall@5 с "
            f"{baseline['recall_at_5']:.3f} до {expanded['recall_at_5']:.3f}, "
            f"MRR@5 — с {baseline['mrr_at_5']:.3f} до {expanded['mrr_at_5']:.3f}."
        )

    if len(conclusions) < 3:
        conclusions.append(
            "Ограничение прототипа: используется фиксированный top_k без переранжирования, "
            "а размер выборки (25 запросов) мал для статистически устойчивых выводов."
        )
    return conclusions[:5]


def build_metrics_markdown(
    retrieval: dict,
    generation: dict,
    top_k: dict,
    abbrev: dict | None,
    analysis: list[str],
    conclusions: list[str],
    answers: list[dict] | None,
) -> str:
    lines = ["# Оценка качества (неделя 7)", ""]

    lines.extend(
        [
            "## 1. Поиск: BM25 vs Vector vs Hybrid",
            "",
            "| Метод | Recall@5 | MRR@5 | Запросов |",
            "|---|---|---|---|",
        ]
    )
    method_titles = {
        "bm25": "BM25 (alpha=1.0)",
        "vector": "Vector (alpha=0.0)",
        "hybrid": f"Hybrid (alpha={ALPHA_DEFAULT})",
    }
    for method in ("bm25", "vector", "hybrid"):
        row = retrieval[method]
        lines.append(
            f"| {method_titles[method]} | {format_value(row['recall_at_5'])} | "
            f"{format_value(row['mrr_at_5'])} | {row['n_queries']} |"
        )
    lines.append("")

    lines.extend(
        [
            "## 2. Генерация ответов (T-lite-it-2.1, Q4_K_M, top_k=5)",
            "",
            "| Метрика | Значение |",
            "|---|---|",
        ]
    )
    n_answers = generation["n_answers"]
    refused_count = 0
    if answers:
        refused_count = sum(1 for record in answers if record.get("refused") is True)
    lines.append(f"| Отказов (refused) | {refused_count} из {n_answers} |")
    lines.append(
        "| Корректных отказов на trap (T1–T3) | "
        f"{format_rate(generation['trap_refusal_correctness'])} |"
    )
    lines.append(
        "| Answer Accuracy (среднее 0/1/2) | "
        f"{format_value(generation['answer_accuracy_mean'])} |"
    )
    accuracy_percent = (
        None
        if generation["answer_accuracy_mean"] is None
        else generation["answer_accuracy_mean"] / 2 * 100
    )
    lines.append(f"| Answer Accuracy, % | {format_percent_value(accuracy_percent)} |")
    lines.append(f"| Hallucination Rate | {format_rate(generation['hallucination_rate'])} |")
    lines.append("")
    lines.extend(
        [
            "**Правила ручной оценки (`Eval/manual_scores.json`):**",
            "",
            "- `answer_accuracy`: `2` — верно и полно, `1` — частично верно или неполно, "
            "`0` — неверно либо необоснованный отказ; для trap-запросов корректный отказ = `2`.",
            "- `hallucination`: `true`, если в ответе есть факт, число или единица, "
            "которых нет в переданном контексте.",
            "- `comment` обязателен при `answer_accuracy != 2` или `hallucination = true`.",
            "",
        ]
    )
    if generation["answer_accuracy_mean"] is None or generation["hallucination_rate"] is None:
        lines.append("Требуется ручная оценка: заполните `Eval/manual_scores.json`.")
        lines.append("")

    lines.extend(
        [
            "## 3. Влияние top_k (3 vs 5 vs 10)",
            "",
            "| top_k | Answer Accuracy | Hallucination Rate |",
            "|---|---|---|",
        ]
    )
    if top_k:
        for value in ("3", "5", "10"):
            if value not in top_k:
                continue
            row = top_k[value]
            lines.append(
                f"| {value} | {format_value(row['answer_accuracy_mean'])} | "
                f"{format_rate(row['hallucination_rate'])} |"
            )
    else:
        lines.append("| — | — | — |")
    lines.append("")
    missing_top_k = [value for value in ("3", "10") if value not in top_k]
    if missing_top_k:
        lines.append(
            "Прогоны для top_k=" + ", ".join(missing_top_k) + " не найдены; "
            "для сравнения выполните команды из раздела 9 плана."
        )
        lines.append("")
    if any(row["answer_accuracy_mean"] is None for row in top_k.values()):
        lines.append("Для вариантов top_k требуется ручная оценка ответов.")
        lines.append("")

    lines.extend(
        [
            "## 4. Раскрытие аббревиатур (категория 3)",
            "",
            "| Вариант | Recall@5 | MRR@5 |",
            "|---|---|---|",
        ]
    )
    if abbrev:
        lines.append(
            f"| Базовый запрос | {format_value(abbrev['baseline']['recall_at_5'])} | "
            f"{format_value(abbrev['baseline']['mrr_at_5'])} |"
        )
        expanded = abbrev.get("expanded")
        lines.append(
            f"| С раскрытием аббревиатур | {format_value(expanded['recall_at_5'] if expanded else None)} | "
            f"{format_value(expanded['mrr_at_5'] if expanded else None)} |"
        )
    else:
        lines.append("| Базовый запрос | — | — |")
        lines.append("| С раскрытием аббревиатур | — | — |")
    lines.append("")
    if not abbrev or not abbrev.get("expanded"):
        lines.append(
            "Прогон `answers_abbrev.jsonl` не найден; выполните "
            "`python Generation/generate.py --run_all --expand_abbrev --tag abbrev`."
        )
        lines.append("")

    lines.extend(["## 5. Анализ ошибок", ""])
    lines.extend(analysis)
    lines.append("")

    lines.extend(["## 6. Выводы", ""])
    lines.extend(f"{index}. {text}" for index, text in enumerate(conclusions, start=1))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    queries = json.loads(TEST_QUERIES_PATH.read_text(encoding="utf-8"))
    labeled = sorted(
        (query for query in queries if query.get("gold_chunk_ids")),
        key=lambda query: query["query_id"],
    )

    bm25_index, bm25_chunk_ids = bm25_search.load_index()
    vector_model = vector_search.load_model()
    vector_index, vector_chunk_ids = vector_search.load_index()

    retrieval = evaluate_retrieval(
        labeled,
        bm25_index=bm25_index,
        bm25_chunk_ids=bm25_chunk_ids,
        vector_model=vector_model,
        vector_index=vector_index,
        vector_chunk_ids=vector_chunk_ids,
    )

    answers = load_jsonl(ANSWERS_PATH)
    manual_scores = load_json(MANUAL_SCORES_PATH)
    generation = generation_summary(answers, manual_scores)
    top_k = top_k_comparison()

    abbrev = None
    if (EVAL_DIR / "answers_abbrev.jsonl").exists():
        expanded_retrieval = evaluate_retrieval(
            labeled,
            bm25_index=bm25_index,
            bm25_chunk_ids=bm25_chunk_ids,
            vector_model=vector_model,
            vector_index=vector_index,
            vector_chunk_ids=vector_chunk_ids,
            expand_abbrev=True,
        )
        abbrev = {
            "baseline": {
                "recall_at_5": retrieval["hybrid"]["recall_at_5"],
                "mrr_at_5": retrieval["hybrid"]["mrr_at_5"],
            },
            "expanded": {
                "recall_at_5": expanded_retrieval["hybrid"]["recall_at_5"],
                "mrr_at_5": expanded_retrieval["hybrid"]["mrr_at_5"],
            },
        }

    analysis = build_analysis(answers, manual_scores, queries)
    conclusions = build_conclusions(retrieval, generation, top_k, abbrev)

    metrics = {
        "retrieval": retrieval,
        "generation": generation,
        "top_k": top_k,
        "abbrev": abbrev,
    }
    METRICS_JSON_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    METRICS_MD_PATH.write_text(
        build_metrics_markdown(
            retrieval, generation, top_k, abbrev, analysis, conclusions, answers
        ),
        encoding="utf-8",
    )

    print("Поиск (размеченных запросов: %d)" % len(labeled))
    print("Метод            Recall@5   MRR@5")
    for method in ("bm25", "vector", "hybrid"):
        row = retrieval[method]
        print(f"{method:<16} {row['recall_at_5']:.3f}      {row['mrr_at_5']:.3f}")

    print()
    print("Генерация:")
    print(f"  отказов: {generation['refusal_rate'] if generation['refusal_rate'] is not None else '—'}")
    print(f"  trap refusal correctness: {generation['trap_refusal_correctness']}")
    print(f"  answer accuracy mean: {generation['answer_accuracy_mean']}")
    print(f"  hallucination rate: {generation['hallucination_rate']}")

    print()
    print(f"Сохранено: {METRICS_JSON_PATH}")
    print(f"Сохранено: {METRICS_MD_PATH}")


if __name__ == "__main__":
    main()
