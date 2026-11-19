"""Read-only вкладка метрик: читает Eval/metrics.json и Eval/metrics.md."""

import json
from pathlib import Path

import gradio as gr

BASE_DIR = Path(__file__).resolve().parents[1]
METRICS_JSON_PATH = BASE_DIR / "Eval" / "metrics.json"
METRICS_MD_PATH = BASE_DIR / "Eval" / "metrics.md"


def format_value(value, digits: int = 3) -> str:
    """Число метрики или прочерк, если оценка ещё не заполнена."""
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def format_rate(value) -> str:
    """Доля с процентом для читаемости."""
    if value is None:
        return "—"
    return f"{value:.3f} ({value * 100:.1f}%)"


def load_metrics() -> dict:
    """Прочитать метрики, не изменяя артефакты."""
    if not METRICS_JSON_PATH.exists():
        raise gr.Error(
            f"Не найден файл метрик: {METRICS_JSON_PATH}. "
            "Выполните python Eval/evaluate.py из корня репозитория."
        )
    try:
        return json.loads(METRICS_JSON_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise gr.Error(f"Не удалось прочитать {METRICS_JSON_PATH}: {exc}") from exc


def build_metrics_tab() -> None:
    """Нарисовать вкладку метрик внутри уже открытого gr.Tab."""
    metrics = load_metrics()
    retrieval = metrics.get("retrieval", {})
    generation = metrics.get("generation", {})
    top_k = metrics.get("top_k", {})
    abbrev = metrics.get("abbrev") or {}

    gr.Markdown(
        "# Метрики качества\n\n"
        "Данные читаются из `Eval/metrics.json` и `Eval/metrics.md` только для просмотра. "
        "Вкладка не запускает оценку и ничего не записывает в `Eval/`."
    )

    method_titles = {
        "bm25": "BM25 (alpha=1.0)",
        "vector": "Vector (alpha=0.0)",
        "hybrid": "Hybrid (alpha=0.7)",
    }
    retrieval_lines = [
        "## Поиск",
        "",
        "| Метод | Recall@5 | MRR@5 | Запросов |",
        "|---|---|---|---|",
    ]
    for method in ("bm25", "vector", "hybrid"):
        row = retrieval.get(method, {})
        retrieval_lines.append(
            f"| {method_titles[method]} | {format_value(row.get('recall_at_5'))} | "
            f"{format_value(row.get('mrr_at_5'))} | {row.get('n_queries', '—')} |"
        )
    gr.Markdown("\n".join(retrieval_lines))

    accuracy = generation.get("answer_accuracy_mean")
    hallucination = generation.get("hallucination_rate")
    accuracy_percent = None if accuracy is None else accuracy / 2 * 100
    generation_lines = [
        "## Генерация ответов",
        "",
        "| Метрика | Значение |",
        "|---|---|",
        f"| Отказов (refused) | {format_rate(generation.get('refusal_rate'))} |",
        f"| Корректных отказов на trap | {format_rate(generation.get('trap_refusal_correctness'))} |",
        f"| Answer Accuracy (среднее 0/1/2) | {format_value(accuracy)} |",
        f"| Answer Accuracy, % | {format_value(accuracy_percent, 1)} |",
        f"| Hallucination Rate | {format_rate(hallucination)} |",
        f"| Ответов в прогоне | {generation.get('n_answers', '—')} |",
    ]
    gr.Markdown("\n".join(generation_lines))

    if accuracy is None or hallucination is None:
        gr.Markdown(
            "⚠️ **Answer Accuracy и Hallucination Rate ожидают ручной оценки.** "
            "Заполните `Eval/manual_scores.json` и запустите `python Eval/evaluate.py`. "
            "Интерфейс не подставляет оценки автоматически."
        )

    top_k_lines = [
        "## Влияние top_k",
        "",
        "| top_k | Answer Accuracy | Hallucination Rate | Refusal Rate |",
        "|---|---|---|---|",
    ]
    for value in ("3", "5", "10"):
        row = top_k.get(value)
        if row is None:
            top_k_lines.append(f"| {value} | — | — | — |")
            continue
        top_k_lines.append(
            f"| {value} | {format_value(row.get('answer_accuracy_mean'))} | "
            f"{format_rate(row.get('hallucination_rate'))} | "
            f"{format_rate(row.get('refusal_rate'))} |"
        )
    gr.Markdown("\n".join(top_k_lines))

    abbrev_lines = [
        "## Раскрытие аббревиатур",
        "",
        "| Вариант | Recall@5 | MRR@5 |",
        "|---|---|---|",
        f"| Базовый запрос | {format_value(abbrev.get('baseline', {}).get('recall_at_5'))} | "
        f"{format_value(abbrev.get('baseline', {}).get('mrr_at_5'))} |",
        f"| С раскрытием аббревиатур | {format_value(abbrev.get('expanded', {}).get('recall_at_5'))} | "
        f"{format_value(abbrev.get('expanded', {}).get('mrr_at_5'))} |",
    ]
    gr.Markdown("\n".join(abbrev_lines))

    with gr.Accordion("Полный отчёт Eval/metrics.md", open=False):
        if METRICS_MD_PATH.exists():
            gr.Markdown(METRICS_MD_PATH.read_text(encoding="utf-8"))
        else:
            gr.Markdown(f"Файл `{METRICS_MD_PATH}` не найден.")
