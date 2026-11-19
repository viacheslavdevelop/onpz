"""Gradio-интерфейс RAG-поиска по технологическому регламенту ЭЛОУ-АВТ."""

import os

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import argparse
import json
import sys
import tempfile
import traceback
from dataclasses import asdict
from pathlib import Path

import gradio as gr

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(BASE_DIR / "Generation"))
sys.path.insert(0, str(BASE_DIR / "Retrieval"))

import llm_client
import rag_service
from metrics_view import build_metrics_tab
from runtime import get_runtime
from ui_components import (
    MAX_SOURCE_CARDS,
    build_assistant_content,
    compact_sources_line,
    format_llm_status,
    format_meta,
    render_source_card,
    to_chatbot_messages,
)

EVAL_QUERIES_PATH = BASE_DIR / "Eval" / "test_queries.json"

CATEGORY_TITLES = {
    "factual": "Фактологические",
    "procedural": "Процедурные",
    "synonym_abbreviation": "Синонимы и аббревиатуры",
    "numeric_tabular": "Числовые и табличные",
    "trap": "Ловушки (ответа нет)",
}

DISCLAIMER = (
    "> Прототип для учебной задачи. Не является руководством по эксплуатации. "
    "Ответы ограничены учебным корпусом."
)


def load_example_queries() -> list[dict]:
    """Прочитать примеры вопросов, не показывая эталонные ответы."""
    try:
        queries = json.loads(EVAL_QUERIES_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [query for query in queries if query.get("query_text")]


def category_choices(queries: list[dict]) -> list[tuple[str, str]]:
    """Русские подписи категорий в порядке тестового набора."""
    seen: list[str] = []
    for query in queries:
        category = query.get("category")
        if category and category not in seen:
            seen.append(category)
    return [
        (CATEGORY_TITLES.get(category, category), category)
        for category in seen
        if category in CATEGORY_TITLES
    ]


def example_choices(queries: list[dict], category: str) -> list[str]:
    return [query["query_text"] for query in queries if query.get("category") == category]


def hidden_card_updates() -> tuple[list, list]:
    return (
        [gr.update(visible=False) for _ in range(MAX_SOURCE_CARDS)],
        ["" for _ in range(MAX_SOURCE_CARDS)],
    )


def card_updates(sources: list[dict]) -> tuple[list, list]:
    """Обновления десяти заранее созданных карточек источников."""
    accordion_updates: list = []
    body_updates: list[str] = []
    for number in range(1, MAX_SOURCE_CARDS + 1):
        if number <= len(sources):
            label, body = render_source_card(sources[number - 1])
            accordion_updates.append(gr.update(label=label, visible=True, open=False))
            body_updates.append(body)
        else:
            accordion_updates.append(gr.update(visible=False))
            body_updates.append("")
    return accordion_updates, body_updates


def export_history(history: list[dict]) -> str | None:
    """Сохранить историю во временный JSON вне репозитория."""
    if not history:
        return None
    path = Path(tempfile.gettempdir()) / f"onpz_rag_history_{os.getpid()}.json"
    path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def build_demo() -> gr.Blocks:
    queries = load_example_queries()
    categories = category_choices(queries)
    default_category = categories[0][1] if categories else None
    default_questions = example_choices(queries, default_category) if default_category else []

    with gr.Blocks(title="RAG ЭЛОУ-АВТ") as demo:
        history_state = gr.State([])
        llm_state = gr.State({})

        gr.Markdown(
            "# RAG-поиск по регламенту ЭЛОУ-АВТ\n"
            "10 документов · 206 чанков · гибридный поиск BM25 + vector · "
            "ответы со ссылками `[n]`.\n\n"
            "История пуста. Задайте вопрос в поле под чатом или выберите пример слева."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Параметры поиска")
                top_k = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="top_k",
                    info="Сколько фрагментов подать в LLM.",
                )
                alpha = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.05,
                    label="alpha",
                    info="Вес BM25 в гибридном поиске.",
                )
                method = gr.Radio(
                    choices=["hybrid", "bm25", "vector"],
                    value="hybrid",
                    label="Метод поиска",
                    info="Гибридный, только BM25 или только векторный.",
                )
                expand_abbrev = gr.Checkbox(
                    label="Раскрывать аббревиатуры",
                    value=False,
                    info="Дописывать расшифровки в поисковый запрос.",
                )
                show_context = gr.Checkbox(
                    label="Показывать контекст, отправленный в LLM",
                    value=False,
                )

                gr.Markdown("### Статус LLM")
                llm_status = gr.Markdown("Проверка статуса LLM…")
                check_button = gr.Button("Проверить соединение")

                gr.Markdown("### Примеры вопросов")
                category = gr.Dropdown(
                    choices=categories,
                    value=default_category,
                    label="Категория",
                )
                example = gr.Dropdown(
                    choices=default_questions,
                    value=default_questions[0] if default_questions else None,
                    label="Вопрос",
                )
                trap_note = gr.Markdown(
                    "В регламенте нет ответа, ожидается отказ."
                    if default_category == "trap"
                    else "",
                    visible=default_category == "trap",
                )
                example_button = gr.Button("Задать вопрос")

                clear_button = gr.Button("Очистить чат")
                download_button = gr.DownloadButton(
                    "Скачать историю",
                    value=export_history,
                    inputs=history_state,
                )

                gr.Markdown(DISCLAIMER)

            with gr.Column(scale=4):
                with gr.Tabs():
                    with gr.Tab("Чат"):
                        chatbot = gr.Chatbot(
                            type="messages",
                            height=520,
                            show_copy_button=True,
                            sanitize_html=True,
                            label="Диалог",
                        )
                        with gr.Row():
                            question_box = gr.Textbox(
                                placeholder=(
                                    "Например: какое давление поддерживается "
                                    "в колонне К-1?"
                                ),
                                label="Вопрос",
                                scale=4,
                            )
                            send_button = gr.Button(
                                "Отправить", variant="primary", scale=1
                            )

                        meta_md = gr.Markdown()
                        gr.Markdown(
                            "Номера `[n]` в ответе соответствуют источникам ниже."
                        )

                        source_accords = []
                        source_bodies = []
                        for number in range(1, MAX_SOURCE_CARDS + 1):
                            with gr.Accordion(
                                f"{number}. Источник", open=False, visible=False
                            ) as accordion:
                                body = gr.Markdown()
                            source_accords.append(accordion)
                            source_bodies.append(body)

                        with gr.Accordion(
                            "Контекст, отправленный в LLM",
                            open=False,
                            visible=False,
                        ) as context_acc:
                            context_code = gr.Code(
                                language=None,
                                interactive=False,
                                lines=12,
                                label="Блоки контекста",
                            )

                    with gr.Tab("Метрики"):
                        build_metrics_tab()

        handler_outputs = [
            chatbot,
            history_state,
            meta_md,
            context_acc,
            context_code,
            *source_accords,
            *source_bodies,
            llm_status,
            llm_state,
        ]

        def check_llm() -> tuple[str, dict]:
            status = llm_client.probe(timeout_s=1.5)
            return format_llm_status(status), status

        def update_examples(category_value: str):
            question_values = example_choices(queries, category_value)
            is_trap = category_value == "trap"
            return (
                gr.update(
                    choices=question_values,
                    value=question_values[0] if question_values else None,
                ),
                gr.update(
                    value="В регламенте нет ответа, ожидается отказ."
                    if is_trap
                    else "",
                    visible=is_trap,
                ),
            )

        def handle_question(
            question: str,
            history: list[dict],
            top_k_value: int,
            alpha_value: float,
            method_value: str,
            expand_value: bool,
            show_context_value: bool,
            current_llm_state: dict,
        ) -> tuple:
            history = list(history or [])
            question = (question or "").strip()

            if not question:
                gr.Warning("Введите вопрос.")
                return (
                    gr.update(),
                    history,
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    *([gr.update()] * (MAX_SOURCE_CARDS * 2)),
                    gr.update(),
                    current_llm_state,
                )

            try:
                retrieval_runtime = get_runtime()
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                raise gr.Error(
                    "Не удалось загрузить поисковые индексы. Проверьте файлы "
                    "Chunks/bm25_index.pkl, Chunks/faiss_index.bin, "
                    "Chunks/faiss_chunk_ids.json и модель эмбеддингов "
                    "(см. README.md)."
                ) from exc

            history.append({"role": "user", "content": question})

            try:
                result = rag_service.ask(
                    question,
                    top_k=int(top_k_value),
                    alpha=float(alpha_value),
                    expand_abbrev=bool(expand_value),
                    method=method_value,
                    runtime=retrieval_runtime,
                )
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                message = (
                    "⚠️ Не удалось получить ответ от LLM.\n\n"
                    "Проверьте, что llama-server запущен (см. README.md, раздел "
                    "«Генерация ответа»), и нажмите «Проверить соединение»."
                )
                meta = {
                    "latency_s": 0.0,
                    "retrieval_latency_s": 0.0,
                    "generation_latency_s": 0.0,
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                    "settings": {
                        "top_k": int(top_k_value),
                        "alpha": float(alpha_value),
                        "expand_abbrev": bool(expand_value),
                        "think": False,
                        "method": method_value,
                        "model": llm_client.MODEL,
                    },
                    "context_truncated": False,
                    "omitted_sources": 0,
                    "refused": False,
                }
                history.append(
                    {
                        "role": "assistant",
                        "content": message,
                        "answer": "",
                        "sources": [],
                        "context": "",
                        "meta": meta,
                        "error": str(exc),
                    }
                )
                accordion_updates, body_updates = hidden_card_updates()
                status = llm_client.probe(timeout_s=1.0)
                if status.get("ok"):
                    status = {
                        "ok": False,
                        "model": None,
                        "error": "ошибка генерации, подробности в чате",
                    }
                gr.Warning(
                    "LLM недоступна. Запустите llama-server (см. README.md)."
                )
                return (
                    to_chatbot_messages(history),
                    history,
                    "⚠️ Ответ не получен: LLM недоступна.",
                    gr.update(visible=False),
                    gr.update(value=""),
                    *accordion_updates,
                    *body_updates,
                    format_llm_status(status),
                    status,
                )

            sources = [asdict(source) for source in result.sources]
            meta = {
                "latency_s": result.latency_s,
                "retrieval_latency_s": result.retrieval_latency_s,
                "generation_latency_s": result.generation_latency_s,
                "usage": result.usage,
                "settings": result.settings,
                "context_truncated": result.context_truncated,
                "omitted_sources": result.omitted_sources,
                "refused": result.refused,
            }
            content = build_assistant_content(result.answer, result.refused, sources)
            history.append(
                {
                    "role": "assistant",
                    "content": content,
                    "answer": result.answer,
                    "sources": sources,
                    "context": result.context,
                    "meta": meta,
                    "error": None,
                }
            )

            accordion_updates, body_updates = card_updates(sources)
            status = {
                "ok": True,
                "model": result.settings.get("model") or llm_client.MODEL,
                "error": None,
            }
            return (
                to_chatbot_messages(history),
                history,
                format_meta(meta, len(sources), result.answer),
                gr.update(visible=bool(show_context_value)),
                gr.update(value=result.context or "Контекст пуст: поиск не вернул фрагментов."),
                *accordion_updates,
                *body_updates,
                format_llm_status(status),
                status,
            )

        def clear_chat() -> tuple:
            accordion_updates = [
                gr.update(visible=False, label=f"{number}. Источник")
                for number in range(1, MAX_SOURCE_CARDS + 1)
            ]
            return (
                [],
                [],
                "",
                gr.update(visible=False),
                gr.update(value=""),
                *accordion_updates,
                *["" for _ in range(MAX_SOURCE_CARDS)],
            )

        handler_inputs = [
            question_box,
            history_state,
            top_k,
            alpha,
            method,
            expand_abbrev,
            show_context,
            llm_state,
        ]

        question_box.submit(
            fn=handle_question, inputs=handler_inputs, outputs=handler_outputs
        )
        send_button.click(
            fn=handle_question, inputs=handler_inputs, outputs=handler_outputs
        )
        example_button.click(
            fn=handle_question,
            inputs=[example, *handler_inputs[1:]],
            outputs=handler_outputs,
        )

        clear_button.click(
            fn=clear_chat,
            inputs=None,
            outputs=[
                chatbot,
                history_state,
                meta_md,
                context_acc,
                context_code,
                *source_accords,
                *source_bodies,
            ],
        )

        category.change(
            fn=update_examples,
            inputs=category,
            outputs=[example, trap_note],
        )
        show_context.change(
            fn=lambda show: gr.update(visible=bool(show)),
            inputs=show_context,
            outputs=context_acc,
        )
        check_button.click(
            fn=check_llm, inputs=None, outputs=[llm_status, llm_state]
        )
        demo.load(fn=check_llm, inputs=None, outputs=[llm_status, llm_state])

    return demo


def main() -> int:
    parser = argparse.ArgumentParser(description="Веб-интерфейс RAG-поиска ЭЛОУ-АВТ")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if args.host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("Допустим только локальный хост (127.0.0.1).")

    demo = build_demo()
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        inbrowser=not args.no_browser,
        show_api=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
