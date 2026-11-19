# ПЛАН РАБОТ ДЛЯ ИИ-АГЕНТА: неделя 8 (веб-интерфейс RAG-системы на Gradio)

> Исполняемое задание для ИИ-агента. Прочитать целиком перед началом.
> Стек, форматы данных и границы этапа зафиксированы. Агент не пересматривает их
> без явного указания человека. Если состояние репозитория противоречит плану —
> остановиться, описать расхождение и задать вопрос, а не «додумывать».
>
> Дата: 19.11.2026. Основание: `MES_ТЗ_2_на_RAG_подходы_на_примере_АВТ.docx`,
> текущий `README.md`, код и артефакты репозитория.

---

## 0. Анализ текущего состояния

Курсовая: RAG-поиск по учебному технологическому регламенту ЭЛОУ-АВТ.
Репозиторий: `C:\My\University\onpz` (Windows, PowerShell, Python 3.12).
Ветка: `main`, рабочее дерево на момент анализа чистое.

### 0.1. Что уже готово (не переделывать)

| Компонент | Путь | Состояние |
|---|---|---|
| Корпус | `Processed data/00…09_*.md` | 10 markdown-документов |
| Чанкинг | `Utils/chunking.py` → `Chunks/chunks_v2_by_fact.jsonl` | 206 чанков V2 `by_fact` |
| BM25 | `Retrieval/bm25_search.py`, `Chunks/bm25_index.pkl` | BM25Okapi + pymorphy3 |
| Векторный поиск | `Retrieval/vector_search.py`, `Chunks/faiss_index.bin` | rubert-tiny-turbo + FAISS `IndexFlatIP` |
| Гибрид | `Retrieval/hybrid_retrieval.py` | `alpha = 0.7`, пул кандидатов 20 |
| Подбор alpha | `Retrieval/alpha_tuning.py` → `Eval/alpha_tuning.md` | Recall@5 = 1.000 на 22 запросах |
| Тестовые запросы | `Eval/test_queries.json` | 25 запросов, 5 категорий, разметка полная |
| Генерация | `Generation/*`, `Eval/answers*.jsonl` | T-lite-it-2.1 8B Q4_K_M через llama-server |
| Оценка | `Eval/evaluate.py`, `Eval/metrics.json`, `Eval/metrics.md` | retrieval-метрики готовы |

### 0.2. Что является следующим этапом

По ТЗ и `README.md` недели 1–7 закрыты, недели 8–10 — нет.
**Следующий этап — неделя 8: веб-интерфейс (чат + отображение источников).**
Результат этапа: локальное демо-приложение, которое принимает вопрос на русском,
показывает ответ LLM и список источников с номерами `[n]`, метаданными чанков
и оценками релевантности.

### 0.3. Найденные технические разрывы, которые нужно закрыть по ходу этапа

| № | Проблема | Почему важна для UI | Решение в плане |
|---|---|---|---|
| 1 | `prompt.build_context()` обрезает контекст по 9000 символов, а `prompt.format_sources()` перечисляет **все** результаты top_k | В UI номера источников могут не совпасть с `[n]` в ответе; пользователь увидит источник, которого модель не видела | T1: вернуть структурированные блоки контекста и строить источники по ним |
| 2 | `hybrid_retrieval.hybrid_search()` при каждом вызове заново читает BM25/FAISS и загружает `SentenceTransformer` | Каждый вопрос в UI будет загружать модель эмбеддингов — секунды/десятки секунд на ровном месте | T2: кэш ресурсов `Retrieval/runtime.py` |
| 3 | `llm_client.healthcheck()` использует `LLM_TIMEOUT=180` и ссылается на удалённый старый план | Проверка статуса в UI может «зависнуть» на 3 минуты; сообщение об ошибке ведёт в никуда | T4: быстрый `probe(timeout_s=2)` + актуальная ссылка на README |
| 4 | `gradio` отсутствует в окружении и в `requirements.txt` | Приложение не запустится | T5/T7: установка, проверка совместимости, точный пин версии |
| 5 | `Eval/manual_scores.json`: у всех 25 ответов `answer_accuracy = null`, `hallucination = null`; `manual_scores_top3.json` и `manual_scores_top10.json` отсутствуют | `Answer Accuracy` и `Hallucination Rate` в `metrics.json` — `null`; сравнение top_k 3/5/10 неполное | **Не выдумывать оценки.** UI показывает `—` и предупреждение; заполняет человек (см. §9) |
| 6 | `README.md` ссылался на `PLAN_AI_AGENT.md` как на план недель 6–7, самого файла не было | Навигация по репозиторию сломана | Исправлено при подготовке плана: файл восстановлен, ссылка в README обновлена; T7 — финальная сверка |
| 7 | `llm_client` не умеет быстро проверять доступность без исключения | UI должен показывать статус, а не падать | T4: `probe()` |

### 0.4. Зафиксированное решение по интерфейсу

Выбран **Gradio** (по указанию человека; ТЗ допускает Gradio и Streamlit):

* `gr.Blocks` + `gr.Chatbot(type="messages")` дают полноценный чат без ручной вёрстки;
* `gr.State` хранит историю и источники в рамках сессии;
* `gr.Accordion` (фиксированный набор карточек) раскрывает детали источника;
* `gr.Tab` позволяет держать чат и read-only метрики в одном процессе;
* кэш индексов — обычный `functools.lru_cache` в `Retrieval/runtime.py`,
  без привязки к UI-фреймворку;
* запуск — один процесс `python App/app.py`, без отдельного сервера.

Сервисный слой T1–T4 (контекст, runtime, `rag_service`, `probe`) от выбора UI
не зависит и сохраняется без изменений.

Ограничение Gradio, которое влияет на дизайн: компоненты нельзя создавать
динамически по числу источников. Поэтому заранее создаётся **10 карточек-аккордеонов**
(максимум `top_k = 10`), лишние скрываются через `visible=False`. Подробные источники
показываются для **последнего ответа**; в `gr.State` сохраняются источники всех ответов,
а в тексте каждого ответа остаётся компактная строка «Источники: [1] … [2] …».

Интерфейс — **локальный, однопользовательский, без авторизации и без внешних API**.
Все сетевые вызовы — только `127.0.0.1` (llama-server). Телеметрия Gradio отключена
через `GRADIO_ANALYTICS_ENABLED=False`.

---

## 1. Цель этапа, границы и Definition of Done

### 1.1. Цель

Сделать демонстрационное веб-приложение, которое:

1. принимает вопрос на естественном языке;
2. выполняет гибридный поиск по регламенту;
3. генерирует ответ локальной LLM со ссылками `[n]`;
4. показывает ответ и **те же** источники, которые были переданы в контекст LLM;
5. корректно ведёт себя при отказе модели, недоступной LLM и обрезанном контексте;
6. запускается одной командой на Windows и пригодно для показа на защите.

### 1.2. Приоритеты

**P0 (обязательно):**

* `App/app.py` — чат + источники на Gradio;
* `Generation/rag_service.py` — сервисный слой без привязки к UI-фреймворку;
* `Retrieval/runtime.py` — кэш индексов и модели эмбеддингов;
* исправление соответствия «контекст ↔ источники» (T1);
* быстрый `probe()` для статуса LLM;
* тесты без сети и UI smoke-тест;
* обновлённые `README.md`, `requirements.txt`;
* приложение запускается и отвечает на вопросы при работающем llama-server.

**P1 (если P0 закрыт и осталось время):**

* вкладка «Метрики» (`App/metrics_view.py`) — read-only таблицы из `Eval/metrics.json`;
* переключатель метода поиска `hybrid / BM25 / vector` в панели настроек;
* кнопка «Скачать историю» (JSON) и «Скопировать последний ответ»;
* `run_app.ps1` для запуска в один клик;
* выбор источника из истории ответов (dropdown «Ответ №»);
* скриншоты для пояснительной записки.

**Вне scope этапа (не делать):**

* изменение корпуса, чанкинга, индексов, `alpha`, формул retrieval;
* пересборка `Chunks/*`;
* автоматическая ручная оценка ответов и заполнение `manual_scores.json`;
* стриминг ответа LLM (SSE) — только если P0 полностью закрыт и есть запас времени;
* RAG по истории диалога (каждый вопрос независим);
* авторизация, БД, Docker, деплой в интернет;
* пояснительная записка и презентация (недели 9–10).

### 1.3. Definition of Done

Этап считается завершённым, только если одновременно выполнено:

- [ ] `python App/app.py` поднимает приложение на `http://127.0.0.1:7860`;
- [ ] при работающем llama-server вопрос F2 даёт содержательный ответ с `[n]`,
      и все `n` существуют в списке источников;
- [ ] каждая карточка источника показывает номер, `chunk_id`, документ, раздел,
      тип, score и текст;
- [ ] trap-вопрос `query_id=T1` даёт ровно `Информация не найдена в регламенте.`;
- [ ] при недоступной LLM приложение не падает, показывает понятную ошибку и команду запуска;
- [ ] смена `top_k` (3/5/10) меняет число источников у **новых** ответов;
- [ ] при уменьшении `top_k` лишние карточки скрыты и не показывают данные прошлого ответа;
- [ ] контекст, обрезанный по 9000 символов, не порождает «лишних» источников;
- [ ] повторный вопрос не перезагружает модель эмбеддингов;
- [ ] unit-тесты и CLI-регрессии проходят;
- [ ] `README.md` и `requirements.txt` актуальны, неделя 8 отмечена;
- [ ] рабочее дерево без `__pycache__`, `Models/`, временных файлов;
- [ ] итоговый отчёт агенту-человеку содержит версии, команды, результаты и известные ограничения.

---

## 2. Зафиксированные технические решения

| Параметр | Значение |
|---|---|
| UI | Gradio `gr.Blocks`, локально |
| Адрес | `http://127.0.0.1:7860` (порт можно изменить `--port`) |
| LLM | llama-server, OpenAI-совместимый API, `http://127.0.0.1:8080/v1` |
| Модель | `t-lite-it-2.1`, GGUF Q4_K_M, thinking выключен |
| Поиск | гибрид BM25 + vector, `alpha = 0.7`, top_k = 5 по умолчанию |
| Контекст | не более 9000 символов, обрезка по границе блока |
| Ответ | до 150 слов, ссылки `[n]`, отказ одной фразой |
| История чата | в `gr.State`, только отображение; в LLM не передаётся |
| Источники | только реально вошедшие в контекст блоки, нумерация = `[n]` |
| Кэш ресурсов | `get_runtime()` + `functools.lru_cache`, один экземпляр на процесс |
| Очередь | `demo.queue(default_concurrency_limit=1)`, запросы к LLM последовательны |
| Внешняя сеть | запрещена; только localhost |
| Формат файлов | UTF-8 без BOM; в PowerShell `$env:PYTHONIOENCODING='utf-8'` |

---

## 3. Целевая структура файлов

```text
onpz/
├── PLAN_AI_AGENT.md                 # этот файл
├── README.md                        # ИЗМЕНИТЬ: структура, неделя 8, запуск UI
├── requirements.txt                 # ИЗМЕНИТЬ: +gradio (точный пин)
├── run_app.ps1                      # P1: запуск в один клик (опционально)
├── App/                             # НОВОЕ
│   ├── app.py                       # точка входа Gradio (gr.Blocks)
│   ├── ui_components.py             # рендер источников, статус LLM, метаданные
│   └── metrics_view.py              # P1: read-only вкладка метрик
├── Generation/
│   ├── rag_service.py               # НОВОЕ: единый пайплайн retrieve/ask
│   ├── prompt.py                    # ИЗМЕНИТЬ: ContextBlock + источники по контексту
│   ├── generate.py                  # ИЗМЕНИТЬ минимально: использовать ContextBlock
│   └── llm_client.py                # ИЗМЕНИТЬ: probe(), актуальный текст ошибки
├── Retrieval/
│   ├── runtime.py                   # НОВОЕ: кэш BM25/FAISS/модели
│   └── hybrid_retrieval.py          # ИЗМЕНИТЬ: +group_siblings_by_source()
├── Tests/                           # НОВОЕ
│   ├── __init__.py
│   ├── _bootstrap.py                # sys.path для Generation/Retrieval
│   ├── test_prompt_context.py
│   ├── test_rag_service.py
│   └── test_runtime_parity.py
├── Eval/                            # НЕ ТРОГАТЬ артефакты
├── Chunks/                          # НЕ ПЕРЕСОБИРАТЬ
├── Processed data/                  # НЕ ТРОГАТЬ
└── Models/                          # в .gitignore, не коммитить
```

Новые модули используют тот же приём с путями, что `Generation/generate.py`, но
с явными каталогами (важно для `App/app.py`, который лежит не в `Generation/`):

```python
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "Generation"))
sys.path.insert(0, str(BASE_DIR / "Retrieval"))
```

Для `Tests/_bootstrap.py` — тот же код; для модулей внутри `Generation/` допустим
вариант `sys.path.insert(0, str(Path(__file__).parent))`.

---

## 4. Инварианты (нарушение = провал задачи)

1. **Не менять** корпус, чанки, индексы, `Eval/test_queries.json`, `Eval/answers*.jsonl`,
   `Eval/manual_scores.json`, `Eval/metrics.*`. UI работает read-only по отношению к ним.
2. **Не пересобирать** `Chunks/bm25_index.pkl`, `Chunks/faiss_index.bin`,
   `Chunks/faiss_chunk_ids.json`. Если индексов нет или они битые — остановиться и сообщить.
3. **Не менять семантику** существующих публичных функций:
   `hybrid_search`, `combine_scores`, `expand_results`, `load_chunks`,
   `bm25_search.search`, `vector_search.search`, `generate_answer`.
   Разрешены только аддитивные изменения (новые функции/поля с обратной совместимостью).
4. **Не выдумывать метрики.** `Answer Accuracy` и `Hallucination Rate` заполняет человек.
   Агент не ставит оценки и не подставляет «правдоподобные» числа.
5. **Не ходить в интернет** из кода приложения. Единственный сетевой адрес —
   `127.0.0.1` (llama-server). Телеметрию Gradio отключить (`GRADIO_ANALYTICS_ENABLED=False`).
6. **Не скачивать** модели и не устанавливать llama.cpp. Запуск llama-server — человек.
   Если сервер недоступен, UI обязан показать инструкцию, а не падать.
7. Все текстовые файлы — UTF-8 без BOM. Перед запуском:
   `$env:PYTHONIOENCODING='utf-8'`.
8. Скрипты запускаются **из корня репозитория**. Внутри модулей — `pathlib`, `argparse`,
   type hints, docstring на русском, без side-effect при импорте.
9. **Не отключать санитизацию HTML** (`sanitize_html=False`) и не использовать
   `gr.HTML` для текста ответа, чанков и имён файлов.
   Текст чанков — недоверенные данные, их нельзя вставлять как HTML.
10. **Никаких секретов** в репозитории. Не создавать `.env`, `tokens.txt` и подобные файлы.
11. Ошибки пользователю — на русском и с действием («запустите…», «проверьте…»).
    Трассировки — только в stderr/терминал, не в UI.
12. Коммиты атомарные, сообщения в стиле истории (`feat:`, `fix:`, `docs:`, `test:`).
    Не коммитить `__pycache__/`, `Models/`, `*.gguf`, временные логи.

---

## 5. Задачи по порядку

Зависимости: `T0 → T1 → T2 → T3 → T4 → T5 → T6 → T7 → T8`.
T1 и T2 можно делать параллельно после T0; T3 требует T1 и T2; T5 требует T3 и T4.

---

### T0. Preflight и фиксация baseline (0,5–1 ч)

**Цель:** убедиться, что исходное состояние воспроизводимо, и записать baseline
до изменений.

**Действия:**

1. `git status --short`, `git branch --show-current`, `git log --oneline -5`.
   Ожидаемые незакоммиченные изменения на старте: новый `PLAN_AI_AGENT.md` и правка
   ссылки на него в `README.md`. Их не откатывать; остальные изменения — повод спросить.
2. `python --version`, `python -m pip check`, `python -m pip show gradio` (ожидается «не найден»).
3. Проверить наличие файлов:
   `Chunks/bm25_index.pkl`, `Chunks/faiss_index.bin`, `Chunks/faiss_chunk_ids.json`,
   `Eval/test_queries.json`, `Eval/answers.jsonl`, `Eval/metrics.json`.
4. Проверить импорты без сети:
   ```powershell
   $env:PYTHONIOENCODING='utf-8'
   python -c "import faiss, numpy, sentence_transformers, pymorphy3, rank_bm25; print('imports ok')"
   ```
5. Проверить llama-server:
   ```powershell
   Invoke-RestMethod http://127.0.0.1:8080/v1/models
   ```
   * если доступен — выполнить контрольный CLI-запрос и записать ответ/латентность;
   * если недоступен — **не запускать сервер самостоятельно**, зафиксировать
     «LLM offline» и продолжить офлайн-часть. Финальная приёмка T8 будет заблокирована
     до запуска сервера человеком.
6. Зафиксировать baseline `Eval/metrics.json` (скопировать значения в отчёт, не менять файл).

**Acceptance:** в итоговом отчёте есть версии Python/пакетов, статус LLM, хэш/дата
`git log`, baseline-метрики. Код на этом шаге не меняется.

---

### T1. Контекст и источники: единая структура (P0, 1–2 ч)

**Проблема.** Сейчас `build_context()` может выбросить часть top_k из-за лимита
9000 символов, а `format_sources()` всё равно перечисляет все результаты.
UI покажет источник, которого модель не видела; `sanitize_citations` может удалить
ссылку `[4]`, но список источников всё равно останется неверным.

**Файлы:** `Generation/prompt.py`, `Generation/generate.py`, `Tests/test_prompt_context.py`.

**Контракт `Generation/prompt.py` (аддитивно):**

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ContextBlock:
    number: int
    chunk_id: str
    chunk: dict
    extra_chunks: tuple[dict, ...] = ()
    text: str = ""          # готовый блок: заголовок + тело, как в старом build_context

def build_context_blocks(
    results: list[tuple[str, float]],
    chunk_lookup: dict,
    expanded_chunks: dict[str, list[dict]] | None = None,
) -> list[ContextBlock]: ...

def render_context(blocks: list[ContextBlock]) -> str:
    """Склеить блоки через '\n\n'.""" ...

def build_context(
    results: list[tuple[str, float]],
    chunk_lookup: dict,
    expanded_chunks: dict[str, list[dict]] | None = None,
) -> str:
    """Обратная совместимость: render_context(build_context_blocks(...)).""" ...

def format_sources_from_blocks(blocks: list[ContextBlock]) -> str:
    """Источники по реально включённым блокам; номера берутся из block.number.""" ...
```

**Требования:**

* Алгоритм обрезки и нумерации — **точно как сейчас**: пропускать `chunk_id`,
  которого нет в `chunk_lookup`; нумеровать только включённые блоки; не разрывать блок;
  лимит `MAX_CONTEXT_CHARS = 9000`; разделитель между блоками — `\n\n`.
* `ContextBlock.text` включает заголовок `[n] chunk_id=… | document=… | section=…`
  и тело, включая `extra_chunks` (соседей процедуры), как `_format_block`.
* `format_sources_from_blocks` выдаёт тот же формат, что старый `format_sources`,
  но строки — только для включённых блоков и с их номерами.
* `format_sources()` оставить без изменения (обратная совместимость); новый код использует
  `format_sources_from_blocks`.
* В `generate.generate_answer` заменить:
  * `context = build_context(...)` → `blocks = build_context_blocks(...)`;
  * `context = render_context(blocks)`;
  * `sources = format_sources_from_blocks(blocks)`.
* `build_record` продолжает писать **все** top_k в `retrieval.chunks` — это журнал поиска,
  а не контекст. Не менять.
* Не перегенерировать `Eval/answers*.jsonl`; старые артефакты остаются как есть.

**Тесты (`Tests/test_prompt_context.py`, без сети):**

1. 6 фейковых чанков → 6 блоков, номера 1..6, `render_context` содержит все `chunk_id`,
   `format_sources_from_blocks` — 6 строк.
2. 3 огромных чанка (например, по 5000 символов) → в блоки попадает меньше 3;
   `format_sources_from_blocks` не содержит выброшенный `chunk_id`;
   номера оставшихся идут подряд.
3. `results` содержит несуществующий `chunk_id` → нумерация не рвётся,
   источники соответствуют блокам.
4. `build_context(...) == render_context(build_context_blocks(...))`.
5. `extra_chunks` попадают в текст блока, но не увеличивают число источников.

**Acceptance:**

```powershell
$env:PYTHONIOENCODING='utf-8'
python -m unittest Tests.test_prompt_context -v
python Generation/generate.py --query "Какое давление поддерживается в отбензинивающей колонне К-1?" --show_context
```

В выводе CLI номера `[n]` в ответе существуют в блоке «Источники»; при `--show_context`
число блоков совпадает с числом источников.

---

### T2. Кэш ресурсов поиска: `Retrieval/runtime.py` (P0, 1–2 ч)

**Проблема.** `hybrid_search()` заново читает индексы и загружает
`SentenceTransformer` при каждом вызове. В UI это недопустимо.

**Файлы:** `Retrieval/runtime.py`, `Tests/test_runtime_parity.py`.

**Контракт:**

```python
class RetrievalRuntime:
    chunks: list[dict]
    chunk_lookup: dict[str, dict]
    list_groups: dict
    bm25_index: ...
    bm25_chunk_ids: list[str]
    vector_model: ...
    vector_index: ...
    vector_chunk_ids: list[str]

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        alpha: float = ALPHA_DEFAULT,
        method: str = "hybrid",
        pool_size: int = CANDIDATE_POOL_SIZE,
    ) -> list[tuple[str, float]]: ...

def load_runtime() -> RetrievalRuntime: ...

@lru_cache(maxsize=1)
def get_runtime() -> RetrievalRuntime: ...
```

**Требования:**

* `load_runtime()` читает `chunks` **один раз**, строит `chunk_lookup` и `list_groups`
  через функции `hybrid_retrieval`, загружает BM25/FAISS/модель один раз.
* `method="hybrid"` вызывает `combine_scores` с теми же аргументами, что
  `hybrid_search`, и даёт **тот же порядок** `chunk_id` и те же scores (с точностью `1e-6`).
* `method="bm25"` / `"vector"` возвращают top_k соответствующего метода
  (P1 для UI, но runtime поддерживает сразу).
* Валидация: `top_k >= 1`, `0.0 <= alpha <= 1.0`, `method` из допустимого набора;
  иначе `ValueError`.
* `get_runtime()` кэширует экземпляр на процесс; `get_runtime.cache_clear()` доступен тестам.
* Не менять `hybrid_search`; новый runtime — единственный путь для UI.
* Потокобезопасность: FAISS `IndexFlatIP.search` и `SentenceTransformer.encode`
  используются только на чтение; отдельные блокировки не нужны.

**Acceptance:**

```powershell
$env:PYTHONIOENCODING='utf-8'
python -c "import sys; sys.path.insert(0,'Retrieval'); from runtime import get_runtime; r=get_runtime(); print('cached:', r is get_runtime()); print(r.search('Какое давление в колонне К-1?', top_k=5))"
python -m unittest Tests.test_runtime_parity -v
```

Тест сравнивает `runtime.search(q)` с `hybrid_search(q)` на 3 запросах:
одинаковые `chunk_id`, scores с точностью `1e-6`. Второй вызов `get_runtime()` — тот же объект.
Замер: первый вызов включает загрузку модели, второй — только поиск; в логах
`SentenceTransformer` загружается один раз.

---

### T3. Сервисный слой: `Generation/rag_service.py` (P0, 1,5–2,5 ч)

**Цель:** один пайплайн `retrieve/ask`, который используют UI и тесты; без Gradio,
без записи артефактов, без глобального состояния кроме кэша runtime.

**Файлы:** `Generation/rag_service.py`, `Retrieval/hybrid_retrieval.py` (аддитивно),
`Tests/test_rag_service.py`.

**Аддитивная функция в `hybrid_retrieval.py`:**

```python
def group_siblings_by_source(
    results: list[tuple[str, float]],
    chunk_lookup: dict,
    list_groups: dict,
) -> dict[str, list[dict]]:
    """Для каждого найденного источника вернуть соседей его list_item-группы,
    исключая уже найденные источники. Поведение = build_expanded_by_source."""
```

`generate.build_expanded_by_source` можно сделать тонкой обёрткой над ней;
вывод CLI не должен измениться (проверить на одном запросе).

**Контракт `rag_service.py`:**

```python
@dataclass
class SourceCard:
    number: int
    chunk_id: str
    score: float
    document: str
    section: str | None
    subsection: str | None
    type: str
    text: str                      # humanize_math(chunk["text"])
    expanded_chunk_ids: list[str] = field(default_factory=list)
    expanded_text: str = ""        # humanize_math текстов соседей через "\n\n"

@dataclass
class RetrievalView:
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

class RAGError(RuntimeError): ...

def retrieve(
    question: str,
    *,
    top_k: int = 5,
    alpha: float = ALPHA_DEFAULT,
    expand_abbrev: bool = False,
    method: str = "hybrid",
    runtime=None,
) -> RetrievalView: ...

def ask(
    question: str,
    *,
    top_k: int = 5,
    alpha: float = ALPHA_DEFAULT,
    expand_abbrev: bool = False,
    think: bool = False,
    method: str = "hybrid",
    runtime=None,
) -> RAGAnswer: ...
```

**Поток `retrieve`:**

1. `question = question.strip()`; пустой → `RAGError("Введите вопрос.")`.
2. `runtime = runtime or get_runtime()`.
3. `search_query = expand_query(question)` при `expand_abbrev`, иначе `question`.
4. `results = runtime.search(search_query, top_k=top_k, alpha=alpha, method=method)`.
5. `expanded_by_source = group_siblings_by_source(results, runtime.chunk_lookup, runtime.list_groups)`.
6. `blocks = build_context_blocks(results, runtime.chunk_lookup, expanded_by_source)`.
7. `sources` — по `blocks`: номер из `block.number`, score из `dict(results)`,
   метаданные из чанка, `text = humanize_math(chunk["text"])`,
   `expanded_chunk_ids` и `expanded_text` из `block.extra_chunks`.
8. `context_truncated = len(blocks) < len(results)`;
   `omitted_sources = len(results) - len(blocks)`.
9. Вернуть `RetrievalView` с замером времени только retrieval.

**Поток `ask`:**

1. `view = retrieve(...)`.
2. `messages = build_messages(question, view.context)`.
3. `response = llm_client.chat(messages, temperature=0.1, think=think)`.
4. `answer = sanitize_citations(response["text"].strip(), view.context)`.
5. `refused = is_refusal(answer)`.
6. `settings = {"top_k": top_k, "alpha": alpha, "expand_abbrev": expand_abbrev,
   "think": think, "method": method, "model": llm_client.MODEL}`.
7. `usage` нормализовать до `{"prompt_tokens": int, "completion_tokens": int}`.
8. `latency_s` — общее время; `generation_latency_s = response["latency_s"]`.
9. Никаких записей в `Eval/`, `Chunks/`, `Processed data/`.

**Ошибки:**

* пустой вопрос, `top_k`/`alpha` вне диапазона — `RAGError`/`ValueError`;
* LLM недоступна/таймаут — пробросить `RuntimeError` из `llm_client` (UI покажет
  понятное сообщение), не глотать и не возвращать фиктивный ответ;
* пустой ответ модели — вернуть `answer=""`, `refused=False`; UI покажет предупреждение.

**Тесты (`Tests/test_rag_service.py`, без сети):**

* подменить `rag_service.get_runtime` на stub с 2 чанками и `list_groups`;
* подменить `rag_service.llm_client.chat` на fake, возвращающий `"... [1]"`;
* `ask()` → sources `[1]`, `answer` содержит `[1]`, `context` содержит chunk 1;
* fake-ответ `[1] [9]` → после `sanitize_citations` `[9]` удалён;
* fake-ответ `REFUSAL_PHRASE` → `refused is True`;
* `retrieve()` не вызывает `llm_client.chat` (patch, который бросает исключение);
* пустой вопрос → `RAGError`;
* `context_truncated` и `omitted_sources` корректны при огромных чанках.

**Acceptance:**

```powershell
$env:PYTHONIOENCODING='utf-8'
python -m unittest Tests.test_rag_service -v
```

Тесты не создают и не меняют файлы в `Eval/`.

---

### T4. Быстрый статус LLM: `llm_client.probe()` (P0, 0,5 ч)

**Файл:** `Generation/llm_client.py`.

**Контракт:**

```python
def probe(timeout_s: float = 2.0) -> dict:
    """Быстрая проверка доступности без исключения.

    Возвращает {"ok": bool, "model": str | None, "error": str | None}.
    """
```

**Требования:**

* `GET {BASE_URL}/models` с `timeout=timeout_s`, без ретраев.
* Успех: `ok=True`, `model = data["data"][0]["id"]` (fallback `MODEL`), `error=None`.
* Ошибка соединения/таймаут/не-2xx/невалидный JSON: `ok=False`, `model=None`,
  `error` — короткая причина без трассировки.
* `healthcheck()` не менять по поведению; обновить только текст ошибки:
  `"LLM недоступна. Запустите llama-server (см. README.md, раздел «Генерация ответа»)."`
* UI использует только `probe()` для статуса; `chat()` — для генерации.

**Acceptance:**

```powershell
$env:PYTHONIOENCODING='utf-8'
python -c "import sys; sys.path.insert(0,'Generation'); import llm_client; print(llm_client.probe())"
$env:LLM_BASE_URL='http://127.0.0.1:9/v1'
python -c "import sys; sys.path.insert(0,'Generation'); import llm_client; print(llm_client.probe(timeout_s=1))"
```

Второй вызов возвращает `ok=False` менее чем за ~2 с и не бросает исключение.
Не забыть восстановить `LLM_BASE_URL`.

---

### T5. Gradio-приложение (P0, 3–5 ч)

**Файлы:** `App/app.py`, `App/ui_components.py`.

#### T5.1. Установка

```powershell
$env:PYTHONIOENCODING='utf-8'
python -m pip install "gradio>=5.0,<6"
python -m pip show gradio
python -m pip check
python -c "import faiss, numpy, gradio; print(faiss.__version__, numpy.__version__, gradio.__version__)"
```

* Записать **точную** установленную версию Gradio в `requirements.txt`.
* Если pip пытается понизить/повысить `numpy`, `faiss-cpu`, `sentence-transformers`,
  `pydantic` или `fastapi` и после этого ломаются импорты — остановиться и сообщить,
  не ломать рабочее окружение.
* Если установка невозможна — остановиться; **не переключаться молча на Streamlit**.

#### T5.2. Конфигурация запуска

Отдельный файл конфигурации не нужен. Параметры задаются в `app.py`:

```python
import os
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import argparse
import gradio as gr

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--no-browser", action="store_true")
args = parser.parse_args()

if args.host not in {"127.0.0.1", "localhost"}:
    raise SystemExit("Допустим только локальный хост (127.0.0.1).")

# ... построение demo = gr.Blocks(...) ...

demo.queue(default_concurrency_limit=1)
demo.launch(
    server_name=args.host,
    server_port=args.port,
    share=False,
    inbrowser=not args.no_browser,
    show_api=False,
)
```

* `share=False` — публичная ссылка Gradio запрещена.
* `default_concurrency_limit=1` — llama-server обрабатывает один запрос за раз.
* `--host` принимать только `127.0.0.1`/`localhost`; иное значение — `SystemExit`.
* Для автотестов и фонового запуска — `--no-browser`.
* Порт по умолчанию 7860; при занятом порте — `--port 7861`.

#### T5.3. Структура UI

```text
┌ Левая колонка (scale=1) ─────┬ Чат (scale=4) ──────────────────────────┐
│ Параметры поиска             │ RAG-поиск по регламенту ЭЛОУ-АВТ       │
│ top_k        [5]             │ 10 документов · 206 чанков · гибрид    │
│ alpha        [0.7]           │                                        │
│ метод        [hybrid]        │ ┌ gr.Chatbot (messages) ─────────────┐ │
│ [ ] раскрывать аббревиатуры  │ │ user: вопрос                       │ │
│ [ ] показывать контекст LLM  │ │ assistant: ответ [1], [2]          │ │
│                              │ │ Источники: [1] … [2] …             │ │
│ Статус LLM                   │ └────────────────────────────────────┘ │
│ ● доступна: t-lite-it-2.1    │ [ поле ввода вопроса ]  [Отправить]    │
│ [Проверить соединение]       │                                        │
│                              │ Источники последнего ответа (5)        │
│ Примеры вопросов             │ ▸ 1. документ — раздел (score)         │
│ Категория [фактологические]  │ ▸ 2. ...                               │
│ Вопрос    [...............]  │                                        │
│ [Задать вопрос]              │ Контекст LLM (скрыт по умолчанию)      │
│ [Очистить чат]               │ ▸ блоки [1]…[N]                        │
│ [Скачать историю]            │                                        │
│ Дисклеймер прототипа         │ Вкладка «Метрики» — P1                 │
└──────────────────────────────┴────────────────────────────────────────┘
```

Технически: `gr.Blocks` → `gr.Row` → `gr.Column(scale=1)` (настройки) и
`gr.Column(scale=4)` (чат). Внутри правой колонки — `gr.Tabs` с вкладками
«Чат» и «Метрики» (P1).

#### T5.4. Обязательные элементы

**Левая колонка (настройки):**

* `gr.Slider(1, 10, value=5, step=1, label="top_k")` с `info`
  «сколько фрагментов подать в LLM».
* `gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="alpha")` с `info`
  «вес BM25 в гибридном поиске».
* `gr.Radio(["hybrid", "bm25", "vector"], value="hybrid", label="Метод поиска")`
  — P1; в P0 допускается оставить только `hybrid` и скрыть компонент.
* `gr.Checkbox(label="Раскрывать аббревиатуры", value=False)`.
* `gr.Checkbox(label="Показывать контекст, отправленный в LLM", value=False)`;
  `checkbox.change(fn=lambda show: gr.update(visible=show), inputs=checkbox,
  outputs=context_acc)` сразу скрывает/показывает аккордеон, содержимое обновляется
  при следующем ответе.
* статус LLM — `gr.Markdown`:
  * при загрузке страницы `demo.load(fn=check_llm, outputs=[llm_status, llm_state])`
    вызывает `llm_client.probe(timeout_s=1.5)` **один раз**;
  * зелёный статус с именем модели либо красный с причиной и командой запуска;
  * кнопка «Проверить соединение» обновляет статус и `llm_state`;
  * `probe()` не вызывать на каждый вопрос;
  * команда для человека:
    ```powershell
    llama-server -m "Models/T-lite-it-2.1-Q4_K_M.gguf" -c 8192 -ngl 99 -fa on --jinja --host 127.0.0.1 --port 8080
    ```
* примеры вопросов из `Eval/test_queries.json`:
  * показывать только `query_text`; **не показывать** `reference_answer`,
    `gold_chunk_ids` и `expected_behavior` (это эталон для оценки, а не подсказка);
  * `gr.Dropdown` категории (русские подписи);
  * `gr.Dropdown` вопроса; при смене категории обновлять `choices` вопросов;
  * кнопка «Задать вопрос» вызывает тот же обработчик, что и поле ввода,
    с текстом выбранного примера;
  * для категории `trap` — `gr.Markdown("В регламенте нет ответа, ожидается отказ")`.
* кнопка «Очистить чат»: очищает `chatbot`, `history_state`, карточки источников,
  метаданные и контекст.
* кнопка «Скачать историю» — P1 (`gr.DownloadButton`, JSON из `history_state`).
* дисклеймер: «Прототип для учебной задачи. Не является руководством по эксплуатации.
  Ответы ограничены учебным корпусом».

**Правая колонка (чат):**

* `gr.Blocks(title="RAG ЭЛОУ-АВТ")`; заголовок и краткое описание: 10 документов,
  206 чанков, гибридный поиск, ответы со ссылками `[n]`.
* `gr.Chatbot(type="messages", height=520, show_copy_button=True,
  sanitize_html=True)` — история отображается как сообщения `user`/`assistant`.
* Если история пуста — `gr.Markdown` с 2–3 примерами и объяснением, что вопрос
  задаётся в поле ниже.
* `gr.Textbox(placeholder="Например: какое давление поддерживается в колонне К-1?")`
  + `gr.Button("Отправить", variant="primary")`; `textbox.submit` и `button.click`
  вызывают один обработчик.
* При вопросе:
  1. проверить непустой вопрос; пустой → `gr.Warning("Введите вопрос.")`;
  2. добавить user-сообщение в `history_state` до вызова сервиса;
  3. вызвать `rag_service.ask(...)` с текущими настройками (Gradio сам показывает
     индикатор выполнения; дополнительно можно использовать `gr.Progress`);
  4. собрать assistant-сообщение: ответ + компактная строка
     «Источники: [1] документ — раздел; [2] …»;
  5. записать assistant-сообщение и `dataclasses.asdict(sources)` в `history_state`;
  6. обновить `gr.Chatbot`, карточки источников, метаданные и (если включено) контекст.
* Ошибка LLM: **перехватить** исключение, добавить в чат assistant-сообщение
  «Не удалось получить ответ…», вернуть обновлённые компоненты и показать
  `gr.Warning(...)`. Не использовать `raise gr.Error` в этом месте: оно прерывает
  обработчик и не даёт обновить чат. Traceback — только в терминал.
* `gr.ClearButton` не использовать для чата: он не очищает `gr.State` и карточки;
  очистка — собственный обработчик.
* Не передавать историю чата в LLM. Каждый вопрос независим; это явно написать в UI.

#### T5.5. Карточка источника

Динамически создавать компоненты в Gradio нельзя, поэтому в правой колонке заранее
создаются **10 пар** «аккордеон + Markdown»:

```python
source_accords = []
source_bodies = []
for i in range(1, 11):
    with gr.Accordion(f"{i}. Источник", open=False, visible=False) as acc:
        body = gr.Markdown()
    source_accords.append(acc)
    source_bodies.append(body)
```

Для каждого `SourceCard` при обновлении:

* заголовок аккордеона: `"{number}. {document} — {section или subsection}"`,
  длинные строки обрезать до ~90 символов; `visible=True`;
* тело: caption `` `chunk_id` · type · score=0.861 ``; score — балл гибридного
  ранжирования, а не вероятность правильности ответа;
* текст (сервис уже применил `humanize_math`, повторно не преобразовывать):
  * `table_row` → fenced code block с длиной ограды больше любой серии backticks
    в тексте (сохраняет строку таблицы);
  * остальные → обычный Markdown;
* если `expanded_text` непустой — подпись «Дополнено соседними пунктами процедуры»
  и текст соседей в том же Markdown;
* лишние аккордеоны — `visible=False`;
* перед списком caption: «Номера [n] в ответе соответствуют источникам ниже».

`gr.Markdown` использовать с `sanitize_html=True` (по умолчанию). **Запрещено**
передавать текст чанков, ответ и имена файлов в `gr.HTML`; отключать `sanitize_html`
у `gr.Markdown`/`gr.Chatbot`; собирать HTML-строки из `chunk_id`/`section`
без экранирования.

#### T5.5.1. Контракт обработчика

Один обработчик `handle_question` обслуживает поле ввода, кнопку «Отправить»
и примеры вопросов. Порядок `outputs` фиксирован:

```python
outputs = [
    chatbot,          # gr.Chatbot
    history_state,    # gr.State
    meta_md,          # gr.Markdown
    context_acc,      # gr.Accordion
    context_code,     # gr.Code
    *source_accords,  # 10 шт.
    *source_bodies,   # 10 шт.
]
```

Возврат — кортеж той же длины. Для скрытых карточек возвращать
`gr.update(visible=False)` и пустой Markdown. Если вопрос пустой — `gr.Warning`
и возврат текущих значений без вызова LLM. При ошибке LLM — вернуть обновлённый
чат с сообщением об ошибке (см. T5.4), а не прерывать обработчик.

#### T5.6. Метаданные ответа

Под чатом, отдельным `gr.Markdown`:

```text
⏱ 3.2 с (поиск 0.8 / генерация 2.4) · tokens 2361 in / 71 out
top_k=5 · alpha=0.7 · метод=hybrid · модель=t-lite-it-2.1
```

* Если `context_truncated`: строка «В контекст вошли N из top_k источников;
  M не поместились из-за лимита 9000 символов».
* Если `refused`: ответ отображается как цитата (`> …`) + caption «Модель не нашла
  ответа в переданном контексте». Источники остаются видимыми с пометкой
  «Найденные фрагменты (не использованы)».
* Если `answer == ""` и не отказ: предупреждение в метаданных
  «Модель вернула пустой ответ…».
* Если ошибка LLM: сообщение об ошибке в чате + `gr.Warning(...)`,
  `llm_status` перерисовывается как красный; traceback — только в терминал.
  `raise gr.Error` оставить только для неустранимых ошибок загрузки индексов.
* Контекст LLM — `gr.Accordion("Контекст, отправленный в LLM", open=False)` +
  `gr.Code(language=None, interactive=False)`; аккордеон `visible` только при
  включённом чекбоксе. Длинный контекст прокручивается внутри компонента.

#### T5.7. Хранение состояния

```python
history_state = gr.State([])   # полная история, включая источники
llm_state = gr.State({})       # последний результат probe()

# элемент history_state:
[
    {"role": "user", "content": str},
    {
        "role": "assistant",
        "content": str,               # ответ + компактная строка источников
        "answer": str,                # чистый ответ без служебной строки
        "sources": [dict, ...],       # dataclasses.asdict(SourceCard)
        "context": str,
        "meta": {
            "latency_s": float,
            "retrieval_latency_s": float,
            "generation_latency_s": float,
            "usage": {"prompt_tokens": int, "completion_tokens": int},
            "settings": dict,
            "context_truncated": bool,
            "omitted_sources": int,
            "refused": bool,
        },
        "error": str | None,
    },
]
```

* В `gr.Chatbot` передавать **только** `{"role", "content"}`: helper
  `to_chatbot_messages(history)` отбрасывает служебные поля. Не передавать
  dataclass-объекты в `gr.State`.
* Не мутировать входной список истории на месте: создавать новый список
  (`history = [*history, new_message]`), иначе возможны дубли при повторных событиях.
* Хранить источники как **dict**, не dataclass: `gr.State` должен быть сериализуем.
* Старые сообщения отображаются с теми настройками, с которыми были получены;
  смена левой панели влияет только на новые вопросы.
* История живёт в рамках browser-сессии; перезагрузка страницы её очищает — это допустимо.
* `runtime` в `gr.State` не хранить: кэш живёт в `Retrieval/runtime.py`.

#### T5.8. Кэш runtime в UI

Отдельный UI-кэш не нужен:

```python
from runtime import get_runtime

def handle_question(...):
    ...
    result = rag_service.ask(..., runtime=get_runtime())
```

* `get_runtime()` кэшируется через `lru_cache` и вызывается только при первом
  вопросе (не при загрузке страницы), чтобы приложение открывалось быстро.
* Если загрузка упала — `raise gr.Error(...)` с проверкой `Chunks/` и `Models/`; не пытаться
  пересобрать индексы автоматически.
* Не вызывать `hybrid_search()` напрямую из UI: только `rag_service`/runtime.

#### T5.9. P1: вкладка «Метрики»

`App/metrics_view.py`, функция `build_metrics_tab()`, вызывается внутри
`gr.Tab("Метрики")` в `app.py`:

* пути считать от `Path(__file__).resolve().parents[1]` (корень репозитория);
* читает `Eval/metrics.json` (read-only), при отсутствии — `raise gr.Error(...)` с путём;
* `gr.Dataframe` или Markdown-таблица retrieval: BM25 / Vector / Hybrid —
  Recall@5, MRR@5, n_queries;
* генерация: refusal rate, trap refusal correctness, answer accuracy,
  hallucination rate; `null` показывать как `—`;
* если `answer_accuracy_mean is None` или `hallucination_rate is None` —
  `gr.Markdown("⚠️ Answer Accuracy и Hallucination Rate ожидают ручной оценки…")`;
* таблицы top_k 3/5/10 и аббревиатур;
* `gr.Accordion` с полным `Eval/metrics.md`;
* **не запускать** `Eval/evaluate.py` и не писать в `Eval/` из UI.

#### Acceptance T5

```powershell
$env:PYTHONIOENCODING='utf-8'
python App/app.py --no-browser
```

В другом терминале:

```powershell
$r = Invoke-WebRequest http://127.0.0.1:7860/ -UseBasicParsing
$r.StatusCode
```

Ожидается `200`. Далее — браузерный smoke по матрице §7.
Для остановки — `Ctrl+C` в терминале приложения.

---

### T6. Тесты и smoke-проверка (P0, 1,5–2,5 ч)

**Файлы:** `Tests/__init__.py`, `Tests/test_prompt_context.py`,
`Tests/test_runtime_parity.py`, `Tests/test_rag_service.py`.

**Правила:**

* только `unittest` из стандартной библиотеки, без `pytest` (не добавлять зависимость);
* тесты не ходят в сеть, не пишут в `Eval/`, `Chunks/`, `Processed data/`;
* `Tests/_bootstrap.py` (или путь в каждом файле) добавляет `Generation` и `Retrieval`
  в `sys.path`;
* `test_runtime_parity.py` — интеграционный, может быть медленным (загрузка модели);
  если индексов нет — `unittest.skipUnless`, но в текущем репозитории они есть.

**Запуск:**

```powershell
$env:PYTHONIOENCODING='utf-8'
python -m unittest discover -s Tests -v
```

**Обязательные кейсы** — см. T1/T2/T3. Дополнительно:

* `sanitize_citations` не ломает текст без ссылок;
* `expand_query` не меняет запрос без аббревиатур;
* `is_refusal` распознаёт фразу с точкой/без точки и в нижнем регистре.

**UI smoke (вручную или browser-инструментом):**

1. Открыть `http://127.0.0.1:7860`, дождаться заголовка.
2. Проверить, что статус LLM зелёный (при запущенном сервере).
3. Задать F2, дождаться ответа; проверить:
   * ответ содержит давление и `[1]`;
   * в источниках есть `03_reglament_at_rezhim_k1_k2__001`;
   * число видимых карточек = top_k (если нет обрезки);
   * ни одна карточка не пустая; текст не «кракозябры».
4. Раскрыть источник 1 — видны `chunk_id`, раздел, score, текст.
5. Задать trap-запрос `query_id=T1` — ровно фраза отказа, без выдуманных фактов.
6. Поставить top_k=3, задать F5 — 3 карточки.
7. Включить «Раскрывать аббревиатуры», задать S1 — ответ не падает, источники релевантны.
8. Включить «Показывать контекст» — аккордеон с контекстом, блоки `[1]…`.
9. «Очистить чат» — история пуста, карточки скрыты, контекст очищен.
10. Проверить консоль браузера и терминал Gradio: нет tracebacks/ошибок.
11. Проверить недоступную LLM (см. §8.4): красный статус, понятная ошибка, приложение живо.
12. Обновить страницу — история очищена (ожидаемо), приложение работоспособно.

---

### T7. Документация и зависимости (P0, 1 ч)

**Файлы:** `README.md`, `requirements.txt` (при P1 — `run_app.ps1`).

**`requirements.txt`:**

* добавить `gradio==<точная версия из pip show>`; остальные строки не менять;
* проверить `python -m pip check` и импорты `faiss`, `numpy`, `sentence_transformers`, `gradio`;
* при необходимости добавить в `.gitignore` служебные каталоги Gradio
  (`flagged/`, `.gradio/`, `gradio_cached_examples/`), если они появились.

**`README.md`:**

1. В структуре репозитория добавить `App/`, `Generation/rag_service.py`,
   `Retrieval/runtime.py`, `Tests/` (и `run_app.ps1`, если создан).
2. Исправить строку про `PLAN_AI_AGENT.md`: «План работ недели 8 (веб-интерфейс на Gradio)».
3. В таблице стека: «Интерфейс — Gradio; альтернатива — Streamlit».
4. В плане работ отметить неделю 8 как ✅ **только после полной приёмки T8**
   (если LLM offline — оставить ⬜ и написать «реализовано, требуется проверка с llama-server»).
5. Добавить раздел «Веб-интерфейс (неделя 8)»:
   * установка `python -m pip install -r requirements.txt`;
   * запуск llama-server (команда из раздела «Генерация ответа»);
   * запуск `python App/app.py` (или `python App/app.py --no-browser`);
   * адрес `http://127.0.0.1:7860`;
   * возможности: чат, источники, top_k/alpha, раскрытие аббревиатур, статус LLM;
   * ограничения: локальный однопользовательский прототип, история не передаётся в LLM,
     ответы ограничены корпусом, generation-метрики ждут ручной оценки;
   * troubleshooting: LLM недоступна, порт занят (`--port 7861`),
     нет gradio, нет индексов.
6. Добавить примечание: `Answer Accuracy` и `Hallucination Rate` остаются `null`
   до заполнения `Eval/manual_scores.json`; UI не подставляет оценки.
7. Проверить, что все команды из README копируются и работают из корня репозитория.

**Acceptance:** README не содержит ссылок на удалённые планы; все команды валидны;
`requirements.txt` воспроизводит окружение (с учётом уже установленных пакетов).

---

### T8. Финальная приёмка и демо-сценарий (P0, 1–2 ч)

#### 8.1. Команды приёмки

```powershell
$env:PYTHONIOENCODING='utf-8'

# 1. Синтаксис
python -m compileall App Generation Retrieval Eval Tests

# 2. Unit-тесты
python -m unittest discover -s Tests -v

# 3. Зависимости
python -m pip check
python -c "import faiss, numpy, gradio, sentence_transformers; print('imports ok')"

# 4. Регрессия retrieval и CLI
python Retrieval/hybrid_retrieval.py --query "Какое давление поддерживается в отбензинивающей колонне К-1?"
python Generation/generate.py --query "Какое давление поддерживается в отбензинивающей колонне К-1?" --top_k 5

# 5. Регрессия метрик (артефакты не должны измениться без причины)
python Eval/evaluate.py

# 6. UI
python App/app.py --no-browser
```

#### 8.2. Ожидаемые результаты

* тесты: `OK`, без сетевых ошибок;
* CLI-ответ содержит `[n]` и источники, число источников = числу блоков контекста;
* `Eval/metrics.json`: retrieval-метрики не изменились
  (BM25 1.000/0.823, vector 0.773/0.583, hybrid 1.000/0.871);
* generation-метрики остаются `null` — это ожидаемо;
* UI открывается, вопросы работают, источники отображаются.

#### 8.3. Демо-сценарий на защиту (5 минут)

1. Показать терминал: llama-server запущен; статус в левой панели зелёный.
2. Задать F2 «Какое давление поддерживается в отбензинивающей колонне К-1?»
   — ответ с `[1]`, источник `03_reglament_at_rezhim_k1_k2__001`, score.
3. Раскрыть источник, показать раздел/подраздел и дословный текст.
4. Задать trap-запрос `query_id=T1` «Какой катализатор используется в процессе
   гидроочистки дизельного топлива?» — корректный отказ.
5. Поставить top_k=3, задать N1 — показать 3 источника.
6. Включить «Раскрывать аббревиатуры», задать S1 — показать устойчивость.
7. P1: открыть «Метрики» — Recall@5/MRR, честно сказать, что Answer Accuracy
   и Hallucination Rate ждут ручной оценки.
8. Назвать ограничения: 25 запросов, учебный корпус, локальная модель, без истории в LLM.

#### 8.4. Проверка недоступной LLM

```powershell
$env:LLM_BASE_URL='http://127.0.0.1:9/v1'
python App/app.py --no-browser --port 7861
```

Ожидается: приложение открывается, статус красный, при вопросе — понятная ошибка,
процесс не падает. После проверки закрыть процесс и восстановить `LLM_BASE_URL`
(если переменная изначально не была задана — `Remove-Item Env:LLM_BASE_URL`).

#### 8.5. Уборка

* остановить Gradio и временные процессы;
* удалить временные логи/скриншоты вне репозитория;
* `git status --short` — только осмысленные изменения;
* `git diff --check` — без trailing whitespace;
* не коммитить `__pycache__`, `Models/`, `*.gguf`, `Eval/answers*.jsonl` (если не менялись намеренно).

---

## 6. Текстовая спецификация ответов UI

| Ситуация | Что видит пользователь |
|---|---|
| Успешный ответ | Markdown-ответ, `[n]`, метаданные, «Источники (N)», карточки |
| Отказ | `Информация не найдена в регламенте.` + caption + найденные фрагменты с пометкой |
| Пустой ответ | Предупреждение «Модель вернула пустой ответ…», источники можно показать |
| LLM offline | Красный статус слева + сообщение в чате и `gr.Warning` с командой запуска |
| Нет индексов | `gr.Error("Не найдены поисковые индексы в Chunks/…")`, без пересборки |
| Контекст обрезан | Caption «В контекст вошли N из top_k; M не поместились…» |
| Долгая генерация | Индикатор выполнения Gradio; очередь не параллелит запросы |
| Trap-пример | Caption «В регламенте нет ответа, ожидается отказ» |
| Пустой вопрос | `gr.Warning("Введите вопрос.")`; сервис дополнительно валидирует |

Все тексты — на русском, кратко, без технического жаргона в основной части.
Технические детали (`chunk_id`, score, latency) — в caption/аккордеоне.

---

## 7. Тест-матрица UI

| # | Сценарий | Действие | Ожидание |
|---|---|---|---|
| 1 | Старт | открыть `/` | Заголовок, панель настроек, пустая история, статус LLM |
| 2 | Факт F2 | вопрос про К-1 | Ответ с давлением и `[1]`, источник `…__001` |
| 3 | Факт F3 | вопрос про мазут | Ответ 390–415 °C, ссылка на источник |
| 4 | Процедура P1 | вопрос про цепочку ЭЛОУ→АТ | Ответ по шагам, несколько источников |
| 5 | Аббревиатура S1 | «параметры АВТ» | Ответ осмысленный, без падения |
| 6 | Числовой N1 | диапазоны фракций | Числа дословно, источник `table_row` |
| 7 | Trap T1 | катализатор гидроочистки | Ровно фраза отказа, без фактов |
| 8 | top_k=3 | сменить и задать вопрос | 3 карточки у нового ответа, остальные скрыты |
| 9 | top_k=10 | сменить и задать вопрос | ≤10 карточек; при обрезке — caption |
| 10 | Abbrev on | включить и задать S1 | Ответ не хуже, источники релевантны |
| 11 | Контекст | включить чекбокс | Аккордеон с блоками `[1]…` |
| 12 | Очистка | «Очистить чат» | История пуста, карточки скрыты |
| 13 | LLM down | остановить сервер, задать вопрос | Сообщение об ошибке, приложение живо, статус красный |
| 14 | LLM up again | «Проверить соединение» | Статус зелёный, вопросы работают |
| 15 | Длинный чанк | top_k=10 | Текст переносится, аккордеон не ломает layout |
| 16 | Кириллица | все ответы/источники | Нет mojibake, UTF-8 |
| 17 | Browser console | вся сессия | Нет ошибок/исключений |
| 18 | Terminal | вся сессия | Нет tracebacks, только ожидаемые warning |

---

## 8. Риски и stop-условия

| Риск | Признак | Действие |
|---|---|---|
| llama-server не запущен | `probe().ok == False` | Не запускать сервер самому; реализовать offline-часть, финальную приёмку заблокировать, сообщить человеку |
| Gradio конфликтует с `numpy/faiss` | `pip check` или импорт падает | Остановиться, подобрать совместимую версию, не ломать текущее окружение; при неудаче — вопрос человеку |
| Модель эмбеддингов грузится при каждом вопросе | В логах повторная загрузка, latency растёт | Проверить `get_runtime()` (lru_cache); не вызывать `hybrid_search` из UI |
| Номера источников не совпадают с `[n]` | В ответе `[4]`, а источников 3 | Вернуться к T1; UI строит источники только из `ContextBlock` |
| Контекст обрезан, источники «лишние» | top_k=10, карточек 10, блоков 6 | T1 + caption об omitted; не показывать выброшенные как источники |
| Дублирование сообщений | Вопрос/ответ появляются дважды | Проверить, что обработчик один раз добавляет пару в `history_state`; не смешивать `gr.State` и локальный список |
| Динамические аккордеоны | Попытка создать компонент в обработчике | Заранее создать 10 карточек, обновлять `visible`/`label`/`value` |
| Порт 7860 занят | Gradio пишет address already in use | Запустить `--port 7861`, указать в отчёте |
| Кириллица ломается в консоли | `????????` | `$env:PYTHONIOENCODING='utf-8'`; файлы UTF-8 без BOM |
| Случайно изменены индексы/артефакты | `git status` показывает `Chunks/`, `Eval/answers*` | Откатить (`git restore`), не коммитить; UI не должен писать в эти папки |
| Ручные метрики всё ещё null | `metrics.json` | Не подставлять числа; сообщить, что это блокер недели 9 |
| Неожиданное расхождение с планом | Код/файлы отличаются от §0 | Остановиться и задать вопрос, не «додумывать» |

---

## 9. Человеческие зависимости (не делает агент)

1. **Запуск llama-server** и проверка модели — человек. Агент только вызывает `probe()`.
2. **Ручная оценка ответов**:
   * заполнить `Eval/manual_scores.json` (25 записей): `answer_accuracy` 0/1/2,
     `hallucination` true/false, комментарий при неудовлетворительной оценке;
   * для сравнения top_k 3/10 нужны `Eval/manual_scores_top3.json` и
     `Eval/manual_scores_top10.json`; сейчас их нет — это отдельная задача перед неделей 9;
   * после заполнения — `python Eval/evaluate.py`, затем проверить `Eval/metrics.*`.
3. **Скриншоты** для пояснительной записки (если нужны) — человек выбирает кадры;
   агент может подготовить приложение и сценарий.
4. **Решение о публикации** репозитория/демо — человек.

**Агенту запрещено:** заполнять оценки «по смыслу», генерировать `manual_scores_*.json`
с непустыми оценками, менять `reference_answer`, подгонять метрики.

---

## 10. План коммитов

Коммиты делать после соответствующих acceptance, не в конце «одним куском».

| # | Сообщение | Содержимое |
|---|---|---|
| 1 | `fix: align sources with llm context` | T1: `prompt.py`, `generate.py`, тест |
| 2 | `feat: cached retrieval runtime` | T2: `Retrieval/runtime.py`, тест |
| 3 | `feat: rag service for web ui` | T3: `Generation/rag_service.py`, `hybrid_retrieval.py`, тест |
| 4 | `feat: gradio chat interface` | T5: `App/app.py`, `App/ui_components.py` |
| 5 | `test: ui smoke and service checks` | T6: `Tests/*`, правки тестов |
| 6 | `docs: week 8 web interface` | T7: `README.md`, `requirements.txt`, этот план |

Перед каждым коммитом:

```powershell
git diff --check
git status --short
```

Не коммитить: `__pycache__/`, `*.pyc`, `Models/`, `*.gguf`, временные скриншоты/логи,
изменённые `Eval/answers*.jsonl`, `Eval/metrics.*` (если они не менялись намеренно).

---

## 11. Итоговый отчёт агента человеку

В конце этапа агент обязан прислать:

1. **Версии:** Python, Gradio, numpy, faiss, sentence-transformers, модель LLM.
2. **Статус LLM:** `probe()` до/после, команда запуска, если сервер не работал.
3. **Изменённые файлы:** список с кратким описанием.
4. **Команды и результаты:** unit-тесты, `compileall`, `pip check`, CLI-регрессии,
   `Eval/evaluate.py`, UI health.
5. **UI-проверка:** какие кейсы из §7 прошли, какие нет; latency первого/повторного вопроса.
6. **Известные ограничения:** manual scores = null, top3/top10 scores отсутствуют,
   история не передаётся в LLM, контекст 9000 символов.
7. **Скриншоты/пути** (если делались).
8. **Что дальше:** неделя 9 — пояснительная записка; перед ней человек заполняет
   `manual_scores.json` и (при необходимости) `manual_scores_top3/top10.json`,
   затем запускается `python Eval/evaluate.py`.
9. **Явный чек-лист Definition of Done** из §1.3 с отметками.

Если какой-то пункт DoD не выполнен — этап **не считается завершённым**, даже если
код написан. В этом случае в отчёте должны быть причина и точный следующий шаг.
