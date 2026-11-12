# ПЛАН РАБОТ ДЛЯ ИИ-АГЕНТА: недели 6–7 (генерация ответа + оценка)

> Этот файл — исполняемое задание для ИИ-агента. Читать целиком перед началом.
> Все решения по стеку и форматам данных уже зафиксированы. Агент не пересматривает их
> без явного указания человека. Если требование противоречит состоянию репозитория —
> остановиться и задать вопрос, а не «додумывать».

---

## 0. Контекст проекта (что уже сделано руками)

Курсовая: RAG-поиск по учебному технологическому регламенту ЭЛОУ-АВТ.
Репозиторий: `C:\My\University\onpz` (Windows, PowerShell, Python 3.12).

Готово (не переделывать):

| Компонент | Путь | Что делает |
|---|---|---|
| Корпус | `Processed data/00…09_*.md` | 10 markdown-документов регламента |
| Чанкинг | `Utils/chunking.py` → `Chunks/chunks_v2_by_fact.jsonl` | 206 чанков, основная стратегия V2 (`by_fact`) |
| BM25 | `Retrieval/bm25_search.py`, `Chunks/bm25_index.pkl` | BM25Okapi + лемматизация pymorphy3 |
| Векторный поиск | `Retrieval/vector_search.py`, `Chunks/faiss_index.bin` | rubert-tiny-turbo + FAISS `IndexFlatIP` |
| Гибрид | `Retrieval/hybrid_retrieval.py` | `hybrid_score = alpha·bm25_norm + (1−alpha)·vector_norm`, `ALPHA_DEFAULT = 0.7`, `CANDIDATE_POOL_SIZE = 20` |
| Подбор alpha | `Retrieval/alpha_tuning.py` → `Eval/alpha_tuning.md` | alpha=0.7, Recall@5 = 13/13 на 13 размеченных запросах |
| Тестовые запросы | `Eval/test_queries.json` | 25 запросов, 5 категорий; `gold_chunk_ids` заполнены только у 13 |
| Прогоны поиска | `Eval/preliminary_results.md`, `Eval/hybrid_results.md` | выдача BM25/vector/hybrid по 25 запросам |

**Чего нет (это и есть задача):** генерации ответа через LLM, полной разметки,
расчёта метрик Recall@5/MRR скриптом, отчёта по неделе 7.

### Схема чанка (вход для генерации)

```json
{
  "chunk_id": "03_reglament_at_rezhim_k1_k2__009",
  "document": "03_reglament_at_rezhim_k1_k2.md",
  "document_title": "Технологический регламент комплекса первичной переработки нефти ЭЛОУ-АВТ",
  "section": "1. Режим работы колонн К-1 и К-2",
  "subsection": "1.2. Режим работы атмосферной колонны К-2",
  "type": "text | list_item | table_row",
  "part": null,
  "list_intro": null,
  "text": "..."
}
```

---

## 1. Стек генерации (зафиксирован, не менять)

| Параметр | Значение |
|---|---|
| Бэкенд | **llama.cpp `llama-server`**, OpenAI-совместимый API |
| Модель | **T-lite-it-2.1 8B**, GGUF `Q4_K_M` (5.03 ГБ), ru-tuned, Qwen3-архитектура, ctx 40K |
| Файл модели | `t-tech/T-lite-it-2.1-GGUF` → `T-lite-it-2.1-Q4_K_M.gguf` |
| Алиас в Ollama | `second_constantine/t-lite-it-2.1:8b` (те же веса, тот же Q4_K_M) |
| Base URL | `http://127.0.0.1:8080/v1` (fallback: `http://127.0.0.1:11434/v1` для Ollama) |
| Эндпоинт | `POST /v1/chat/completions` |
| Контекст сервера | `-c 8192` (в промпт уходит ≤ ~3000 токенов) |
| temperature | `0.1` |
| top_p / top_k | `0.9` / `20` |
| repeat_penalty | `1.05` |
| max_tokens | `512` |
| seed | `42` (фиксируется для воспроизводимости замеров) |
| thinking | **выключен** по умолчанию (`chat_template_kwargs: {"enable_thinking": false}`) |

### Запуск сервера (выполняет человек, агент только проверяет доступность)

```powershell
# Вариант A — llama.cpp (основной)
llama-server -hf t-tech/T-lite-it-2.1-GGUF:Q4_K_M `
  -c 8192 -ngl 99 -fa on --jinja `
  --host 127.0.0.1 --port 8080

# Вариант B — если установлен Ollama
ollama pull second_constantine/t-lite-it-2.1:8b
ollama serve
```

Проверка перед работой:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/v1/models
# ожидается JSON с id модели; если ошибка — СТОП, сообщить человеку
```

Соответствие железу (проверено по замерам T-Bank): Q4_K_M занимает ~5.8 ГБ при ctx 4096,
~7.6 ГБ при ctx 16384. ПК (12 ГБ VRAM) и Mac (18 ГБ unified) тянут `-c 8192` целиком на GPU.

---

## 2. Инварианты (нарушение = провал задачи)

1. **Не менять** существующие публичные функции: `hybrid_search`, `combine_scores`,
   `expand_results`, `load_chunks`, `bm25_search.search`, `vector_search.search`.
   Разрешено только добавлять новые модули и вызывать существующие.
2. **Не пересобирать** индексы BM25/FAISS, не перечанкивать корпус: они зафиксированы
   в `Chunks/` и соответствуют `alpha=0.7` из `Eval/alpha_tuning.md`.
3. **Не выдумывать** значения метрик. Answer Accuracy и Hallucination Rate — ручная
   экспертная оценка: агент готовит шаблон, заполняет человек.
4. **Не ходить в интернет** из кода. Единственный сетевой вызов — `127.0.0.1`.
5. Все текстовые файлы — UTF-8 без BOM. В консоли Windows перед запуском:
   `$env:PYTHONIOENCODING='utf-8'`.
6. Скрипты запускаются **из корня репозитория**. Внутри модулей, которым нужны
   соседние импорты, повторять приём из `Retrieval/alpha_tuning.py`:
   ```python
   sys.path.insert(0, str(Path(__file__).parent))
   sys.path.insert(0, str(Path(__file__).parent.parent / "Retrieval"))
   ```
7. Стиль кода — как в репозитории: type hints, docstring на русском, `pathlib`,
   `argparse`, никаких side-effect при импорте (кроме загрузки модели в `vector_search`).
8. Коммиты — атомарные, сообщения в стиле истории: `feat: llm generation`,
   `feat: full gold labeling`, `feat: evaluation metrics`.

---

## 3. Целевая структура файлов (после выполнения)

```
onpz/
├── PLAN_AI_AGENT.md                 # этот файл (не удалять)
├── README.md                        # обновить: разделы «Генерация» и «Оценка»
├── requirements.txt                 # +requests
├── .gitignore                       # +models/, +*.gguf
├── Generation/                      # НОВОЕ
│   ├── llm_client.py                # клиент llama-server + healthcheck + retries
│   ├── prompt.py                    # system prompt + сборка контекста + блок источников
│   ├── generate.py                  # CLI: --query / --run_all
│   └── __init__.py                  # пустой
├── Retrieval/
│   └── query_expansion.py           # НОВОЕ: словарь аббревиатур (эксперимент ТЗ)
├── Eval/
│   ├── test_queries.json            # ИЗМЕНИТЬ: gold_chunk_ids у всех 25 (+reference_answer)
│   ├── answers.jsonl                # НОВОЕ: машинный вывод генерации (25 строк)
│   ├── generation_results.md        # НОВОЕ: человекочитаемый прогон
│   ├── manual_scores.json           # НОВОЕ: шаблон для ручной оценки ответов
│   ├── evaluate.py                  # НОВОЕ: Recall@5, MRR для 3 методов + сводка ответов
│   ├── metrics.json                 # НОВОЕ: машинные метрики
│   └── metrics.md                   # НОВОЕ: отчёт недели 7
└── models/                          # в .gitignore, вне репозитория
    └── T-lite-it-2.1-Q4_K_M.gguf
```

---

## 4. Задачи по порядку (строго последовательно)

Зависимости: `T1 → T2 → T3 → T4 → T5 → T6 → T7 → T8`.
Задачи T3 и T4 можно делать параллельно только после T2.

---

### T1. Подключение бэкенда: `Generation/llm_client.py`

**Цель:** один тонкий клиент, скрывающий детали llama-server.

**Контракт модуля:**

```python
BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8080/v1")
MODEL = os.getenv("LLM_MODEL", "t-lite-it-2.1")
TIMEOUT = float(os.getenv("LLM_TIMEOUT", "180"))

def healthcheck() -> str:
    """GET /models. Вернуть id модели. При недоступности — RuntimeError
    с текстом: 'LLM недоступна. Запустите llama-server (см. PLAN_AI_AGENT.md, раздел 1).'"""

def chat(messages: list[dict], *, temperature: float = 0.1, max_tokens: int = 512,
         think: bool = False) -> dict:
    """POST /chat/completions. Вернуть {'text': str, 'usage': dict, 'latency_s': float}."""
```

**Требования:**

* `json` + `requests`, `timeout=TIMEOUT`.
* Тело запроса: `model`, `messages`, `temperature`, `top_p=0.9`, `top_k=20`,
  `repeat_penalty=1.05`, `max_tokens`, `seed=42`, `stream=False`,
  `chat_template_kwargs={"enable_thinking": False}` при `think=False`.
* **Fallback:** если сервер вернул 400 на `chat_template_kwargs` (старая сборка/другой
  шаблон) — повторить запрос без этого поля, добавив в начало system-сообщения `/no_think`.
  Факт fallback писать в `stderr`.
* Ретраи: 2 повтора с паузой 2 с только на `ConnectionError` и HTTP 5xx.
* Латентность мерить `time.perf_counter()` вокруг запроса.
* Никаких `print` в stdout из этого модуля — только `logging`/`stderr`, чтобы не ломать
  машиночитаемый вывод.

**Проверка (acceptance):**

```powershell
$env:PYTHONIOENCODING='utf-8'
python -c "import sys; sys.path.insert(0,'Generation'); import llm_client as c; print(c.healthcheck()); print(c.chat([{'role':'user','content':'Ответь одним словом: работает?'}])['text'])"
```

Ожидается осмысленный ответ на русском и `latency_s` < 30.

---

### T2. Промпт и сборка контекста: `Generation/prompt.py`

**Контракт:**

```python
SYSTEM_PROMPT: str
REFUSAL_PHRASE = "Информация не найдена в регламенте."
MAX_CONTEXT_CHARS = 9000

def build_context(results: list[tuple[str, float]], chunk_lookup: dict) -> str
def build_messages(query: str, context: str) -> list[dict]
def format_sources(results: list[tuple[str, float]], chunk_lookup: dict) -> str
def is_refusal(answer: str) -> bool
```

**Формат контекста** — нумерованные блоки, номер = ссылка `[n]`:

```
[1] chunk_id=03_reglament_at_rezhim_k1_k2__009 | document=03_reglament_at_rezhim_k1_k2.md | section=... | subsection=...
<текст чанка дословно>

[2] ...
```

**Текст `SYSTEM_PROMPT` (обязательные пункты, формулировки можно слегка причесать):**

* роль: инженер-технолог НПЗ, работаешь с технологическим регламентом ЭЛОУ-АВТ;
* отвечать **только** по блокам контекста, внешние знания запрещены;
* если ответа нет — ответить ровно `Информация не найдена в регламенте.` и ничего больше;
* после каждого утверждения ставить номер источника: `[1]`, `[2]` (номера только из контекста);
* числа, единицы, температуры, давления приводить дословно, как в контексте, не округлять
  и не пересчитывать;
* если данные в разных блоках противоречат — указать оба значения с ссылками;
* язык ответа — русский, стиль — краткий технический, без вступлений и извинений;
* не пересказывать вопрос, не добавлять разделы «Введение»/«Заключение».

**User-сообщение:**

```
Контекст:
{context}

Вопрос: {query}
```

**`format_sources`** формируется **кодом, а не моделью** (защита от выдуманных ссылок):

```markdown
**Источники:**
1. `03_reglament_at_rezhim_k1_k2__009` — 03_reglament_at_rezhim_k1_k2.md, 1.2. Режим работы атмосферной колонны К-2
2. ...
```

**`build_context`:** обрезать по `MAX_CONTEXT_CHARS` **по границе блока** (не посередине
чанка); блоки нумеровать по порядку после обрезки.

**`is_refusal`:** `True`, если нормализованный (lower, strip, без знаков препинания) ответ
начинается с `информация не найдена`.

**Проверка:** unit-прогон без сети — собрать контекст из 6 фейковых чанков, убедиться,
что нумерация сквозная, обрезка не рвёт блок, `format_sources` даёт 6 строк.

---

### T3. Полная разметка: `Eval/test_queries.json`

**Цель:** закрыть риск из ТЗ «Нет разметки релевантных чанков».

**Что сделать:**

1. Для **всех** 25 запросов заполнить `gold_chunk_ids` (сейчас пусто у 12:
   P1–P5, N1–N4, T1–T3).
2. Для `procedural` (P1–P5) в `gold_chunk_ids` — **все** чанки, которые вместе образуют
   ответ (обычно 2–5 `list_item` из одной группы `list_intro`, порядок — по `part`).
3. Для `numeric_tabular` (N1–N4) — чанки типа `table_row`, где лежит нужное число.
4. Для `trap` (T1–T3) — `gold_chunk_ids: []` **оставить пустым** и добавить
   `"expected_behavior": "refusal"`.
5. Добавить во **все** записи поле `reference_answer` (эталонный ответ 1–3 предложения,
   собранный строго из `gold_chunk_ids`; для trap — `Информация не найдена в регламенте.`).
6. Для запросов категории 3 добавить `"abbreviations": {"ЭЛОУ": "...", ...}` — короткое
   раскрытие термина, которое встречается в документе.

**Правила поиска gold-чанков:** только по содержимому `Chunks/chunks_v2_by_fact.jsonl`
(не по памяти, не по `Processed data/`), каждый id проверять на существование:

```powershell
python -c "import json; ids={c['chunk_id'] for c in map(json.loads, open('Chunks/chunks_v2_by_fact.jsonl',encoding='utf-8'))}; import json as j; q=j.load(open('Eval/test_queries.json',encoding='utf-8')); bad=[(x['query_id'],g) for x in q for g in x.get('gold_chunk_ids',[]) if g not in ids]; print('BAD:',bad)"
```

**Acceptance:**

* у 25/25 запросов поле `gold_chunk_ids` присутствует;
* у 22 запросов (все, кроме trap) оно непустое;
* `reference_answer` у 25/25;
* проверка выше печатает `BAD: []`;
* JSON валиден, порядок запросов и `query_id` сохранены.

**Обязательно:** после изменения файла перегенерировать `Eval/alpha_tuning.md`
(`python Retrieval/alpha_tuning.py`) — теперь Recall@5 считается по 22 запросам.
Если `ALPHA_DEFAULT` перестал быть лучшим — **не менять код**, а зафиксировать факт
в отчёте и сообщить человеку.

---

### T4. CLI генерации: `Generation/generate.py`

**Интерфейс:**

```powershell
python Generation/generate.py --query "Какое давление в колонне К-1?"
python Generation/generate.py --run_all --top_k 5 --tag baseline
python Generation/generate.py --run_all --top_k 3 --tag top3
python Generation/generate.py --query "..." --think          # контрольный режим
```

Флаги: `--query`, `--run_all`, `--top_k` (default 5), `--alpha` (default 0.7),
`--tag` (default `baseline`, попадает в имена артефактов и в JSONL),
`--think` (включает рассуждения), `--show_context` (печатать контекст перед ответом),
`--expand_abbrev` (включает `Retrieval/query_expansion.py`, см. T8).

**Поток для одного запроса:**

1. `healthcheck()` — при ошибке выйти с кодом 2 и понятным сообщением.
2. `hybrid_search(query, top_k=top_k, alpha=alpha)`.
3. Если среди результатов есть `type == "list_item"` — **дополнить контекст**
   соседями по группе через `expand_results(build_list_groups(...))` из
   `hybrid_retrieval.py`. Дополненные чанки идут **после** top_k и **не нумеруются
   как отдельные источники**, а печатаются отдельным блоком
   «Развёрнутый контекст процедуры» (как в `hybrid_retrieval.py`).
   В контекст LLM они попадают внутри блока того источника, к которому относятся.
4. `build_context` → `build_messages` → `chat()`.
5. Вывод в stdout:

```
ВОПРОС: ...
ОТВЕТ:
<текст модели>

**Источники:**
1. `chunk_id` — document, section / subsection
...
[latency: 4.2 s | tokens: 812 in / 156 out | tag=baseline]
```

**Поток `--run_all`:**

* читает `Eval/test_queries.json` (25 запросов, порядок сохранять);
* пишет **две** вещи:
  * `Eval/answers.jsonl` — одна JSON-строка на запрос (схема ниже);
  * `Eval/generation_results.md` — разделы `## <query_id> [<category>] <query_text>`,
    затем контекст (свёрнуто: только `chunk_id` + score), ответ, источники, метрики.
    Имя при `--tag != baseline`: `Eval/generation_results_<tag>.md`,
    `Eval/answers_<tag>.jsonl`.
* при `--tag != baseline` **не перезаписывать** baseline-артефакты.

**Схема строки `answers.jsonl` (порядок ключей фиксирован):**

```json
{
  "query_id": "F2",
  "category": "factual",
  "query_text": "...",
  "tag": "baseline",
  "retrieval": {
    "method": "hybrid",
    "alpha": 0.7,
    "top_k": 5,
    "expand_abbrev": false,
    "chunks": [{"rank": 1, "chunk_id": "...", "score": 0.703}]
  },
  "llm": {"model": "t-lite-it-2.1", "temperature": 0.1, "think": false, "seed": 42},
  "prompt_chars": 4210,
  "answer": "...",
  "refused": false,
  "usage": {"prompt_tokens": 812, "completion_tokens": 156},
  "latency_s": 4.2,
  "timestamp": "2026-11-12T14:03:11"
}
```

**Обработка ошибок:**

* LLM недоступна → код выхода 2, без частичных артефактов;
* ошибка на одном запросе в `--run_all` → записать строку с `"error": "<текст>"`,
  продолжить остальные, в конце вернуть код 1 и напечатать сводку `N ok / M error`;
* пустой ответ модели → `"answer": ""`, `"refused": false`, предупреждение в stderr.

**Acceptance:**

* `python Generation/generate.py --query "..."` даёт ответ с валидными `[n]`, и все `n`
  существуют в блоке «Источники»;
* `--run_all` создаёт 25 строк в `answers.jsonl`, JSONL парсится построчно без ошибок;
* у F2/F3/F5 в `retrieval.chunks[0].chunk_id` тот же чанк, что в `Eval/hybrid_results.md`;
* время прогона 25 запросов ≤ 15 минут.

---

### T5. Ручная оценка: шаблон `Eval/manual_scores.json`

**Агент создаёт шаблон, человек заполняет.** Агент **не ставит оценки сам**.

Сгенерировать из `answers.jsonl` + `test_queries.json`:

```json
[
  {
    "query_id": "F1",
    "category": "factual",
    "reference_answer": "...",
    "model_answer": "...",
    "answer_accuracy": null,
    "hallucination": null,
    "comment": ""
  }
]
```

Правила заполнения (вписать в шапку файла нельзя — JSON, поэтому правила идут в
`Eval/metrics.md`):

* `answer_accuracy`: `2` — верно и полно, `1` — частично верно/неполно, `0` — неверно или
  необоснованный отказ; для `trap` корректный отказ = `2`.
* `hallucination`: `true`, если в ответе есть факт/число/единица, которых нет в
  переданном контексте (сверять с `retrieval.chunks` и текстом чанков); иначе `false`.
* `comment` — обязателен при `answer_accuracy != 2` или `hallucination == true`.

**Acceptance:** файл создан, 25 записей, `model_answer` совпадает с `answers.jsonl`,
все оценки `null`.

---

### T6. Расчёт метрик: `Eval/evaluate.py`

**Что считает:**

1. **Retrieval-метрики** для трёх методов на размеченных запросах
   (все, у кого `gold_chunk_ids` непусто, т.е. 22):
   * BM25 (`alpha=1.0`), Vector (`alpha=0.0`), Hybrid (`alpha=0.7`) — через
     `bm25_search.search`, `vector_search.search`, `combine_scores`;
   * **Recall@5** = доля запросов, где ≥1 gold-чанк в топ-5;
   * **MRR@5** = среднее по запросам `1 / rank` первого gold-чанка
     (`0`, если gold-чанка нет в топ-5). Формула округляется до 3 знаков.
2. **Сводку по ответам** из `answers.jsonl` + `manual_scores.json`:
   * `refusal_rate` — доля `refused == true`;
   * `refusal_correctness` — по trap-запросам: доля корректных отказов;
   * `answer_accuracy_mean` — среднее `answer_accuracy` по шкале 0/1/2
     (для отчёта также в процентах: `mean / 2 * 100`);
   * `hallucination_rate` — доля `hallucination == true` среди всех ответов.
3. **Сравнение top_k (ТЗ: 3 vs 5 vs 10)** — если существуют
   `answers_top3.jsonl` / `answers_top10.jsonl`, добавить таблицу
   `answer_accuracy_mean` и `hallucination_rate` по тегам.
4. **Сравнение «до/после раскрытия аббревиатур»** — если существует
   `answers_abbrev.jsonl`, добавить строку с Recall@5 и MRR для него
   (сам retrieval-прогон делает T8).

**Выходные файлы:**

* `Eval/metrics.json` — вся структура чисел;

```json
{
  "retrieval": {
    "bm25":   {"recall_at_5": 0.0, "mrr_at_5": 0.0, "n_queries": 22},
    "vector": {"recall_at_5": 0.0, "mrr_at_5": 0.0, "n_queries": 22},
    "hybrid": {"recall_at_5": 0.0, "mrr_at_5": 0.0, "n_queries": 22}
  },
  "generation": {"refusal_rate": 0.0, "answer_accuracy_mean": 0.0,
                 "hallucination_rate": 0.0, "n_answers": 25,
                 "trap_refusal_correctness": 0.0}
}
```

* `Eval/metrics.md` — отчёт по шаблону из раздела 5.

**Правила:**

* Если `manual_scores.json` содержит `null` хотя бы в одной оценке — **не падать**:
  писать в `metrics.json` `null`, а в `metrics.md` — «требуется ручная оценка».
* Числа не хардкодить, всё считать из файлов.
* Детерминизм: один и тот же вход → один и тот же выход (сортировка запросов по `query_id`
  при выводе, округление до 3 знаков).

**Acceptance:**

```powershell
python Eval/evaluate.py
```

печатает таблицу метрик, создаёт `Eval/metrics.json` и `Eval/metrics.md`;
`hybrid.recall_at_5` на 22 запросах ≥ `bm25` и ≥ `vector` (иначе — отметить в отчёте,
не подгонять).

---

### T7. Документация

1. **`README.md`** — добавить два раздела:
   * «Генерация ответа (неделя 6)»: модель, как поднять `llama-server`, env-переменные,
     команды запуска, формат артефактов;
   * «Оценка (неделя 7)»: список метрик, команды, куда смотреть результаты.
   Обновить дерево репозитория и раздел «План работы» (отметить недели 6–7).
2. **`requirements.txt`** — добавить `requests==2.32.3` (или актуальную стабильную).
   Зависимости `llama.cpp` в requirements **не входят** — он ставится отдельно.
3. **`.gitignore`** — добавить `models/` и `*.gguf` (веса не коммитить).
4. **`PLAN_AI_AGENT.md`** — в конце отметить выполненные задачи (галочки), не удалять.

---

### T8. Эксперимент с аббревиатурами (требование ТЗ, категория 3)

**`Retrieval/query_expansion.py`:**

```python
ABBREVIATIONS: dict[str, str] = {
    "ЭЛОУ": "электрообессоливающая установка обессоливание обезвоживание",
    "АВТ": "атмосферно-вакуумная трубчатка перегонка нефти",
    "АТ": "атмосферная перегонка",
    "ВТ": "вакуумная перегонка",
    "НПЗ": "нефтеперерабатывающий завод",
    "гудрон": "тяжёлый остаток вакуумной перегонки",
    "вакуумный газойль": "лёгкий дистиллят вакуумной перегонки",
    "разрежение": "остаточное давление вакуум",
}

def expand_query(query: str) -> str:
    """Дописать к запросу раскрытия найденных аббревиатур/синонимов (без дублей)."""
```

**Правила:** регистронезависимый поиск; раскрытие дописывается в конец через пробел;
исходный текст запроса сохраняется первым.

**Прогон:**

```powershell
python Generation/generate.py --run_all --expand_abbrev --tag abbrev
```

**Acceptance:** `Eval/answers_abbrev.jsonl` существует; в `metrics.md` есть строка
сравнения «базовый vs с раскрытием аббревиатур» по Recall@5/MRR (считать в T6).

**Важно:** если прирост нулевой или отрицательный — это **нормальный результат**, зафиксировать
как есть. Подгонять словарь под тестовые запросы запрещено (утечка разметки).

---

## 5. Шаблон `Eval/metrics.md` (обязательная структура)

```markdown
# Оценка качества (неделя 7)

## 1. Поиск: BM25 vs Vector vs Hybrid

| Метод | Recall@5 | MRR@5 | Запросов |
|---|---|---|---|
| BM25 (alpha=1.0) | | | 22 |
| Vector (alpha=0.0) | | | 22 |
| Hybrid (alpha=0.7) | | | 22 |

## 2. Генерация ответов (T-lite-it-2.1, Q4_K_M, top_k=5)

| Метрика | Значение |
|---|---|
| Отказов (refused) | |
| Корректных отказов на trap (T1–T3) | |
| Answer Accuracy (среднее 0/1/2) | |
| Answer Accuracy, % | |
| Hallucination Rate | |

## 3. Влияние top_k (3 vs 5 vs 10)

| top_k | Answer Accuracy | Hallucination Rate |
|---|---|---|

## 4. Раскрытие аббревиатур (категория 3)

| Вариант | Recall@5 | MRR@5 |
|---|---|---|

## 5. Анализ ошибок

Для каждого запроса с `answer_accuracy < 2` или `hallucination = true`:
ID, категория, что не так (retrieval miss / неверная интерпретация / галлюцинация / отказ),
цитата проблемного фрагмента.

## 6. Выводы

3–5 пунктов: что подтвердилось, где гибрид проиграл, какие ограничения у прототипа.
```

---

## 6. Порядок проверки (чек-лист Definition of Done)

Выполнять сверху вниз, каждый пункт — команда + ожидание.

| # | Проверка | Ожидание |
|---|---|---|
| 1 | `Invoke-RestMethod http://127.0.0.1:8080/v1/models` | 200, id модели |
| 2 | `python -c "...llm_client.healthcheck()..."` | имя модели, без traceback |
| 3 | `python Generation/generate.py --query "Какое давление поддерживается в отбензинивающей колонне К-1?"` | ответ про К-1 с `[n]`, источники непустые |
| 4 | `python Generation/generate.py --query "Какова стоимость строительства новой ЭЛОУ-АВТ на ОНПЗ?"` | ровно «Информация не найдена в регламенте.» (trap) |
| 5 | `python Generation/generate.py --run_all` | 25 строк в `Eval/answers.jsonl`, `Eval/generation_results.md` |
| 6 | Проверка gold: команда из T3 | `BAD: []` |
| 7 | `python Retrieval/alpha_tuning.py` | отчёт пересчитан на 22 запросах |
| 8 | `python Eval/evaluate.py` | `Eval/metrics.md`, `Eval/metrics.json`, таблица в консоли |
| 9 | `python Generation/generate.py --run_all --top_k 3 --tag top3` | `answers_top3.jsonl` |
| 10 | `python Generation/generate.py --run_all --top_k 10 --tag top10` | `answers_top10.jsonl` |
| 11 | `python Generation/generate.py --run_all --expand_abbrev --tag abbrev` | `answers_abbrev.jsonl` |
| 12 | `git status` | нет `*.gguf`, нет `models/`, нет `__pycache__` |
| 13 | `python -m json.tool Eval/test_queries.json > $null` | без ошибок |
| 14 | `python -c "import json; [json.loads(l) for l in open('Eval/answers.jsonl',encoding='utf-8')]"` | без ошибок |

Пункты 9–11 — только если человек подтвердил наличие времени/железа; в приоритете 1–8.

---

## 7. Запрещённые действия (типичные ошибки агента)

1. ❌ Пересобирать FAISS/BM25 или менять `chunks_v2_by_fact.jsonl` «для улучшения метрик».
2. ❌ Менять `alpha` без нового прогона `alpha_tuning.py` и явного подтверждения.
3. ❌ Подавать в промпт больше 5 источников (кроме осознанного прогона `--top_k 10`).
4. ❌ Просить LLM самой перечислить источники вместо `format_sources`.
5. ❌ Считать `gold_chunk_ids` «по названию файла» без проверки содержимого чанка.
6. ❌ Заполнять `manual_scores.json` самостоятельно или выводить метрики генерации
   при пустых оценках как числа (только `null` + пометка).
7. ❌ Добавлять в промпт `reference_answer` (утечка эталона).
8. ❌ Логировать полный контекст в stdout при `--run_all` (ломает читаемость отчёта).
9. ❌ Коммитить `models/`, `*.gguf`, `__pycache__/`.
10. ❌ Ставить `temperature` выше 0.3 для этой задачи.

---

## 8. Риски и заранее заготовленные реакции

| Риск | Признак | Действие |
|---|---|---|
| Модель «отвечает по памяти» | ответ содержит факты вне контекста | усилить SYSTEM_PROMPT, `--top_k 5`, прогнать trap-тесты, зафиксировать в «Анализе ошибок» |
| Ломается chat template | бессвязный вывод, повторение промпта | проверить, что сервер запущен с `--jinja`; при необходимости `/no_think` в system |
| `chat_template_kwargs` не поддерживается | HTTP 400 | fallback из T1 (повтор без поля) |
| Модель не влезает в 12 ГБ | в логе `llama-server` offload слоёв в CPU | `-c 4096`, `-ngl 99` оставить, Q4_K_M не менять |
| Q4 теряет числа из таблиц | путает 180–220 °C и 220–280 °C | это законный результат: отметить в ошибках, для сравнения можно Q5_K_M/Q6_K (5.85/6.73 ГБ) |
| Гибрид хуже BM25 на 22 запросах | `hybrid.recall_at_5 < bm25` | не подгонять alpha; описать в выводах, предложить переранжирование как future work |
| Разметка субъективна | расходятся gold-чанки | для procedural допускается несколько gold-чанков; фиксировать правило в комментарии поля |

---

## 9. Быстрый справочник команд

```powershell
$env:PYTHONIOENCODING='utf-8'

# сервер (отдельное окно, запускает человек)
llama-server -hf t-tech/T-lite-it-2.1-GGUF:Q4_K_M -c 8192 -ngl 99 -fa on --jinja --host 127.0.0.1 --port 8080

# один запрос
python Generation/generate.py --query "До какой температуры нагревается мазут перед вакуумной колонной?"

# базовый прогон + метрики
python Generation/generate.py --run_all
python Eval/evaluate.py

# эксперименты ТЗ
python Generation/generate.py --run_all --top_k 3  --tag top3
python Generation/generate.py --run_all --top_k 10 --tag top10
python Generation/generate.py --run_all --expand_abbrev --tag abbrev

# пересчёт разметки
python Retrieval/alpha_tuning.py
```

---

## 10. Отчёт о выполнении (заполняет агент в конце)

```
- [x] T1 llm_client.py
- [x] T2 prompt.py
- [x] T3 полная разметка 25/25
- [x] T4 generate.py (--query, --run_all)
- [x] T5 manual_scores.json (шаблон)
- [x] T6 evaluate.py + metrics.md/metrics.json
- [x] T7 README / requirements / .gitignore
- [x] T8 query_expansion.py + прогон abbrev
- [x] Чек-лист раздела 6, пункты 1–8
- [ ] Коммиты: feat: llm generation / full gold labeling / evaluation metrics

Итог прогонов: 25/25 ответов без ошибок, 3/3 корректных отказа на trap,
Recall@5 = 1.000 и MRR@5 = 0.871 у гибридного поиска на 22 размеченных запросах.
Коммиты пользователь делает самостоятельно.

Что осталось человеку: заполнить manual_scores.json, проверить metrics.md,
затем неделя 8 — веб-интерфейс (Gradio/Streamlit).
```
