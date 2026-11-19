# RAG-поиск по технологическому регламенту установки ЭЛОУ-АВТ

Курсовая работа на тему **«Семантический поиск по технологическому регламенту атмосферно-вакуумной перегонки нефти»**. Реализован прототип системы нейронного поиска (RAG), которая по вопросу на естественном языке находит релевантные фрагменты технологического регламента комплекса первичной переработки нефти ЭЛОУ-АВТ и формирует связный ответ со ссылкой на источник.

## Задача

На Омском НПЗ расхождение между планом и фактом производства основного ассортимента продуктов достигает до 18 % по массе в месяц. В рамках цифровой трансформации требуются подходы формирования операционных заданий на уровень управления техпроцессом. Установка ЭЛОУ-АВТ — головной процесс любого НПЗ (комплекс мощностью 8,4 млн т/год по нефти и 1,2 млн т/год по стабильному газовому конденсату, введён в эксплуатацию в 2019 году).

**Цель работы:** построить прототип RAG-системы, которая по вопросу на естественном языке находит релевантные фрагменты технологического регламента установки ЭЛОУ-АВТ и формирует связный ответ со ссылкой на источник.

## Структура репозитория

```
onpz/
├── MES_ТЗ_2_на_RAG_подходы_на_примере_АВТ.docx   # Техническое задание на курсовую работу
├── PLAN_AI_AGENT.md                                 # План работ недели 8 (веб-интерфейс на Gradio)
├── README.md
├── requirements.txt
├── run_app.py                                       # Единый запуск llama-server + Gradio (Windows/macOS)
├── run_app.ps1                                      # Обёртка для Windows
├── run_app.sh                                       # Обёртка для macOS/Linux
├── .gitignore
├── App/                                             # Веб-интерфейс (неделя 8)
│   ├── app.py                                       # Точка входа Gradio
│   ├── ui_components.py                             # Источники, статус LLM, метаданные
│   └── metrics_view.py                              # Read-only вкладка метрик
├── Utils/
│   └── chunking.py                                  # Скрипт чанкинга (V1 и V2)
├── Raw data/                                        # Исходные материалы (.pdf, .doc, .docx)
│   ├── Технологическая схема АВТ-6 (Studfile).doc
│   ├── Технологический регламент и описание аппаратов ЭЛОУ-АВТ (Документ).pdf
│   ├── Переработка Нефти и Газа(ЭЛОУ-АВТ)-Отчет по пра....docx
│   ├── Расчет и параметры ректификационной колонны К-2 (Studfile).doc
│   ├── Техническое описание и регламент безопасности установок ЭЛОУ-АВТ.pdf
│   └── Rasch_elektrodegidr.pdf
├── Processed data/                                  # Обработанный корпус (10 markdown-файлов 00–09)
│   ├── 00_reglament_obzor_eloy_avt.md
│   ├── 01_reglament_eloy.md
│   ├── 02_reglament_at_shema.md
│   ├── 03_reglament_at_rezhim_k1_k2.md
│   ├── 04_reglament_vt_vakuum.md
│   ├── 05_reglament_stabilizaciya.md
│   ├── 06_reglament_material_balans.md
│   ├── 07_reglament_oborudovanie.md
│   ├── 08_reglament_tb.md
│   └── 09_reglament_operativnoe_upravlenie.md
├── Chunks/                                          # Результаты чанкинга и индексы
│   ├── chunks_v1_by_section.jsonl   (45 чанков)
│   ├── chunks_v2_by_fact.jsonl      (206 чанков)
│   ├── chunking_report.txt
│   ├── bm25_index.pkl
│   ├── faiss_index.bin
│   └── faiss_chunk_ids.json
├── Retrieval/                                       # Поиск
│   ├── bm25_search.py
│   ├── vector_search.py
│   ├── hybrid_retrieval.py
│   ├── runtime.py                                   # Кэш индексов и модели эмбеддингов
│   ├── alpha_tuning.py
│   ├── query_expansion.py                           # Раскрытие аббревиатур в запросе
│   └── retrieval.py
├── Generation/                                      # Генерация ответа (неделя 6)
│   ├── llm_client.py                                # Клиент llama-server
│   ├── prompt.py                                    # Промпт и сборка контекста
│   ├── rag_service.py                               # Сервисный слой retrieve/ask
│   └── generate.py                                  # CLI генерации
├── Tests/                                           # Unit-тесты (unittest)
│   ├── test_prompt_context.py
│   ├── test_runtime_parity.py
│   ├── test_rag_service.py
│   ├── test_ui_components.py
│   └── test_launcher.py
├── Eval/                                            # Разметка, прогоны и оценка (неделя 7)
│   ├── test_queries.json                            # 25 запросов с gold_chunk_ids
│   ├── alpha_tuning.md
│   ├── preliminary_results.md
│   ├── hybrid_results.md
│   ├── answers.jsonl                                # Машинный вывод генерации
│   ├── answers_top3.jsonl
│   ├── answers_top10.jsonl
│   ├── answers_abbrev.jsonl
│   ├── generation_results.md
│   ├── generation_results_top3.md
│   ├── generation_results_top10.md
│   ├── generation_results_abbrev.md
│   ├── manual_scores.json                           # Шаблон ручной оценки
│   ├── evaluate.py                                  # Recall@5, MRR@5, сводка
│   ├── metrics.json
│   └── metrics.md
└── Models/                                          # Веса LLM, в .gitignore
    └── T-lite-it-2.1-Q4_K_M.gguf
```

## Стек технологий

| Компонент | Выбор | Альтернатива |
|-----------|-------|--------------|
| Язык | Python 3.10+ | – |
| Чанкинг | langchain RecursiveCharacterTextSplitter | ручной разбор по разделам |
| Эмбеддинги | sentence-transformers (all-MiniLM-L6-v2 / rubert-tiny) | OpenAI text-embedding-3-small |
| Векторная БД | FAISS / ChromaDB | Qdrant |
| BM25 | rank_bm25 | elasticsearch |
| LLM | llama.cpp + T-lite-it-2.1 8B (Q4_K_M) | transformers / внешний API |
| Интерфейс | Gradio (локальный веб-интерфейс) | Streamlit |
| Оценка | scikit-learn, numpy | – |

## План работы (10 недель)

1. ✅ Анализ предметной области, сбор открытых источников, структура учебного регламента.
2. ✅ Подготовка корпуса: 10 markdown-документов с разметкой структуры.
3. ✅ Чанкинг: нарезка, сохранение метаданных (206 чанков V2 `by_fact`).
4. ✅ BM25-поиск и векторный индекс (FAISS), первый прогон.
5. ✅ Гибридный поиск BM25 + векторный, подбор веса `alpha = 0.7`.
6. ✅ Подключение LLM: промпт-инжиниринг, генерация ответов с контекстом и ссылками.
7. ✅ Оценка: Recall@5, MRR, Answer Accuracy, Hallucination Rate.
8. ✅ Веб-интерфейс на Gradio: чат, источники, метрики; проверен с запущенным llama-server.
9. ⬜ Пояснительная записка (25–35 стр.).
10. ⬜ Презентация и защита.

## Метрики оценки

* **Recall@5** — доля вопросов, для которых релевантный чанк попал в топ-5.
* **MRR** — средняя позиция первого релевантного чанка.
* **Answer Accuracy** — доля правильных ответов LLM (шкала 0/1/2).
* **Hallucination Rate** — доля ответов с фактами, отсутствующими в контексте.

## Тестовые запросы

Набор из 25 запросов разных категорий: фактологические, процедурные, по терминологии, числовые, синонимичные, «ловушки».
Примеры:
* При какой температуре отбирается керосиновая фракция на атмосферной колонне?
* Какое давление поддерживается в отбензинивающей колонне К-1?
* До какой температуры нагревается мазут перед подачей в вакуумную колонну?
* Сколько ступеней обессоливания в блоке ЭЛОУ?

## Чанкинг (Utils/chunking.py)

Реализованы две стратегии нарезки markdown-корпуса с сохранением технологической структуры (заголовки `##`/`###`, таблицы, списки):

* **V1 `by_section`** — чанк = целый раздел/подраздел/таблица (при превышении 800 токенов — деление по абзацам с сохранением заголовка в метаданных). Получено **45 чанков**.
* **V2 `by_fact`** — более мелкая нарезка: каждый пункт списка (>= 3 пунктов) и каждая строка таблицы — отдельный чанк; обычный текст режется по абзацам до 220 токенов. Получено **206 чанков** (целевой диапазон по ТЗ — 150–250, достигнут).

Метаданные чанка: `document`, `document_title`, `section`, `subsection`, `type`, `part`, `list_intro`, `token_estimate`, `char_count`.

Результаты сохранены в `Chunks/chunks_v1_by_section.jsonl` и `Chunks/chunks_v2_by_fact.jsonl`, подробный отчёт — в `Chunks/chunking_report.txt`.

Запуск чанкинга:
```bash
python Utils/chunking.py
```

## Retrieval

Реализованы `Retrieval/bm25_search.py` (BM25Okapi + лемматизация `pymorphy3`) и `Retrieval/vector_search.py` (эмбеддинги `sergeyzh/rubert-tiny-turbo` + FAISS `IndexFlatIP`), индексы сохранены в `Chunks/`. Скрипт `Retrieval/retrieval.py` запускает оба метода по CLI (`--query`/`--run_all`) и сохраняет прогон всех 25 тестовых запросов в `Eval/preliminary_results.md`. Тестовые запросы из ТЗ вынесены в `Eval/test_queries.json` (25 шт., 5 категорий).

## Гибридный поиск

Реализован `Retrieval/hybrid_retrieval.py`: кандидаты от BM25 и векторного поиска (top-20) нормализуются min-max в [0, 1] и объединяются взвешенной суммой `hybrid_score = alpha * bm25_norm + (1 - alpha) * vector_norm`.

Вес `alpha` подобран на ручной разметке `gold_chunk_ids` (22 из 25 запросов, все категории, кроме `trap`, см. `Eval/alpha_tuning.md`): сравнивался Recall@5 при `alpha ∈ {0.0, 0.3, 0.5, 0.7, 1.0}`. Выбрано **`alpha = 0.7`** — полный Recall@5 (22/22) на размеченных запросах при сохранении вклада векторного метода (в отличие от `alpha = 1.0`, эквивалентного чистому BM25).

Прогон гибридного метода по всем 25 тестовым запросам сохранён в `Eval/hybrid_results.md` (сравнение с `Eval/preliminary_results.md` — материал для недели 7).

После полной разметки 22 нетрап-запросов отчёт `Eval/alpha_tuning.md` пересчитан: `alpha = 0.7` даёт Recall@5 = 22/22 = 1.000 (у `alpha = 1.0` тот же Recall при более низком MRR гибрида).

## Генерация ответа

Ответ синтезируется локальной LLM **T-lite-it-2.1 8B** (GGUF `Q4_K_M`, архитектура Qwen3, русскоязычная настройка) через `llama-server` с OpenAI-совместимым API. Веса лежат в `Models/T-lite-it-2.1-Q4_K_M.gguf` и не коммитятся.

Для одновременного запуска LLM и веб-интерфейса используйте `python run_app.py` (см. раздел «Запуск всего стека одной командой»). Ниже — ручной запуск сервера (llama.cpp устанавливается отдельно):

```powershell
llama-server -m "Models/T-lite-it-2.1-Q4_K_M.gguf" `
  -c 8192 -ngl 99 -fa on --jinja `
  --host 127.0.0.1 --port 8080
```

Если используется Ollama, подойдёт алиас `second_constantine/t-lite-it-2.1:8b`; клиент обратится к `http://127.0.0.1:11434/v1`.

Параметры клиента задаются переменными окружения:

| Переменная | Значение по умолчанию | Назначение |
|---|---|---|
| `LLM_BASE_URL` | `http://127.0.0.1:8080/v1` | адрес OpenAI-совместимого API |
| `LLM_MODEL` | `t-lite-it-2.1` | имя модели в запросе |
| `LLM_TIMEOUT` | `180` | таймаут одного запроса, секунд |

Перед запуском в PowerShell:

```powershell
$env:PYTHONIOENCODING='utf-8'
```

Примеры команд:

```powershell
python Generation/generate.py --query "Какое давление поддерживается в отбензинивающей колонне К-1?"
python Generation/generate.py --query "..." --show_context
python Generation/generate.py --run_all
python Generation/generate.py --run_all --top_k 3 --tag top3
python Generation/generate.py --run_all --top_k 10 --tag top10
python Generation/generate.py --run_all --expand_abbrev --tag abbrev
```

Как устроена генерация:

* гибридный поиск возвращает top_k источников (`alpha = 0.7`);
* для `list_item`-чанков в контекст добавляются соседи по группе процедуры, но отдельными источниками они не нумеруются;
* контекст ограничен 9000 символами и обрезается по границе блока;
* inline-формулы LaTeX в контексте преобразуются в обычную текстовую запись;
* модель обязана ссылаться на источники `[n]`, а блок «Источники» формируется кодом, а не моделью;
* ссылки на источники, которых нет в контексте, удаляются из ответа после генерации;
* ответ ограничен 150 словами, чтобы модель не пересказывала весь контекст;
* при отсутствии ответа модель возвращает ровно `Информация не найдена в регламенте.`;
* `temperature = 0.1`, `seed = 42`, thinking выключен по умолчанию (флаг `--think` включает его для контрольных прогонов).

Артефакты:

* `Eval/answers.jsonl` — по одной JSON-строке на запрос (retrieval, ответ, usage, latency);
* `Eval/generation_results.md` — человекочитаемый отчёт по 25 запросам;
* при `--tag <tag>` создаются `Eval/answers_<tag>.jsonl` и `Eval/generation_results_<tag>.md`, baseline-файлы не перезаписываются.

## Запуск всего стека одной командой (Windows и macOS)

`run_app.py` запускает сразу оба процесса: локальный `llama-server` и веб-интерфейс Gradio. Скрипт сам находит модель и `llama-server`, ждёт готовности LLM, открывает браузер, а по `Ctrl+C` останавливает то, что запустил. Внешние API и телеметрия не используются.

### Windows

```powershell
cd C:\My\University\onpz
python -m pip install -r requirements.txt
python run_app.py
```

Или через обёртку:

```powershell
.\run_app.ps1
```

### macOS

```bash
cd /path/to/onpz
python3 -m pip install -r requirements.txt
brew install llama.cpp        # если llama-server ещё не установлен
python3 run_app.py
```

Или через обёртку:

```bash
chmod +x run_app.sh
./run_app.sh
```

### Что делает лаунчер

1. Проверяет Python-зависимости (`gradio`, `requests`).
2. Ищет `llama-server` в `PATH`, стандартных каталогах Homebrew и рядом с репозиторием.
3. Проверяет, что GGUF-модель существует.
4. Если `llama-server` уже отвечает на `/health`, переиспользует его и не запускает второй процесс.
5. Запускает `llama-server` на `127.0.0.1:8080` и ждёт ответа `/health`.
6. Запускает Gradio на `127.0.0.1:7860`, ждёт открытия порта и открывает браузер.
7. По `Ctrl+C` останавливает интерфейс и только тот `llama-server`, который запустил сам.

Проверить окружение без запуска процессов:

```powershell
python run_app.py --check
```

```bash
python3 run_app.py --check
```

Полный набор unit-тестов:

```powershell
python -m unittest discover -s Tests -v
```

### Если llama-server не в PATH

Windows:

```powershell
python run_app.py --llama-server "C:\tools\llama.cpp\llama-server.exe"

# или на постоянную сессию
$env:LLAMA_SERVER_BIN = "C:\tools\llama.cpp\llama-server.exe"
```

macOS:

```bash
python3 run_app.py --llama-server /opt/homebrew/bin/llama-server

# или на постоянную сессию
export LLAMA_SERVER_BIN=/opt/homebrew/bin/llama-server
```

Официальные сборки llama.cpp: <https://github.com/ggml-org/llama.cpp/releases>. Для Windows подойдёт вариант `llama-b<build>-bin-win-vulkan-x64.zip` (проверено на `b12532`), для macOS — `brew install llama.cpp` или архив `llama-b<build>-bin-macos-arm64.tar.gz`.

### Модель

По умолчанию используется `Models/T-lite-it-2.1-Q4_K_M.gguf`. Если файла нет, но в `Models/` лежит ровно один `.gguf`, лаунчер выберет его. Если моделей несколько, укажите нужную:

```powershell
python run_app.py --model "Models\my-model.gguf"
```

```bash
python3 run_app.py --model Models/my-model.gguf
```

Переменная окружения для пути к модели — `ONPZ_MODEL_PATH`. Веса и `*.gguf` не коммитятся.

### Параметры

| Параметр | По умолчанию | Назначение |
|---|---|---|
| `--llama-server PATH` | `LLAMA_SERVER_BIN`, затем `PATH` | путь к исполняемому файлу `llama-server` |
| `--model PATH` | `ONPZ_MODEL_PATH`, затем `Models/T-lite-it-2.1-Q4_K_M.gguf` | путь к GGUF-модели |
| `--llama-host`, `--llama-port` | `127.0.0.1`, `8080` | адрес локального llama-server |
| `--ui-host`, `--ui-port` | `127.0.0.1`, `7860` | адрес веб-интерфейса |
| `--ctx-size` | `8192` | размер контекста llama-server |
| `--ngl` | `99`; на Intel Mac — `0` | число слоёв модели на GPU |
| `--no-flash-attn` | выключено | не передавать `-fa on` |
| `--no-llama` | выключено | не запускать LLM, использовать уже запущенную |
| `--no-browser` | выключено | не открывать браузер |
| `--timeout` | `300` | сколько секунд ждать запуска LLM и интерфейса |
| `--check` | выключено | проверить окружение и выйти |

Примеры:

```powershell
python run_app.py --ui-port 7861          # если 7860 занят
python run_app.py --no-llama              # только интерфейс
python run_app.py --ngl 0                 # только CPU
python run_app.py --timeout 900           # медленный диск с моделью
```

## Веб-интерфейс (неделя 8)

Локальное однопользовательское приложение на **Gradio**: чат по регламенту, ответ LLM со ссылками `[n]`, карточки источников с метаданными и оценками релевантности, read-only вкладка метрик. Все сетевые вызовы идут только на `127.0.0.1` (llama-server); телеметрия Gradio отключена.

### Установка

```powershell
$env:PYTHONIOENCODING='utf-8'
python -m pip install -r requirements.txt
```

### Запуск

Самый простой способ — запустить весь стек одной командой (см. раздел выше):

```powershell
python run_app.py
```

```powershell
.\run_app.ps1
```

Ниже — ручной запуск в двух терминалах, если нужен контроль над каждым процессом.

1. Запустите llama-server (команда из раздела «Генерация ответа»):

   ```powershell
   llama-server -m "Models/T-lite-it-2.1-Q4_K_M.gguf" -c 8192 -ngl 99 -fa on --jinja --host 127.0.0.1 --port 8080
   ```

   На macOS команда та же, если `llama-server` установлен через Homebrew.

2. В корне репозитория запустите интерфейс:

   ```powershell
   python App/app.py
   ```

   Варианты запуска:

   ```powershell
   python App/app.py --no-browser          # не открывать браузер автоматически
   python App/app.py --port 7861           # если порт 7860 занят
   .\run_app.ps1 -NoLlama                  # только интерфейс, без запуска LLM
   .\run_app.ps1 -NoBrowser -Port 7861     # интерфейс на другом порту
   ```

   На macOS вместо `python` используйте `python3`.

3. Откройте `http://127.0.0.1:7860`.

### Возможности

* чат с историей в рамках сессии; каждый вопрос независим — история в LLM не передаётся;
* настройки `top_k` (1–10), `alpha` (0–1) и метода поиска: `hybrid`, `bm25`, `vector`;
* раскрытие аббревиатур в поисковом запросе;
* карточки источников: номер `[n]`, `chunk_id`, документ, раздел/подраздел, тип, score и текст; для `list_item`-групп — соседние пункты процедуры;
* источники строятся только по блокам, реально попавшим в контекст LLM (лимит 9000 символов); при обрезке показывается, сколько фрагментов не поместилось;
* статус LLM с кнопкой проверки соединения и командой запуска;
* примеры вопросов из `Eval/test_queries.json` без эталонных ответов и `gold_chunk_ids`;
* вкладка «Метрики»: Recall@5, MRR@5, refusal rate, top_k 3/5/10, аббревиатуры; данные только читаются из `Eval/`;
* кнопки «Очистить чат» и «Скачать историю» (JSON).

### Ограничения

* прототип локальный, однопользовательский, без авторизации и без внешних API;
* ответы ограничены учебным корпусом из 10 документов и 206 чанков;
* история чата не передаётся в LLM, RAG по диалогу не поддерживается;
* контекст ограничен 9000 символами и обрезается по границе блока;
* `Answer Accuracy` и `Hallucination Rate` остаются `null` до заполнения `Eval/manual_scores.json`; интерфейс показывает `—` и предупреждение, оценки не подставляет.

### Troubleshooting

| Проблема | Что сделать |
|---|---|
| Статус LLM красный, вопрос возвращает ошибку | Запустите стек через `python run_app.py` или поднимите llama-server вручную на `127.0.0.1:8080` и нажмите «Проверить соединение». Проверьте `LLM_BASE_URL`, если сервер на другом адресе. |
| `llama-server не найден` | Установите llama.cpp (`brew install llama.cpp` на macOS), добавьте его в `PATH` или укажите `--llama-server` / `LLAMA_SERVER_BIN`. |
| Порт 7860 занят | Запустите `python run_app.py --ui-port 7861` или `.\run_app.ps1 -Port 7861`. |
| Порт 8080 занят | Остановите старый llama-server или укажите `python run_app.py --llama-port 8081`; интерфейс получит адрес автоматически. |
| Модель не найдена | Положите `.gguf` в `Models/` или укажите `--model` / `ONPZ_MODEL_PATH`. |
| `ModuleNotFoundError: gradio` | Выполните `python -m pip install -r requirements.txt`. |
| Не загружаются индексы | Проверьте `Chunks/bm25_index.pkl`, `Chunks/faiss_index.bin`, `Chunks/faiss_chunk_ids.json` и локальный кэш модели `sergeyzh/rubert-tiny-turbo`. Интерфейс не пересобирает индексы автоматически. |
| Кириллица ломается в консоли | Задайте `$env:PYTHONIOENCODING='utf-8'` перед запуском; лаунчер и обёртки делают это сами. |
| macOS блокирует скачанный `llama-server` | Установите через `brew install llama.cpp` или снимите карантин: `xattr -d com.apple.quarantine /path/to/llama-server`. |
| Непонятно, чего не хватает | Выполните `python run_app.py --check`. |

## Оценка

Все 25 тестовых запросов размечены: у 22 нетрап-запросов заполнены `gold_chunk_ids` и `reference_answer`, у trap-запросов `gold_chunk_ids` пуст и задано `expected_behavior: "refusal"`.

Расчёт метрик:

```powershell
python Retrieval/alpha_tuning.py
python Eval/evaluate.py
```

`Eval/evaluate.py` считает:

* **Recall@5** и **MRR@5** для BM25 (`alpha = 1.0`), vector (`alpha = 0.0`) и hybrid (`alpha = 0.7`) на 22 размеченных запросах;
* **refusal_rate** — долю отказов модели;
* **trap_refusal_correctness** — долю корректных отказов на T1–T3;
* **Answer Accuracy** (среднее по шкале 0/1/2) и **Hallucination Rate** — только после ручной оценки;
* сравнение `top_k` 3/5/10 и варианта с раскрытием аббревиатур (если соответствующие прогоны существуют).

Текущие метрики поиска на 22 запросах:

| Метод | Recall@5 | MRR@5 |
|---|---|---|
| BM25 | 1.000 | 0.823 |
| Vector | 0.773 | 0.583 |
| Hybrid, `alpha = 0.7` | 1.000 | 0.871 |

Раскрытие аббревиатур сохраняет Recall@5, но немного снижает MRR@5:

| Вариант | Recall@5 | MRR@5 |
|---|---|---|
| Базовый запрос | 1.000 | 0.871 |
| С раскрытием аббревиатур | 1.000 | 0.848 |

Прогоны `top_k = 3`, `top_k = 10` и `--expand_abbrev` сохранены в `Eval/answers_*.jsonl`. Их метрики качества генерации появятся после заполнения соответствующих файлов ручной оценки.

Результаты сохраняются в `Eval/metrics.json` и `Eval/metrics.md`. Пока `Eval/manual_scores.json` содержит `null`, метрики генерации выводятся как `null` с пометкой «требуется ручная оценка»; скрипт не подставляет оценки автоматически.

Правила ручной оценки:

* `answer_accuracy`: `2` — верно и полно, `1` — частично верно или неполно, `0` — неверно либо необоснованный отказ; для trap-запросов корректный отказ оценивается в `2`;
* `hallucination`: `true`, если в ответе есть факт, число или единица, которых нет в переданном контексте;
* `comment` обязателен при `answer_accuracy != 2` или `hallucination = true`.


