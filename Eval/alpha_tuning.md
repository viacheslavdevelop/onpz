# Подбор alpha для гибридного поиска (неделя 5)

Размечено запросов: 13 из 25 (категории factual и synonym_abbreviation).

alpha=0.0 эквивалентен чистому vector-поиску, alpha=1.0 — чистому BM25.

| query_id | category | alpha=0.0 | alpha=0.3 | alpha=0.5 | alpha=0.7 | alpha=1.0 |
|---|---|---|---|---|---|---|
| F1 | factual | 0 | 1 | 1 | 1 | 1 |
| F2 | factual | 0 | 0 | 0 | 1 | 1 |
| F3 | factual | 1 | 1 | 1 | 1 | 1 |
| F4 | factual | 1 | 1 | 1 | 1 | 1 |
| F5 | factual | 1 | 1 | 1 | 1 | 1 |
| F6 | factual | 0 | 0 | 0 | 1 | 1 |
| F7 | factual | 1 | 1 | 1 | 1 | 1 |
| F8 | factual | 1 | 1 | 1 | 1 | 1 |
| S1 | synonym_abbreviation | 1 | 1 | 1 | 1 | 1 |
| S2 | synonym_abbreviation | 1 | 1 | 1 | 1 | 1 |
| S3 | synonym_abbreviation | 1 | 1 | 1 | 1 | 1 |
| S4 | synonym_abbreviation | 1 | 1 | 1 | 1 | 1 |
| S5 | synonym_abbreviation | 1 | 1 | 1 | 1 | 1 |

## Recall@5 по alpha

| alpha | Recall@5 |
|---|---|
| 0.0 | 10/13 = 0.769 |
| 0.3 | 11/13 = 0.846 |
| 0.5 | 11/13 = 0.846 |
| 0.7 | 13/13 = 1.000 |
| 1.0 | 13/13 = 1.000 |

**Лучший alpha по Recall@5 на размеченных запросах: 0.7** (возможна ничья — см. таблицу).