"""Расширение запроса раскрытиями аббревиатур и синонимов (неделя 6)."""

import re

import pymorphy3

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

_WORD_RE = re.compile(r"[а-яёa-z0-9]+")
_morph = pymorphy3.MorphAnalyzer()


def _lemmas(text: str) -> list[str]:
    return [_morph.parse(word)[0].normal_form for word in _WORD_RE.findall(text.lower())]


def _contains_phrase(text: str, phrase: str) -> bool:
    text_lemmas = _lemmas(text)
    phrase_lemmas = _lemmas(phrase)
    if not phrase_lemmas:
        return False
    window = len(phrase_lemmas)
    return any(
        text_lemmas[index : index + window] == phrase_lemmas
        for index in range(len(text_lemmas) - window + 1)
    )


def expand_query(query: str) -> str:
    """Дописать к запросу раскрытия найденных аббревиатур и синонимов."""
    additions: list[str] = []
    seen: set[str] = set()

    for term, expansion in ABBREVIATIONS.items():
        if not _contains_phrase(query, term):
            continue
        normalized = expansion.lower()
        if normalized in seen or _contains_phrase(query, expansion):
            continue
        seen.add(normalized)
        additions.append(expansion)

    if not additions:
        return query
    return f"{query} {' '.join(additions)}"
