"""
  V1 "by_section"  - чанк = раздел/подраздел документа целиком (или кусок
                      раздела, если он длиннее MAX_TOKENS_V1 токенов).
                      Таблица -> один чанк на всю таблицу.
                      Это базовая стратегия чанкинга по тексту ТЗ.

  V2 "by_fact"      - более мелкая нарезка: та же иерархия по заголовкам,
                      но внутри раздела:
                        * список из >= 3 пунктов -> каждый пункт - отдельный
                          чанк (с вводной фразой списка в метаданных и тексте
                          для сохранения контекста);
                        * таблица -> ОДНА СТРОКА ДАННЫХ = один чанк
                          (самостоятельная мини-таблица: заголовок + строка);
                        * обычный текст -> режется по абзацам с меньшим
                          лимитом токенов (MAX_TOKENS_V2), чтобы чанк был
                          ближе к размеру одного факта.
                      Такая нарезка даёт число чанков, близкое к диапазону
                      150-250, заданному в плане работы ТЗ (неделя 3).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

SRC_DIR = Path(__file__).parent.parent / "Processed data"
OUT_DIR = Path(__file__).parent.parent / "Chunks"
OUT_V1 = OUT_DIR / "chunks_v1_by_section.jsonl"
OUT_V2 = OUT_DIR / "chunks_v2_by_fact.jsonl"
OUT_REPORT = OUT_DIR / "chunking_report.txt"

MAX_TOKENS_V1 = 800
MAX_TOKENS_V2 = 220
LIST_MIN_ITEMS = 3

TOKENS_PER_WORD = 1.3

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:\-|]+\|?\s*$")
LIST_ITEM_RE = re.compile(r"^\s*(?:[*\-]|\d+\.)\s+.+$")


def estimate_tokens(text: str) -> int:
    """Грубая оценка числа токенов без внешнего токенизатора."""
    words = re.findall(r"\S+", text)
    return max(1, round(len(words) * TOKENS_PER_WORD))



# Шаг 1. Разбор markdown-файла на иерархию "раздел (##) -> подраздел (###)"


@dataclass
class Unit:
    section: str
    subsection: str | None
    lines: list[str] = field(default_factory=list)


def parse_document(text: str) -> tuple[str, list[Unit]]:
    lines = text.splitlines()
    title = ""
    units: list[Unit] = []

    current_section = ""
    current_subsection: str | None = None
    buffer: list[str] = []

    def flush():
        if buffer and any(l.strip() for l in buffer):
            units.append(Unit(current_section, current_subsection, buffer.copy()))
        buffer.clear()

    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            heading_text = m.group(2).strip()
            if level == 1:
                title = heading_text
                continue
            if level == 2:
                flush()
                current_section = heading_text
                current_subsection = None
                continue
            if level == 3:
                flush()
                current_subsection = heading_text
                continue
            buffer.append(line)
            continue
        buffer.append(line)
    flush()
    return title, units



# Шаг 2. Разбиение содержимого Unit на блоки: "table" и "text"


@dataclass
class Block:
    kind: str
    lines: list[str]


def split_into_blocks(lines: list[str]) -> list[Block]:
    blocks: list[Block] = []
    buf: list[str] = []
    mode = None

    def flush():
        nonlocal buf, mode
        if buf and mode is not None:
            blocks.append(Block(mode, buf.copy()))
        buf = []

    for line in lines:
        is_table_line = bool(TABLE_ROW_RE.match(line)) or bool(TABLE_SEP_RE.match(line) and "|" in line)
        if is_table_line:
            if mode != "table":
                flush()
                mode = "table"
            buf.append(line)
        else:
            if mode != "text":
                flush()
                mode = "text"
            buf.append(line)
    flush()
    return blocks



# Утилиты для таблиц


def clean_cell(cell: str) -> str:
    cell = cell.strip()
    cell = re.sub(r"\$\\text\{([^}]*)\}\$?", r"\1", cell)
    cell = re.sub(r"\\text\{([^}]*)\}", r"\1", cell)
    cell = cell.replace("\\,", " ").replace("^\\circ", "°")
    cell = cell.replace("$", "").replace("\\", "")
    cell = re.sub(r"\s+", " ", cell).strip(" *:")
    return cell


def table_columns(table_lines: list[str]) -> list[str]:
    rows = [l for l in table_lines if l.strip()]
    if not rows:
        return []
    header = rows[0]
    cells = [c for c in header.strip().strip("|").split("|")]
    return [clean_cell(c) for c in cells if clean_cell(c)]


def split_table_rows(table_lines: list[str]) -> list[dict]:
    """V2: превращает таблицу в список самостоятельных мини-таблиц (заголовок + 1 строка)."""
    rows = [l for l in table_lines if l.strip()]
    if not rows:
        return []
    header = rows[0]
    cols = table_columns(table_lines)
    if len(rows) > 1 and TABLE_SEP_RE.match(rows[1]):
        sep = rows[1]
        data_rows = rows[2:]
    else:
        ncols = header.strip().strip("|").count("|") + 1
        sep = "|" + "|".join([" :--- "] * ncols) + "|"
        data_rows = rows[1:]

    out = []
    for idx, row in enumerate(data_rows, start=1):
        row_text = "\n".join([header, sep, row])
        out.append({"text": row_text, "row_index": idx, "columns": cols})
    return out



# Утилиты для текста: абзацы, упаковка, разбиение списков


def split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text.strip("\n"))
    return [p for p in (p.strip() for p in parts) if p]


def pack_units(units: list[str], max_tokens: int) -> list[str]:
    packed: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for u in units:
        t = estimate_tokens(u)
        if current and current_tokens + t > max_tokens:
            packed.append("\n\n".join(current))
            current = [u]
            current_tokens = t
        else:
            current.append(u)
            current_tokens += t
    if current:
        packed.append("\n\n".join(current))
    return packed


def try_split_list(paragraph: str, min_items: int = LIST_MIN_ITEMS):
    """Если абзац - это (опционально) вводная фраза + список из >= min_items
    пунктов, возвращает (intro, [items]). Иначе - None."""
    lines = paragraph.splitlines()
    first_idx = None
    for i, l in enumerate(lines):
        if LIST_ITEM_RE.match(l):
            first_idx = i
            break
    if first_idx is None:
        return None
    tail = lines[first_idx:]
    if not all(LIST_ITEM_RE.match(l) for l in tail):
        return None
    if len(tail) < min_items:
        return None
    intro = "\n".join(lines[:first_idx]).strip()
    items = [l.strip() for l in tail]
    return intro, items



# V1: чанк = раздел/подраздел (или его часть при превышении лимита)


def chunk_text_block_v1(text: str, max_tokens: int) -> list[str]:
    text = text.strip("\n")
    if not text.strip():
        return []
    if estimate_tokens(text) <= max_tokens:
        return [text]

    paragraphs = split_paragraphs(text)
    if len(paragraphs) <= 1:
        lines = [l for l in text.splitlines() if l.strip()]
        chunks = pack_units(lines, max_tokens)
    else:
        chunks = pack_units(paragraphs, max_tokens)

    final: list[str] = []
    for c in chunks:
        if estimate_tokens(c) <= max_tokens:
            final.append(c)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", c)
        final.extend(pack_units(sentences, max_tokens))
    return final


def build_chunks_for_file_v1(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    title, units = parse_document(text)
    doc_name = path.name
    chunks: list[dict] = []
    seq = 0

    for unit in units:
        section, subsection = unit.section, unit.subsection
        for block in split_into_blocks(unit.lines):
            block_text = "\n".join(block.lines).strip("\n")
            if not block_text.strip():
                continue
            if block.kind == "table":
                seq += 1
                cols = table_columns(block.lines)
                chunks.append({
                    "chunk_id": f"{path.stem}__{seq:03d}",
                    "document": doc_name, "document_title": title,
                    "section": section, "subsection": subsection,
                    "type": "table", "table_columns": cols,
                    "text": block_text,
                    "token_estimate": estimate_tokens(block_text),
                    "char_count": len(block_text),
                })
            else:
                pieces = chunk_text_block_v1(block_text, max_tokens=MAX_TOKENS_V1)
                multi = len(pieces) > 1
                for i, piece in enumerate(pieces, start=1):
                    seq += 1
                    chunks.append({
                        "chunk_id": f"{path.stem}__{seq:03d}",
                        "document": doc_name, "document_title": title,
                        "section": section, "subsection": subsection,
                        "type": "text",
                        "part": f"{i}/{len(pieces)}" if multi else None,
                        "text": piece,
                        "token_estimate": estimate_tokens(piece),
                        "char_count": len(piece),
                    })
    return chunks



# V2: чанк = один факт (пункт списка / строка таблицы / мелкий абзац)


def chunk_text_block_v2(text: str, max_tokens: int) -> list[dict]:
    text = text.strip("\n")
    if not text.strip():
        return []
    paragraphs = split_paragraphs(text) or [text]

    results: list[dict] = []
    pending: list[str] = []

    def flush_pending():
        if not pending:
            return
        for p in pack_units(pending, max_tokens):
            if estimate_tokens(p) <= max_tokens:
                results.append({"text": p, "is_list_item": False, "list_intro": None})
            else:
                sentences = re.split(r"(?<=[.!?])\s+", p)
                for s in pack_units(sentences, max_tokens):
                    results.append({"text": s, "is_list_item": False, "list_intro": None})
        pending.clear()

    for para in paragraphs:
        parsed = try_split_list(para)
        if parsed is None:
            pending.append(para)
            continue
        intro, items = parsed
        flush_pending()
        for item in items:
            item_text = f"{intro}\n{item}" if intro else item
            results.append({"text": item_text, "is_list_item": True, "list_intro": intro or None})
    flush_pending()
    return results


def build_chunks_for_file_v2(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    title, units = parse_document(text)
    doc_name = path.name
    chunks: list[dict] = []
    seq = 0

    for unit in units:
        section, subsection = unit.section, unit.subsection
        for block in split_into_blocks(unit.lines):
            block_text = "\n".join(block.lines).strip("\n")
            if not block_text.strip():
                continue
            if block.kind == "table":
                rows = split_table_rows(block.lines)
                for r in rows:
                    seq += 1
                    chunks.append({
                        "chunk_id": f"{path.stem}__{seq:03d}",
                        "document": doc_name, "document_title": title,
                        "section": section, "subsection": subsection,
                        "type": "table_row",
                        "table_row_index": r["row_index"],
                        "table_columns": r["columns"],
                        "text": r["text"],
                        "token_estimate": estimate_tokens(r["text"]),
                        "char_count": len(r["text"]),
                    })
            else:
                pieces = chunk_text_block_v2(block_text, max_tokens=MAX_TOKENS_V2)
                multi = len(pieces) > 1
                for i, piece in enumerate(pieces, start=1):
                    seq += 1
                    chunks.append({
                        "chunk_id": f"{path.stem}__{seq:03d}",
                        "document": doc_name, "document_title": title,
                        "section": section, "subsection": subsection,
                        "type": "list_item" if piece["is_list_item"] else "text",
                        "part": f"{i}/{len(pieces)}" if multi else None,
                        "list_intro": piece["list_intro"],
                        "text": piece["text"],
                        "token_estimate": estimate_tokens(piece["text"]),
                        "char_count": len(piece["text"]),
                    })
    return chunks



# Главный цикл + отчёт


def write_jsonl(path: Path, chunks: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def summarize(chunks: list[dict], md_files: list[Path]) -> dict:
    per_doc: dict[str, int] = {p.name: 0 for p in md_files}
    by_type: dict[str, int] = {}
    for c in chunks:
        per_doc[c["document"]] += 1
        by_type[c["type"]] = by_type.get(c["type"], 0) + 1
    tokens = [c["token_estimate"] for c in chunks]
    return {
        "total": len(chunks),
        "per_doc": per_doc,
        "by_type": by_type,
        "min": min(tokens), "max": max(tokens),
        "avg": sum(tokens) / len(tokens),
    }


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    md_files = sorted(SRC_DIR.glob("*.md"))
    if not md_files:
        raise SystemExit(f"Не найдено .md файлов в {SRC_DIR}")

    v1_chunks: list[dict] = []
    v2_chunks: list[dict] = []
    for path in md_files:
        v1_chunks.extend(build_chunks_for_file_v1(path))
        v2_chunks.extend(build_chunks_for_file_v2(path))

    write_jsonl(OUT_V1, v1_chunks)
    write_jsonl(OUT_V2, v2_chunks)

    s1 = summarize(v1_chunks, md_files)
    s2 = summarize(v2_chunks, md_files)

    lines = [
        "ОТЧЁТ ПО ЧАНКИНГУ (V1 vs V2)",

        f"Источник: {SRC_DIR}",
        f"Документов обработано: {len(md_files)}",
        "",
        "V1: chunks_v1_by_section.jsonl (чанк = раздел/подраздел/таблица)",
        f"Лимит токенов: {MAX_TOKENS_V1}",
        f"Всего чанков: {s1['total']}",
        f"По типам: {s1['by_type']}",
        f"Токены: min={s1['min']}, max={s1['max']}, avg={s1['avg']:.1f}",
        "",
        "V2: chunks_v2_by_fact.jsonl (чанк = пункт списка / строка таблицы / мелкий абзац)",
        f"Лимит токенов на текстовый абзац: {MAX_TOKENS_V2}; порог списка: >= {LIST_MIN_ITEMS} пунктов",
        f"Всего чанков: {s2['total']}",
        f"По типам: {s2['by_type']}",
        f"Токены: min={s2['min']}, max={s2['max']}, avg={s2['avg']:.1f}",
        f"Целевой диапазон по плану ТЗ (неделя 3): 150-250 чанков -> "
        f"{'ДОСТИГНУТ' if 150 <= s2['total'] <= 250 else 'НЕ достигнут'}",
        "",
        "Чанков по документам (V1 / V2):",
    ]
    for p in md_files:
        name = p.name
        lines.append(f"  {name}: {s1['per_doc'][name]} / {s2['per_doc'][name]}")

    report = "\n".join(lines)
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
