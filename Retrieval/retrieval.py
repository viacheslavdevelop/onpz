import argparse
import json
from pathlib import Path

import bm25_search
import vector_search
from chunks_io import load_chunks

EVAL_DIR = Path(__file__).parent.parent / "Eval"
TEST_QUERIES_PATH = EVAL_DIR / "test_queries.json"
RESULTS_PATH = EVAL_DIR / "preliminary_results.md"


def build_chunk_lookup() -> dict:
    return {c["chunk_id"]: c for c in load_chunks()}


def chunk_summary(chunk: dict) -> str:
    location = chunk.get("subsection") or chunk.get("section") or chunk.get("document") or ""
    snippet = chunk.get("text", "").replace("\n", " ").strip()
    if len(snippet) > 150:
        snippet = snippet[:150] + "..."
    return f"{location} | {snippet}"


def run_bm25(query: str, top_k: int) -> list[tuple[str, float]]:
    index, chunk_ids = bm25_search.load_index()
    return bm25_search.search(index, chunk_ids, query, top_k)


def run_vector(query: str, top_k: int) -> list[tuple[str, float]]:
    model = vector_search.load_model()
    index, chunk_ids = vector_search.load_index()
    return vector_search.search(model, index, chunk_ids, query, top_k)


def print_results(method: str, results: list[tuple[str, float]], chunk_lookup: dict) -> None:
    print(f"\n[{method}]")
    for rank, (chunk_id, score) in enumerate(results, start=1):
        chunk = chunk_lookup.get(chunk_id, {})
        print(f"{rank}. {chunk_id} (score={score:.3f}) {chunk_summary(chunk)}")


def run_cli_query(query: str, method: str, top_k: int) -> None:
    chunk_lookup = build_chunk_lookup()
    if method in ("bm25", "both"):
        print_results("BM25", run_bm25(query, top_k), chunk_lookup)
    if method in ("vector", "both"):
        print_results("Vector", run_vector(query, top_k), chunk_lookup)


def load_test_queries() -> list[dict]:
    return json.loads(TEST_QUERIES_PATH.read_text(encoding="utf-8"))


def format_markdown_results(title: str, results: list[tuple[str, float]], chunk_lookup: dict) -> str:
    lines = [f"**{title}:**"]
    for rank, (chunk_id, score) in enumerate(results, start=1):
        chunk = chunk_lookup.get(chunk_id, {})
        lines.append(f"{rank}. `{chunk_id}` (score={score:.3f}) — {chunk_summary(chunk)}")
    return "\n".join(lines)


def run_all_test_queries(top_k: int = 5) -> None:
    chunk_lookup = build_chunk_lookup()
    queries = load_test_queries()

    bm25_index, bm25_chunk_ids = bm25_search.load_index()
    vector_model = vector_search.load_model()
    vector_index, vector_chunk_ids = vector_search.load_index()

    sections = ["# Предварительные результаты поиска (неделя 4)", ""]
    for q in queries:
        query_id = q["query_id"]
        query_text = q["query_text"]
        category = q["category"]
        note = q.get("note", "")

        bm25_results = bm25_search.search(bm25_index, bm25_chunk_ids, query_text, top_k)
        vector_results = vector_search.search(vector_model, vector_index, vector_chunk_ids, query_text, top_k)

        sections.append(f"## {query_id} [{category}] {query_text}")
        if note:
            sections.append(f"_Примечание: {note}_")
        sections.append("")
        sections.append(format_markdown_results("BM25", bm25_results, chunk_lookup))
        sections.append("")
        sections.append(format_markdown_results("Vector", vector_results, chunk_lookup))
        sections.append("")

    RESULTS_PATH.write_text("\n".join(sections), encoding="utf-8")
    print(f"Сохранено: {RESULTS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query")
    parser.add_argument("--method", choices=["bm25", "vector", "both"], default="both")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--run_all", action="store_true")
    args = parser.parse_args()

    if args.run_all:
        run_all_test_queries(args.top_k)
    elif args.query:
        run_cli_query(args.query, args.method, args.top_k)
    else:
        parser.error("укажите --query или --run_all")


if __name__ == "__main__":
    main()
