import json
from pathlib import Path

CHUNKS_PATH = Path(__file__).parent.parent / "Chunks" / "chunks_v2_by_fact.jsonl"


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    chunks = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks
