"""Добавляет каталоги Generation и Retrieval в sys.path для тестов."""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

for folder in ("Generation", "Retrieval"):
    path = str(BASE_DIR / folder)
    if path not in sys.path:
        sys.path.insert(0, path)
