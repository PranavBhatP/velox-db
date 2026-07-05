"""Sidecar JSON store mapping vector IDs to source text."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class MetadataStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._docs: dict[str, dict[str, Any]] = {}

    def load(self) -> None:
        if not self.path.exists():
            self._docs = {}
            return
        with self.path.open(encoding="utf-8") as f:
            raw = json.load(f)
        self._docs = {str(k): v for k, v in raw.items()}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self._docs, f, indent=2)

    def add(self, doc_id: int, text: str) -> None:
        self._docs[str(doc_id)] = {
            "text": text,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def get(self, doc_id: int) -> dict[str, Any] | None:
        return self._docs.get(str(doc_id))

    def list_all(self) -> list[dict[str, Any]]:
        items = []
        for key, value in self._docs.items():
            items.append({"id": int(key), **value})
        items.sort(key=lambda x: x["id"])
        return items

    def count(self) -> int:
        return len(self._docs)
