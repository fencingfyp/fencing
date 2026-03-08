import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .tag import Tag


class TagStore:

    def __init__(self, db_path: str | Path):
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._migrate()

    # ------------------------------------------------------------------ lifecycle

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------ schema

    def _migrate(self):
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tags (
                id   INTEGER PRIMARY KEY AUTOINCREMENT,
                data JSON NOT NULL
            )
        """
        )
        self._conn.commit()

    # ------------------------------------------------------------------ CRUD

    def add(self, tag: Tag) -> Tag:
        data = asdict(tag)
        data.pop("id")
        cursor = self._conn.execute(
            "INSERT INTO tags (data) VALUES (?)",
            (json.dumps(data),),
        )
        self._conn.commit()
        tag.id = cursor.lastrowid
        return tag

    def remove(self, tag_id: int):
        self._conn.execute("DELETE FROM tags WHERE id = ?", (tag_id,))
        self._conn.commit()

    def get(self, tag_id: int) -> Tag | None:
        row = self._conn.execute(
            "SELECT id, data FROM tags WHERE id = ?", (tag_id,)
        ).fetchone()
        return self._row_to_tag(row) if row else None

    def all(self) -> list[Tag]:
        rows = self._conn.execute("SELECT id, data FROM tags").fetchall()
        return sorted(
            (self._row_to_tag(r) for r in rows),
            key=lambda t: t.frame_idx,
        )

    # ------------------------------------------------------------------ internal

    @staticmethod
    def _row_to_tag(row: sqlite3.Row) -> Tag:
        data = json.loads(row["data"])
        return Tag(id=row["id"], **data)
