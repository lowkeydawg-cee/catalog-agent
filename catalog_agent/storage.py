"""SQLite persistence for listings and failures."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .schema import ProductListing

SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slug TEXT NOT NULL UNIQUE,
    product_title TEXT NOT NULL,
    store_description TEXT NOT NULL,
    instagram_caption TEXT NOT NULL,
    tiktok_hook TEXT NOT NULL,          -- JSON array of 3 strings
    tags TEXT NOT NULL,                 -- JSON array of hashtags
    visual_analysis TEXT NOT NULL,      -- JSON object
    listing_json TEXT NOT NULL,         -- full validated payload
    original_filename TEXT NOT NULL,
    image_path TEXT NOT NULL,
    image_sha256 TEXT NOT NULL UNIQUE,
    model TEXT NOT NULL,
    attempts INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_filename TEXT NOT NULL,
    moved_to TEXT,
    reason TEXT NOT NULL,
    detail TEXT,
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CatalogStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Explicit transaction so callers can roll back if the file move fails."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def insert_product(
        self,
        conn: sqlite3.Connection,
        *,
        slug: str,
        listing: ProductListing,
        original_filename: str,
        image_path: Path,
        image_sha256: str,
        model: str,
        attempts: int,
    ) -> int:
        cursor = conn.execute(
            """INSERT INTO products (slug, product_title, store_description, instagram_caption, tiktok_hook,
                   tags, visual_analysis, listing_json, original_filename, image_path, image_sha256, model,
                   attempts, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                slug,
                listing.product_title,
                listing.store_description,
                listing.instagram_caption,
                json.dumps(listing.tiktok_hook),
                json.dumps(listing.tags),
                listing.visual_analysis.model_dump_json(),
                listing.model_dump_json(),
                original_filename,
                str(image_path),
                image_sha256,
                model,
                attempts,
                _now(),
            ),
        )
        return int(cursor.lastrowid)

    def record_failure(self, original_filename: str, reason: str, detail: str = "", moved_to: Path | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO failures (original_filename, moved_to, reason, detail, created_at) VALUES (?, ?, ?, ?, ?)",
                (original_filename, str(moved_to) if moved_to else None, reason, detail[:4000], _now()),
            )

    def has_hash(self, image_sha256: str) -> bool:
        with self._lock:
            row = self._conn.execute("SELECT 1 FROM products WHERE image_sha256 = ?", (image_sha256,)).fetchone()
        return row is not None

    def slug_exists(self, slug: str) -> bool:
        with self._lock:
            row = self._conn.execute("SELECT 1 FROM products WHERE slug = ?", (slug,)).fetchone()
        return row is not None

    def get_product(self, slug: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM products WHERE slug = ?", (slug,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        for key in ("tiktok_hook", "tags", "visual_analysis", "listing_json"):
            data[key] = json.loads(data[key])
        return data

    def list_products(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT slug, product_title, image_path, created_at FROM products ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    def list_failures(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM failures ORDER BY id").fetchall()
        return [dict(r) for r in rows]
