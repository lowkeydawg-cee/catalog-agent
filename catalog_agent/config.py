"""Runtime settings, read from environment variables with CLI overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"})


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    input_dir: Path = Path("./input_images")
    processed_dir: Path = Path("./processed_images")
    failed_dir: Path = Path("./failed_images")
    db_path: Path = Path("./catalog.db")
    log_file: Path | None = Path("./logs/catalog_agent.log")
    log_level: str = "INFO"
    ollama_host: str = "http://localhost:11434"
    model: str = "llama3.2-vision"
    max_validation_retries: int = 3
    request_timeout: float = 300.0
    http_retries: int = 3
    max_image_side: int = 1120
    # How long a file's size must stay unchanged before we treat the copy as finished.
    settle_seconds: float = 1.0
    settle_timeout: float = 60.0
    image_extensions: frozenset[str] = field(default=IMAGE_EXTENSIONS)

    @classmethod
    def from_env(cls) -> "Settings":
        log_file = _env("CATALOG_LOG_FILE", "./logs/catalog_agent.log")
        return cls(
            input_dir=Path(_env("CATALOG_INPUT_DIR", "./input_images")),
            processed_dir=Path(_env("CATALOG_PROCESSED_DIR", "./processed_images")),
            failed_dir=Path(_env("CATALOG_FAILED_DIR", "./failed_images")),
            db_path=Path(_env("CATALOG_DB_PATH", "./catalog.db")),
            log_file=Path(log_file) if log_file.lower() != "none" else None,
            log_level=_env("CATALOG_LOG_LEVEL", "INFO").upper(),
            ollama_host=_env("OLLAMA_HOST", "http://localhost:11434"),
            model=_env("CATALOG_MODEL", "llama3.2-vision"),
            max_validation_retries=int(_env("CATALOG_MAX_VALIDATION_RETRIES", "3")),
            request_timeout=float(_env("CATALOG_REQUEST_TIMEOUT", "300")),
            http_retries=int(_env("CATALOG_HTTP_RETRIES", "3")),
            max_image_side=int(_env("CATALOG_MAX_IMAGE_SIDE", "1120")),
        )

    def with_overrides(self, **overrides) -> "Settings":
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})

    def ensure_dirs(self) -> None:
        for path in (self.input_dir, self.processed_dir, self.failed_dir, self.db_path.parent):
            path.mkdir(parents=True, exist_ok=True)
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
