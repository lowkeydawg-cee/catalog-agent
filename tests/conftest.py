from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from catalog_agent.config import Settings
from catalog_agent.processor import ImageProcessor
from catalog_agent.storage import CatalogStore


def valid_listing(**overrides) -> dict:
    data = {
        "visual_analysis": {
            "shape": "Hexagonal open-top container with tapered walls",
            "material_texture": "Matte PLA with fine visible layer lines, sage green",
            "visible_utility": "Desk organizer for pens and small tools",
            "target_audience": "Home-office workers and makers",
        },
        "product_title": "Hexagon Desk Organizer - Modular 3D Printed Pen Holder",
        "store_description": (
            "Keep your workspace tidy with this modular hexagon organizer.\n"
            "- Printed in durable matte PLA\n"
            "- Snaps together with other hex modules\n"
            "- Fits pens, scissors and hex keys"
        ),
        "instagram_caption": "Desk chaos? Not anymore. Our hex organizer locks together like honeycomb. Tap the link in bio to build yours!",
        "tiktok_hook": [
            "Your desk is begging for this.",
            "I 3D printed a honeycomb for my pens.",
            "Watch this organizer snap together.",
        ],
        "tags": ["#3dprinting", "#customdesign", "desk setup", "#Organizer", "#maker", "#3dprinting"],
    }
    data.update(overrides)
    return data


class FakeClient:
    """Stands in for Ollama: returns queued replies and records every call."""

    model = "fake-vision"

    def __init__(self, replies: list[str | Exception]) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def chat(self, messages, json_schema=None):
        self.calls.append([dict(m) for m in messages])
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply


def make_image(path: Path, color=(120, 180, 140), size=(64, 48), fmt=None) -> Path:
    Image.new("RGB", size, color).save(path, format=fmt)
    return path


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings(
        input_dir=tmp_path / "input_images",
        processed_dir=tmp_path / "processed_images",
        failed_dir=tmp_path / "failed_images",
        db_path=tmp_path / "catalog.db",
        log_file=None,
        max_validation_retries=2,
        settle_seconds=0.05,
        settle_timeout=5,
    )
    s.ensure_dirs()
    return s


@pytest.fixture
def store(settings: Settings):
    store = CatalogStore(settings.db_path)
    yield store
    store.close()


@pytest.fixture
def make_processor(settings: Settings, store: CatalogStore):
    def _make(replies):
        client = FakeClient(replies)
        return ImageProcessor(settings, client, store), client

    return _make


@pytest.fixture
def good_reply() -> str:
    return json.dumps(valid_listing())
