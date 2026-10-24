import json
import shutil
import time

import httpx
import pytest

from catalog_agent.vision import OllamaVisionClient, VisionModelError
from catalog_agent.watcher import CatalogAgent

from conftest import make_image


def test_ollama_payload_and_retry_on_server_error():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        if len(seen) == 1:
            return httpx.Response(503, text="loading model")
        return httpx.Response(200, json={"message": {"role": "assistant", "content": '{"ok": true}'}})

    client = OllamaVisionClient(transport=httpx.MockTransport(handler), backoff_seconds=0)
    out = client.chat([{"role": "user", "content": "hi", "images": ["abc"]}], json_schema={"type": "object"})

    assert out == '{"ok": true}'
    assert len(seen) == 2
    assert seen[0]["model"] == "llama3.2-vision"
    assert seen[0]["format"] == {"type": "object"}
    assert seen[0]["stream"] is False
    assert seen[0]["messages"][0]["images"] == ["abc"]


def test_ollama_gives_up_after_retries():
    client = OllamaVisionClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(500)), http_retries=2, backoff_seconds=0
    )
    with pytest.raises(VisionModelError):
        client.chat([{"role": "user", "content": "hi"}])


def test_ollama_check_reports_missing_model():
    payload = {"models": [{"name": "llava:latest"}]}
    client = OllamaVisionClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload)))
    with pytest.raises(VisionModelError, match="ollama pull llama3.2-vision"):
        client.check()
    ok = OllamaVisionClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"models": [{"name": "llama3.2-vision:latest"}]}))
    )
    ok.check()


def test_ollama_check_reports_model_that_cannot_load():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3.2-vision:latest"}]})
        return httpx.Response(500, json={"error": "llama-server reported out-of-memory during startup"})

    client = OllamaVisionClient(transport=httpx.MockTransport(handler))
    with pytest.raises(VisionModelError, match="failed to load.*smaller model"):
        client.check()


@pytest.mark.parametrize("polling", [False, True])
def test_watcher_processes_new_and_existing_files(settings, store, make_processor, good_reply, tmp_path, polling):
    processor, _ = make_processor([good_reply])
    make_image(settings.input_dir / "already_there.png", color=(9, 9, 9))
    agent = CatalogAgent(processor, settings.input_dir, use_polling=polling)
    agent.start()
    try:
        # Simulate a camera/app writing to a temp file and renaming it in.
        staged = make_image(tmp_path / "staged.png", color=(200, 10, 10))
        shutil.move(str(staged), str(settings.input_dir / "new_drop.png"))
        make_image(settings.input_dir / "direct.jpg", color=(10, 200, 10), fmt="JPEG")

        deadline = time.time() + 15
        while time.time() < deadline and len(store.list_products()) < 3:
            time.sleep(0.1)
    finally:
        agent.stop()

    assert len(store.list_products()) == 3
    assert list(settings.input_dir.iterdir()) == []
    assert len(list(settings.processed_dir.iterdir())) == 3


def test_worker_survives_unexpected_exception(settings, store, make_processor, good_reply):
    processor, _ = make_processor([good_reply])
    original = processor.process
    calls = []

    def boom_once(path):
        calls.append(path)
        if len(calls) == 1:
            raise RuntimeError("unexpected")
        return original(path)

    processor.process = boom_once
    make_image(settings.input_dir / "a.png", color=(1, 1, 1))
    make_image(settings.input_dir / "b.png", color=(2, 2, 2))
    agent = CatalogAgent(processor, settings.input_dir)
    agent.start()
    agent.stop(drain=True)

    assert len(calls) == 2
    assert len(store.list_products()) == 1
