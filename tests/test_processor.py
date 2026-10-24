import json

from catalog_agent.vision import VisionModelError

from conftest import make_image, valid_listing


def test_success_saves_row_and_moves_with_slug(settings, store, make_processor, good_reply):
    src = make_image(settings.input_dir / "IMG_0042.JPEG", fmt="JPEG")
    processor, client = make_processor([good_reply])

    result = processor.process(src)

    assert result.status == "processed"
    assert result.slug == "hexagon-desk-organizer-modular-3d-printed-pen-holder"
    assert result.destination == settings.processed_dir / f"{result.slug}.jpg"
    assert result.destination.exists() and not src.exists()
    row = store.get_product(result.slug)
    assert row["original_filename"] == "IMG_0042.JPEG"
    assert row["tiktok_hook"][0] == "Your desk is begging for this."
    assert "#customdesign" in row["tags"]
    assert row["attempts"] == 1
    # The image was sent to the model.
    assert client.calls[0][1]["images"]


def test_validation_retry_feeds_errors_back(settings, store, make_processor, good_reply):
    bad = json.dumps(valid_listing(tiktok_hook=["only one"]))
    src = make_image(settings.input_dir / "part.png")
    processor, client = make_processor(["not json at all", bad, good_reply])

    result = processor.process(src)

    assert result.status == "processed"
    assert result.attempts == 3
    last_call = client.calls[-1]
    assert last_call[-1]["role"] == "user"
    assert "tiktok_hook" in last_call[-1]["content"]
    assert store.get_product(result.slug)["attempts"] == 3


def test_validation_exhausted_moves_to_failed(settings, store, make_processor):
    bad = json.dumps(valid_listing(tags=[]))
    src = make_image(settings.input_dir / "vase.png")
    processor, client = make_processor([bad])

    result = processor.process(src)

    assert result.status == "failed"
    assert len(client.calls) == 1 + settings.max_validation_retries
    assert not src.exists()
    assert result.destination.parent == settings.failed_dir and result.destination.exists()
    assert "Last model output" in result.destination.with_name(result.destination.name + ".error.txt").read_text()
    assert store.list_products() == []
    assert store.list_failures()[0]["reason"] == "validation"


def test_unreadable_file_never_reaches_model(settings, store, make_processor, good_reply):
    src = settings.input_dir / "corrupt.jpg"
    src.write_bytes(b"\xff\xd8\xff\xe0 this is not really a jpeg")
    processor, client = make_processor([good_reply])

    result = processor.process(src)

    assert result.status == "failed"
    assert client.calls == []
    assert store.list_failures()[0]["reason"] == "unreadable"
    assert any(settings.failed_dir.glob("*_corrupt.jpg"))


def test_non_image_is_ignored(settings, make_processor, good_reply):
    src = settings.input_dir / "notes.txt"
    src.write_text("hello")
    processor, client = make_processor([good_reply])

    assert processor.process(src).status == "skipped"
    assert src.exists() and client.calls == []


def test_duplicate_image_not_reprocessed(settings, store, make_processor, good_reply):
    processor, client = make_processor([good_reply])
    first = processor.process(make_image(settings.input_dir / "a.png"))
    second = processor.process(make_image(settings.input_dir / "b.png"))  # identical pixels/bytes

    assert first.status == "processed"
    assert second.status == "duplicate"
    assert len(client.calls) == 1


def test_same_title_gets_unique_slug(settings, store, make_processor, good_reply):
    processor, _ = make_processor([good_reply])
    a = processor.process(make_image(settings.input_dir / "a.png", color=(1, 2, 3)))
    b = processor.process(make_image(settings.input_dir / "b.png", color=(4, 5, 6)))

    assert b.slug == f"{a.slug}-2"
    assert b.destination.exists()


def test_model_outage_leaves_file_in_place(settings, store, make_processor):
    src = make_image(settings.input_dir / "keep.png")
    processor, _ = make_processor([VisionModelError("connection refused")])

    result = processor.process(src)

    assert result.status == "deferred"
    assert src.exists()
    assert store.list_failures()[0]["reason"] == "model_unavailable"


def test_move_failure_rolls_back_db_row(settings, store, make_processor, good_reply, monkeypatch):
    src = make_image(settings.input_dir / "x.png")
    processor, _ = make_processor([good_reply])

    import catalog_agent.processor as proc

    real_move = proc.shutil.move
    calls = {"n": 0}

    def flaky_move(a, b):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError("disk says no")
        return real_move(a, b)

    monkeypatch.setattr(proc.shutil, "move", flaky_move)
    result = processor.process(src)

    assert result.status == "failed"
    assert store.list_products() == []
    assert store.list_failures()[0]["reason"] == "save_error"
