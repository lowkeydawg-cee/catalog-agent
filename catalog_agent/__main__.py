"""CLI entry point: `python -m catalog_agent [--once] [--check] ...`."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
from pathlib import Path

from .config import Settings
from .logging_setup import configure_logging
from .processor import ImageProcessor
from .storage import CatalogStore
from .vision import OllamaVisionClient, VisionModelError
from .watcher import CatalogAgent

log = logging.getLogger("catalog_agent")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="catalog-agent", description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--failed-dir", type=Path)
    parser.add_argument("--db", dest="db_path", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--ollama-host")
    parser.add_argument("--max-retries", dest="max_validation_retries", type=int)
    parser.add_argument("--log-level")
    parser.add_argument("--once", action="store_true", help="Process images already in the input folder, then exit.")
    parser.add_argument("--check", action="store_true", help="Check Ollama and the model are available, then exit.")
    parser.add_argument("--list", action="store_true", help="Print cataloged products and failures as JSON, then exit.")
    parser.add_argument("--polling", action="store_true", help="Use a polling observer (network drives, Docker mounts, WSL).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env().with_overrides(
        input_dir=args.input_dir,
        processed_dir=args.processed_dir,
        failed_dir=args.failed_dir,
        db_path=args.db_path,
        model=args.model,
        ollama_host=args.ollama_host,
        max_validation_retries=args.max_validation_retries,
        log_level=args.log_level.upper() if args.log_level else None,
    )
    settings.ensure_dirs()
    configure_logging(settings.log_level, settings.log_file)

    store = CatalogStore(settings.db_path)
    if args.list:
        print(json.dumps({"products": store.list_products(), "failures": store.list_failures()}, indent=2))
        return 0

    client = OllamaVisionClient(
        host=settings.ollama_host,
        model=settings.model,
        timeout=settings.request_timeout,
        http_retries=settings.http_retries,
    )
    try:
        client.check()
        log.info("Ollama OK, using model %s", settings.model)
    except VisionModelError as exc:
        log.error("%s", exc)
        return 2
    if args.check:
        return 0

    agent = CatalogAgent(ImageProcessor(settings, client, store), settings.input_dir, use_polling=args.polling)

    if args.once:
        agent.start()
        agent.stop(drain=True)
        failed = sum(r.status in ("failed", "deferred") for r in agent.results)
        log.info("Done: %d processed, %d failed", sum(r.status == "processed" for r in agent.results), failed)
        return 1 if failed else 0

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    agent.start()
    log.info("Drop images into %s. Press Ctrl+C to stop.", settings.input_dir.resolve())
    while not stop.wait(0.5):  # short timeout keeps Ctrl+C responsive on Windows
        pass
    log.info("Shutting down...")
    agent.stop()
    client.close()
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
