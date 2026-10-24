"""Filesystem watcher that feeds new images into a single worker thread."""

from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver

from .processor import ImageProcessor, ProcessResult

log = logging.getLogger(__name__)


class _EnqueueHandler(FileSystemEventHandler):
    def __init__(self, agent: "CatalogAgent") -> None:
        self.agent = agent

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self.agent.enqueue(Path(event.src_path))

    def on_moved(self, event: FileSystemEvent) -> None:
        # Many apps write to a temp name then rename into place.
        if not event.is_directory:
            self.agent.enqueue(Path(event.dest_path))


class CatalogAgent:
    """Watches `input_dir` and processes images one at a time (the vision model is the bottleneck)."""

    def __init__(self, processor: ImageProcessor, input_dir: Path, use_polling: bool = False) -> None:
        self.processor = processor
        self.input_dir = Path(input_dir)
        self.use_polling = use_polling
        self.results: list[ProcessResult] = []
        self._queue: "queue.Queue[Path | None]" = queue.Queue()
        self._pending: set[Path] = set()
        self._pending_lock = threading.Lock()
        self._stop = threading.Event()
        self._observer = None
        self._worker: threading.Thread | None = None

    def enqueue(self, path: Path) -> None:
        path = path.resolve()
        if path.parent != self.input_dir.resolve() or not self.processor.is_candidate(path):
            return
        with self._pending_lock:
            if path in self._pending:
                return
            self._pending.add(path)
        log.debug("Queued %s", path.name)
        self._queue.put(path)

    def scan_existing(self) -> int:
        """Queue images already sitting in the folder (e.g. dropped while the agent was off)."""
        count = 0
        for path in sorted(self.input_dir.iterdir()):
            if path.is_file() and self.processor.is_candidate(path):
                self.enqueue(path)
                count += 1
        return count

    def start(self) -> None:
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self._worker = threading.Thread(target=self._run_worker, name="catalog-worker", daemon=True)
        self._worker.start()
        observer_cls = PollingObserver if self.use_polling else Observer
        self._observer = observer_cls()
        self._observer.schedule(_EnqueueHandler(self), str(self.input_dir), recursive=False)
        self._observer.start()
        queued = self.scan_existing()
        log.info("Watching %s (%d existing image%s queued)", self.input_dir, queued, "" if queued == 1 else "s")

    def stop(self, drain: bool = False) -> None:
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
        if drain:
            self._queue.join()
        self._stop.set()
        self._queue.put(None)
        if self._worker:
            self._worker.join(timeout=10)

    def wait_idle(self) -> None:
        self._queue.join()

    def _run_worker(self) -> None:
        while not self._stop.is_set():
            path = self._queue.get()
            try:
                if path is None:
                    return
                try:
                    self.results.append(self.processor.process(path))
                except Exception:  # noqa: BLE001 - one bad file must never stop the loop
                    log.exception("Unexpected error while processing %s", path)
                finally:
                    with self._pending_lock:
                        self._pending.discard(path)
            finally:
                self._queue.task_done()
