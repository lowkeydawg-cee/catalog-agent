"""The per-image pipeline: read -> analyze -> validate (with retries) -> save -> move."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError
from slugify import slugify

from .config import Settings
from .schema import ProductListing, output_json_schema
from .storage import CatalogStore
from .vision import VisionClient, VisionModelError

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a product cataloger and social media copywriter for a small e-commerce shop that sells 3D-printed items.
You look at one photo or CAD render of a single product and reply with ONLY a JSON object, no prose and no markdown fences."""

USER_PROMPT = """Analyze this 3D-printed item. First study it closely:
- shape: overall form and geometry
- material_texture: likely material (PLA, PETG, resin...), surface finish, visible layer lines, colour
- visible_utility: what it is for, judging by its features
- target_audience: who would buy it

Then write the listing. Return a JSON object with exactly these keys:
{
  "visual_analysis": {"shape": str, "material_texture": str, "visible_utility": str, "target_audience": str},
  "product_title": str,       // clean SEO title, 10-120 characters, no hashtags or emoji
  "store_description": str,   // a short intro paragraph, then at least 3 bullet points, one per line, each starting with "- "
  "instagram_caption": str,   // engaging story-style caption ending with a strong call-to-action
  "tiktok_hook": [str, str, str],  // exactly 3 distinct scroll-stopping opening lines
  "tags": [str, ...]          // 5-30 hashtags, e.g. "#3dprinting", "#customdesign"
}"""


class UnreadableImageError(Exception):
    """The file is not an image Pillow can decode."""


class ListingValidationError(Exception):
    """The model never produced output that satisfied the schema."""

    def __init__(self, message: str, last_output: str, attempts: int) -> None:
        super().__init__(message)
        self.last_output = last_output
        self.attempts = attempts


@dataclass
class ProcessResult:
    status: str  # processed | failed | duplicate | skipped | deferred
    source: Path
    destination: Path | None = None
    slug: str | None = None
    attempts: int = 0
    error: str | None = None


def wait_until_stable(path: Path, settle_seconds: float, timeout: float) -> bool:
    """Wait until the file size stops changing, so we don't read a half-copied file."""
    deadline = time.monotonic() + timeout
    last_size = -1
    stable_since: float | None = None
    while time.monotonic() < deadline:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return False
        now = time.monotonic()
        if size == last_size and size > 0:
            if stable_since is not None and now - stable_since >= settle_seconds:
                return True
        else:
            stable_since = now
            last_size = size
        time.sleep(min(0.25, max(settle_seconds / 4, 0.01)))
    return path.exists()


def load_image(path: Path, max_side: int) -> tuple[str, str]:
    """Return (base64 JPEG for the model, sha256 of the original bytes). Raises UnreadableImageError."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise UnreadableImageError(f"cannot read file: {exc}") from exc
    if not raw:
        raise UnreadableImageError("file is empty")
    try:
        with Image.open(io.BytesIO(raw)) as probe:
            probe.verify()  # catches truncated/corrupt files cheaply
        with Image.open(io.BytesIO(raw)) as img:
            img.load()
            img = img.convert("RGB")
            img.thumbnail((max_side, max_side))
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=90)
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise UnreadableImageError(f"not a decodable image: {exc}") from exc
    return base64.b64encode(buffer.getvalue()).decode("ascii"), hashlib.sha256(raw).hexdigest()


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict:
    """Parse the model reply, tolerating markdown fences or chatter around the object."""
    text = text.strip()
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("top-level JSON value must be an object")
    return data


def _describe_errors(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "\n".join(
            f"- {'.'.join(str(p) for p in err['loc']) or '(root)'}: {err['msg']}" for err in exc.errors()
        )
    return f"- invalid JSON: {exc}"


def generate_listing(client: VisionClient, image_b64: str, max_retries: int) -> tuple[ProductListing, int]:
    """Ask the model for a listing, feeding validation errors back until it passes or retries run out."""
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT, "images": [image_b64]},
    ]
    schema = output_json_schema()
    total_attempts = 1 + max(0, max_retries)
    raw = ""
    for attempt in range(1, total_attempts + 1):
        raw = client.chat(messages, json_schema=schema)
        try:
            return ProductListing.model_validate(extract_json(raw)), attempt
        except (ValidationError, ValueError) as exc:  # JSONDecodeError is a ValueError
            problems = _describe_errors(exc)
            log.warning("Schema validation failed (attempt %d/%d):\n%s", attempt, total_attempts, problems)
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": "Your JSON did not pass validation:\n"
                    f"{problems}\nReturn the full corrected JSON object only.",
                }
            )
    raise ListingValidationError(f"no valid output after {total_attempts} attempts", raw, total_attempts)


class ImageProcessor:
    def __init__(self, settings: Settings, client: VisionClient, store: CatalogStore) -> None:
        self.settings = settings
        self.client = client
        self.store = store

    def is_candidate(self, path: Path) -> bool:
        return (
            path.suffix.lower() in self.settings.image_extensions
            and not path.name.startswith(".")
            and not path.name.endswith((".part", ".crdownload", ".tmp"))
        )

    def process(self, path: Path) -> ProcessResult:
        path = Path(path)
        if not self.is_candidate(path):
            log.debug("Ignoring non-image file %s", path.name)
            return ProcessResult("skipped", path)
        if not wait_until_stable(path, self.settings.settle_seconds, self.settings.settle_timeout):
            log.info("File %s disappeared before processing", path.name)
            return ProcessResult("skipped", path)

        log.info("Processing %s", path.name)
        try:
            image_b64, digest = load_image(path, self.settings.max_image_side)
        except UnreadableImageError as exc:
            return self._fail(path, "unreadable", str(exc))

        if self.store.has_hash(digest):
            return self._fail(path, "duplicate", f"identical image already cataloged (sha256 {digest[:12]})", status="duplicate")

        try:
            listing, attempts = generate_listing(self.client, image_b64, self.settings.max_validation_retries)
        except ListingValidationError as exc:
            return self._fail(path, "validation", f"{exc}\n\nLast model output:\n{exc.last_output}", attempts=exc.attempts)
        except VisionModelError as exc:
            # Model server problem, not the image's fault: leave it in place to retry on next start.
            log.error("Model unavailable, leaving %s in input folder: %s", path.name, exc)
            self.store.record_failure(path.name, "model_unavailable", str(exc))
            return ProcessResult("deferred", path, error=str(exc))

        return self._save(path, listing, digest, attempts)

    def _save(self, path: Path, listing: ProductListing, digest: str, attempts: int) -> ProcessResult:
        slug = self._unique_slug(listing.product_title)
        ext = path.suffix.lower().replace(".jpeg", ".jpg")
        destination = self.settings.processed_dir / f"{slug}{ext}"
        try:
            with self.store.transaction() as conn:
                self.store.insert_product(
                    conn,
                    slug=slug,
                    listing=listing,
                    original_filename=path.name,
                    image_path=destination,
                    image_sha256=digest,
                    model=self.client.model,
                    attempts=attempts,
                )
                # Move inside the transaction: if the move fails, the row is rolled back.
                shutil.move(str(path), str(destination))
        except Exception as exc:  # noqa: BLE001 - any failure here must not kill the watcher
            log.exception("Failed to save %s", path.name)
            if path.exists():
                return self._fail(path, "save_error", str(exc), attempts=attempts)
            return ProcessResult("failed", path, error=str(exc), attempts=attempts)

        log.info("Cataloged %s -> %s (%d attempt%s)", path.name, destination.name, attempts, "" if attempts == 1 else "s")
        return ProcessResult("processed", path, destination, slug, attempts)

    def _unique_slug(self, title: str) -> str:
        base = slugify(title, max_length=80, word_boundary=True) or "product"
        slug, n = base, 2
        while self.store.slug_exists(slug) or any(self.settings.processed_dir.glob(f"{slug}.*")):
            slug = f"{base}-{n}"
            n += 1
        return slug

    def _fail(self, path: Path, reason: str, detail: str, *, status: str = "failed", attempts: int = 0) -> ProcessResult:
        log.error("%s: %s (%s)", path.name, reason, detail.splitlines()[0] if detail else "")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        destination = self.settings.failed_dir / f"{stamp}_{path.name}"
        try:
            shutil.move(str(path), str(destination))
            destination.with_name(destination.name + ".error.txt").write_text(f"reason: {reason}\n\n{detail}\n", encoding="utf-8")
        except OSError as exc:
            log.error("Could not move %s to failed folder: %s", path.name, exc)
            destination = None
        self.store.record_failure(path.name, reason, detail, destination)
        return ProcessResult(status, path, destination, attempts=attempts, error=f"{reason}: {detail}")
