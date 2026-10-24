# Catalog Agent

[![tests](https://github.com/lowkeydawg-cee/catalog-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/lowkeydawg-cee/catalog-agent/actions/workflows/tests.yml)

A local, event-driven agent for a 3D-printing shop. Drop a product photo or CAD render into `input_images/` and it:

1. notices the new file (watchdog),
2. sends it to a local vision model through Ollama (`llama3.2-vision` by default),
3. forces the reply into a strict Pydantic schema, feeding validation errors back to the model and retrying if it gets it wrong,
4. saves the listing to SQLite and moves the image to `processed_images/<seo-slug>.<ext>`,
5. logs everything, and parks unreadable or unfixable files in `failed_images/` with an `.error.txt` explaining why.

Everything runs on your machine. No cloud APIs.

## Output schema

Each image produces one record (`catalog_agent/schema.py`):

| Field | Rules |
|---|---|
| `product_title` | 10–120 chars, no hashtags; used for the filename slug |
| `store_description` | intro + at least 3 bullet lines (`- ...`) |
| `instagram_caption` | 40–2200 chars, ends with a call-to-action |
| `tiktok_hook` | exactly 3 distinct opening lines |
| `tags` | 5–30 hashtags, normalized to lowercase `#tag`, de-duplicated |
| `visual_analysis` | `shape`, `material_texture`, `visible_utility`, `target_audience`: what the model saw, kept for traceability |

The schema is also sent to Ollama's `format` field, so the model is constrained to valid JSON from the start; Pydantic then enforces the rules JSON Schema can't (bullet count, distinct hooks, tag cleanup).

## Setup

Requirements: Python 3.10+, and [Ollama](https://ollama.com/download) running locally.

```bash
# 1. Model (about 8 GB; needs ~12 GB RAM/VRAM to run comfortably)
ollama pull llama3.2-vision

# 2. Python environment
git clone https://github.com/lowkeydawg-cee/catalog-agent.git
cd catalog-agent
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# 3. Confirm Ollama is up and the model is pulled
python -m catalog_agent --check
```

## Running

```bash
# Watch input_images/ continuously (Ctrl+C to stop)
python -m catalog_agent

# Process whatever is already in input_images/ and exit (good for cron/batch)
python -m catalog_agent --once

# Dump the catalog and the failure log as JSON
python -m catalog_agent --list
```

Folders are created on first run, relative to where you start it. Any images already sitting in `input_images/` when the agent starts are processed first, so nothing dropped while it was off gets missed.

Useful flags: `--input-dir`, `--processed-dir`, `--failed-dir`, `--db`, `--model bakllava`, `--ollama-host`, `--max-retries`, `--log-level DEBUG`, and `--polling` (use this if the folder is on a network drive, Docker volume or WSL mount where file events don't fire).

All settings can also come from environment variables; see `.env.example`.

## What happens when things go wrong

| Situation | Behaviour |
|---|---|
| File still being copied | waits until its size stops changing before reading |
| Not an image / corrupt / empty | never sent to the model; moved to `failed_images/` with reason `unreadable` |
| Model returns bad JSON or breaks a rule | errors are sent back to the model as a correction prompt; up to `CATALOG_MAX_VALIDATION_RETRIES` (default 3) retries, then `failed_images/` with the last raw output in the `.error.txt` |
| Ollama down, or 5xx while loading the model | HTTP retried with exponential backoff; if it still fails the image **stays in `input_images/`** and is picked up on the next start |
| Same image dropped twice (same bytes) | not re-sent to the model; moved to `failed_images/` as `duplicate` |
| Two products get the same title | slug gets `-2`, `-3`, ... |
| DB write or file move fails | both are done in one transaction: the row is rolled back if the move fails |
| Any unexpected exception | logged with traceback; the watcher keeps running |

Logs go to the console and to `logs/catalog_agent.log` (rotating, 5 × 5 MB). Every failure is also recorded in the `failures` table.

## Database

`catalog.db` (SQLite) has two tables:

- `products`: one row per cataloged image, with each field as a column (`tiktok_hook`, `tags`, `visual_analysis` stored as JSON), the full validated payload in `listing_json`, original filename, new image path, SHA-256, model name, number of attempts and timestamp.
- `failures`: filename, reason (`unreadable`, `validation`, `duplicate`, `save_error`, `model_unavailable`), detail and where the file was moved.

```bash
sqlite3 catalog.db "select product_title, json_extract(tags, '$[0]') from products;"
```

## Tests

```bash
pytest
```

The suite replaces Ollama with a fake client (and `httpx.MockTransport` for the HTTP layer), so it runs without a model. It covers the happy path, validation retries with error feedback, retries exhausted, unreadable files, duplicates, slug collisions, Ollama outages, DB rollback when the move fails, and the live watcher picking up both new files and renamed-in files.

## Layout

```
catalog_agent/
  __main__.py      CLI
  config.py        settings from env vars / flags
  watcher.py       watchdog observer + single worker queue
  processor.py     read -> analyze -> validate/retry -> save -> move
  vision.py        Ollama /api/chat client with HTTP retries
  schema.py        Pydantic output schema
  storage.py       SQLite
  logging_setup.py
tests/
```

## Tuning

- **Speed:** images are downscaled to 1120 px on the long side before sending (`CATALOG_MAX_IMAGE_SIDE`). Llama 3.2 Vision works in 560 px tiles, so larger buys little.
- **Other models:** any Ollama vision model works with `--model` (e.g. `bakllava`, `llava:13b`, `qwen2.5vl`). Smaller models fail validation more often; raise `--max-retries` if so.
- **Voice:** edit `SYSTEM_PROMPT` / `USER_PROMPT` in `processor.py` to match your brand tone.
