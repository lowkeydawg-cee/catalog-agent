"""Strict output schema the vision model must satisfy."""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+\S", re.MULTILINE)
_TAG_CLEAN_RE = re.compile(r"[^a-z0-9_]")


class VisualAnalysis(BaseModel):
    """What the model saw in the photo; stored for traceability."""

    model_config = ConfigDict(extra="forbid")

    shape: NonEmpty = Field(description="Overall form and geometry of the printed object.")
    material_texture: NonEmpty = Field(description="Filament/resin type guess, surface finish, layer lines, colour.")
    visible_utility: NonEmpty = Field(description="What the object is for, based on visible features.")
    target_audience: NonEmpty = Field(description="Who would buy this.")


class ProductListing(BaseModel):
    """The structured listing produced for each image."""

    model_config = ConfigDict(extra="forbid")

    visual_analysis: VisualAnalysis
    product_title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=10, max_length=120)] = Field(
        description="Clean, SEO-optimized title, 10-120 characters, no hashtags or emoji."
    )
    store_description: Annotated[str, StringConstraints(strip_whitespace=True, min_length=80)] = Field(
        description="E-commerce listing copy: a short intro paragraph followed by at least 3 bullet points, one per line, each starting with '- '."
    )
    instagram_caption: Annotated[str, StringConstraints(strip_whitespace=True, min_length=40, max_length=2200)] = Field(
        description="Engaging narrative caption ending with a clear call-to-action."
    )
    tiktok_hook: list[NonEmpty] = Field(
        min_length=3,
        max_length=3,
        description="Exactly 3 distinct, punchy opening lines for a short-form video.",
    )
    tags: list[NonEmpty] = Field(
        min_length=5,
        max_length=30,
        description="5-30 search hashtags such as #3dprinting, #customdesign.",
    )

    @field_validator("product_title")
    @classmethod
    def _clean_title(cls, value: str) -> str:
        if "#" in value:
            raise ValueError("product_title must not contain hashtags")
        return re.sub(r"\s+", " ", value)

    @field_validator("store_description")
    @classmethod
    def _has_bullets(cls, value: str) -> str:
        if len(_BULLET_RE.findall(value)) < 3:
            raise ValueError("store_description must contain at least 3 bullet points, one per line, each starting with '- '")
        return value

    @field_validator("tiktok_hook")
    @classmethod
    def _distinct_hooks(cls, value: list[str]) -> list[str]:
        normalized = {re.sub(r"\W+", " ", hook).strip().lower() for hook in value}
        if len(normalized) != len(value):
            raise ValueError("tiktok_hook must contain 3 distinct lines")
        return value

    @field_validator("tags")
    @classmethod
    def _normalize_tags(cls, value: list[str]) -> list[str]:
        tags: list[str] = []
        for raw in value:
            body = _TAG_CLEAN_RE.sub("", raw.strip().lstrip("#").lower().replace(" ", "").replace("-", ""))
            if body and f"#{body}" not in tags:
                tags.append(f"#{body}")
        if len(tags) < 5:
            raise ValueError("tags must contain at least 5 distinct, non-empty hashtags")
        return tags


def output_json_schema() -> dict:
    """JSON schema passed to Ollama's `format` field for constrained decoding."""
    return ProductListing.model_json_schema()
