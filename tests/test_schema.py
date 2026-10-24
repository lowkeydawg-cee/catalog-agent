import pytest
from pydantic import ValidationError

from catalog_agent.processor import extract_json
from catalog_agent.schema import ProductListing, output_json_schema

from conftest import valid_listing


def test_valid_listing_normalizes_tags():
    listing = ProductListing.model_validate(valid_listing())
    assert listing.tags == ["#3dprinting", "#customdesign", "#desksetup", "#organizer", "#maker"]
    assert len(listing.tiktok_hook) == 3


@pytest.mark.parametrize(
    "overrides, field",
    [
        ({"tiktok_hook": ["one hook", "two hook"]}, "tiktok_hook"),
        ({"tiktok_hook": ["Same line!", "same line", "Another"]}, "tiktok_hook"),
        ({"store_description": "A lovely organizer with no bullet points at all, just a long paragraph of prose text."}, "store_description"),
        ({"product_title": "#cool #print item"}, "product_title"),
        ({"product_title": "Short"}, "product_title"),
        ({"tags": ["#a", "#b", "#a"]}, "tags"),
        ({"extra_field": "nope"}, "extra_field"),
    ],
)
def test_invalid_listings_rejected(overrides, field):
    with pytest.raises(ValidationError) as exc:
        ProductListing.model_validate(valid_listing(**overrides))
    assert field in str(exc.value)


def test_missing_field_rejected():
    data = valid_listing()
    del data["instagram_caption"]
    with pytest.raises(ValidationError):
        ProductListing.model_validate(data)


def test_extract_json_handles_fences_and_chatter():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here it is: {"a": 1} hope that helps') == {"a": 1}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_json_schema_lists_required_fields():
    schema = output_json_schema()
    assert set(schema["required"]) >= {"product_title", "store_description", "instagram_caption", "tiktok_hook", "tags"}
