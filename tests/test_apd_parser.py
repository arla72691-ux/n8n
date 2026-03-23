"""Tests for the APD item long text parser."""
import pytest
from app.utils.apd_parser import (
    parse_item_long_text,
    normalise_revision,
    unique_drawings,
    items_by_drawing,
    APDItem,
)


# ---------------------------------------------------------------------------
# normalise_revision
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rev,expected", [
    ("0",          "0"),
    ("00",         "0"),
    ("NO REVISION","0"),
    ("no revision","0"),
    ("NONE",       "0"),
    ("N/A",        "0"),
    ("-",          "0"),
    ("1",          "1"),
    ("A",          "A"),
    ("REV2",       "REV2"),
])
def test_normalise_revision(rev, expected):
    assert normalise_revision(rev) == expected


# ---------------------------------------------------------------------------
# parse_item_long_text — happy path
# ---------------------------------------------------------------------------

SAMPLE_LINE = "APD,ITEM NAME:HOLDBACK;DRAWING NUMBER:4802-1594;POSITION OR ITEM NUMBER:P3;REVISION:0"

def test_parse_single_line():
    items = parse_item_long_text(SAMPLE_LINE)
    assert len(items) == 1
    item = items[0]
    assert item.drawing_number == "4802-1594"
    assert item.revision == "0"
    assert item.normalised_revision == "0"
    assert item.item_name == "HOLDBACK"
    assert item.position == "P3"
    assert item.material is None


def test_parse_multiple_lines():
    text = "\n".join([
        "APD,ITEM NAME:FLANGE;DRAWING NUMBER:1000-001;POSITION OR ITEM NUMBER:P1;REVISION:1",
        "APD,ITEM NAME:BOLT;DRAWING NUMBER:1000-002;POSITION OR ITEM NUMBER:P2;REVISION:00",
    ])
    items = parse_item_long_text(text)
    assert len(items) == 2
    assert items[0].drawing_number == "1000-001"
    assert items[1].normalised_revision == "0"  # 00 → 0


def test_parse_with_material():
    line = "APD,ITEM NAME:SHAFT;DRAWING NUMBER:2000-100;REVISION:2;MATERIAL:42CRMO4V;POSITION OR ITEM NUMBER:P1"
    items = parse_item_long_text(line)
    assert len(items) == 1
    assert items[0].material == "42CRMO4V"


def test_parse_with_material_specification():
    line = "APD,ITEM NAME:PLATE;DRAWING NUMBER:2000-200;REVISION:0;MATERIAL SPECIFICATION:AISI 316L"
    items = parse_item_long_text(line)
    assert items[0].material == "AISI 316L"


# ---------------------------------------------------------------------------
# Formatting noise tolerance
# ---------------------------------------------------------------------------

def test_parse_without_apd_prefix():
    line = "ITEM NAME:GASKET;DRAWING NUMBER:3000-005;POSITION OR ITEM NUMBER:P5;REVISION:NO REVISION"
    items = parse_item_long_text(line)
    assert len(items) == 1
    assert items[0].normalised_revision == "0"


def test_parse_blank_lines_skipped():
    text = "\n\nAPD,ITEM NAME:X;DRAWING NUMBER:9999-001;REVISION:1\n\n"
    items = parse_item_long_text(text)
    assert len(items) == 1


def test_parse_line_without_drawing_number_skipped():
    text = "ITEM NAME:SOMETHING WITHOUT DRAWING"
    items = parse_item_long_text(text)
    assert len(items) == 0


def test_parse_extra_whitespace():
    line = " APD , ITEM NAME : WIDGET ; DRAWING NUMBER : 4000-001 ; REVISION : 0 "
    items = parse_item_long_text(line)
    # The regex strips the APD prefix but field values may have leading/trailing spaces
    # after partition — the parser should handle this
    assert len(items) == 1


# ---------------------------------------------------------------------------
# unique_drawings and items_by_drawing
# ---------------------------------------------------------------------------

def test_unique_drawings():
    text = "\n".join([
        "APD,ITEM NAME:A;DRAWING NUMBER:DRW-001;REVISION:0;POSITION OR ITEM NUMBER:P1",
        "APD,ITEM NAME:B;DRAWING NUMBER:DRW-001;REVISION:0;POSITION OR ITEM NUMBER:P2",
        "APD,ITEM NAME:C;DRAWING NUMBER:DRW-002;REVISION:1;POSITION OR ITEM NUMBER:P1",
    ])
    items = parse_item_long_text(text)
    uniq = unique_drawings(items)
    assert len(uniq) == 2
    assert {u.drawing_number for u in uniq} == {"DRW-001", "DRW-002"}


def test_items_by_drawing():
    text = "\n".join([
        "APD,ITEM NAME:A;DRAWING NUMBER:DRW-001;REVISION:0;POSITION OR ITEM NUMBER:P1",
        "APD,ITEM NAME:B;DRAWING NUMBER:DRW-001;REVISION:0;POSITION OR ITEM NUMBER:P2",
        "APD,ITEM NAME:C;DRAWING NUMBER:DRW-002;REVISION:1;POSITION OR ITEM NUMBER:P1",
    ])
    items = parse_item_long_text(text)
    groups = items_by_drawing(items)
    assert len(groups["DRW-001"]) == 2
    assert len(groups["DRW-002"]) == 1


# ---------------------------------------------------------------------------
# Edge cases — empty / blank input
# ---------------------------------------------------------------------------

def test_parse_empty_string():
    assert parse_item_long_text("") == []


def test_parse_only_blank_lines():
    assert parse_item_long_text("\n\n\n") == []
