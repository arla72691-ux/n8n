"""
Parse APD item long text lines into structured APDItem objects.

Expected format (one item per line):
  APD,ITEM NAME:HOLDBACK;DRAWING NUMBER:4802-1594;POSITION OR ITEM NUMBER:P3;REVISION:0

Fields are semicolon-delimited key:value pairs. The leading "APD," prefix is optional.
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional


# Treat these revision values as equivalent base revision
_BASE_REVISIONS = {"0", "00", "NO REVISION", "NONE", "N/A", "-"}


@dataclass
class APDItem:
    drawing_number: str
    revision: str
    normalised_revision: str   # canonical form ("0" for all base revisions)
    item_name: str
    position: str
    material: Optional[str] = None
    raw_line: str = field(default="", repr=False)


def normalise_revision(rev: str) -> str:
    """Normalise revision string: 0, 00, NO REVISION → '0'."""
    cleaned = rev.strip().upper()
    if cleaned in _BASE_REVISIONS:
        return "0"
    return cleaned


def _extract_fields(line: str) -> dict:
    """Extract key:value pairs from a semicolon-delimited APD item line."""
    # Strip leading "APD," prefix (case-insensitive)
    line = re.sub(r"(?i)^APD\s*,\s*", "", line.strip())

    fields: dict = {}
    # Split by semicolons; some entries may have extra whitespace
    parts = re.split(r"\s*;\s*", line)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Split on first colon only
        if ":" in part:
            key, _, value = part.partition(":")
            fields[key.strip().upper()] = value.strip()

    return fields


def parse_item_long_text(text: str) -> List[APDItem]:
    """
    Parse a multi-line APD item long text block into a list of APDItem.

    Each non-blank line is treated as one item entry.
    Lines that don't contain a DRAWING NUMBER are skipped with a warning.
    """
    items: List[APDItem] = []

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue

        fields = _extract_fields(stripped)

        drawing_number = (
            fields.get("DRAWING NUMBER") or fields.get("DRAWING NO") or ""
        ).strip()

        if not drawing_number:
            # Not a parseable APD item line; skip silently
            continue

        revision = (
            fields.get("REVISION") or fields.get("REV") or "0"
        ).strip()

        item_name = (
            fields.get("ITEM NAME") or fields.get("DESCRIPTION") or ""
        ).strip()

        position = (
            fields.get("POSITION OR ITEM NUMBER")
            or fields.get("POSITION")
            or fields.get("ITEM NUMBER")
            or fields.get("POS")
            or ""
        ).strip()

        material = (
            fields.get("MATERIAL SPECIFICATION")
            or fields.get("MATERIAL SPEC")
            or fields.get("MATERIAL")
            or fields.get("MAT")
            or None
        )
        if material:
            material = material.strip() or None

        items.append(
            APDItem(
                drawing_number=drawing_number,
                revision=revision,
                normalised_revision=normalise_revision(revision),
                item_name=item_name,
                position=position,
                material=material,
                raw_line=stripped,
            )
        )

    return items


def unique_drawings(items: List[APDItem]) -> List[APDItem]:
    """
    Return one representative APDItem per unique drawing_number + normalised_revision.
    If the same drawing appears with conflicting revisions, keep the first occurrence.
    """
    seen: dict = {}
    result: List[APDItem] = []
    for item in items:
        key = item.drawing_number
        if key not in seen:
            seen[key] = item
            result.append(item)
    return result


def items_by_drawing(items: List[APDItem]) -> dict:
    """Group all items by drawing_number → list of APDItem."""
    groups: dict = {}
    for item in items:
        groups.setdefault(item.drawing_number, []).append(item)
    return groups
