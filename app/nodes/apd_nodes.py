"""
LangGraph nodes for Supply PR (APD — As Per Drawing) validation.

Flow: parse_items → validate_documents → secondary_validations

Validation logic:
  1. parse_items   — parse item_long_text into structured APDItem objects.
  2. validate_docs — for each unique drawing, require an uploaded drawing attachment
                     and use Gemini to read its title block and schedule/BOM table,
                     cross-checking drawing number, part/item number, and revision
                     against the values in item_long_text. Also checks the file
                     contains actual drawing content (not blank pages).
  3. secondary     — flag multi-position drawings that cover several line items.
"""
from typing import List, Optional

from app.state import PRValidationState, ValidationMessage, UploadedFile
from app.utils.apd_parser import parse_item_long_text, unique_drawings, items_by_drawing, APDItem
from app.config import get_settings


def _msg(icon: str, text: str) -> ValidationMessage:
    return {"icon": icon, "text": text}


def _find_uploaded_file_for_drawing(
    drawing_number: str, uploaded_files: list
) -> Optional[UploadedFile]:
    """Find an uploaded file whose name contains the drawing number."""
    dn_upper = drawing_number.upper()
    for f in uploaded_files:
        if dn_upper in f["name"].upper():
            return f
    return None


# ---------------------------------------------------------------------------
# Node 1: Parse item long text
# ---------------------------------------------------------------------------

def parse_items(state: PRValidationState) -> dict:
    text = state.get("item_long_text", "")
    items = parse_item_long_text(text)
    messages = list(state.get("validation_messages", []))

    if not items:
        messages.append(_msg("✗", "No valid APD items could be parsed from the Item Long Text. Please check the format."))
        return {
            "parsed_items": [],
            "validation_messages": messages,
            "blocker_count": state.get("blocker_count", 0) + 1,
        }

    return {"parsed_items": items}


# ---------------------------------------------------------------------------
# Node 2: Validate uploaded drawing attachments against item long text
# ---------------------------------------------------------------------------

def validate_documents(state: PRValidationState) -> dict:
    settings = get_settings()
    parsed_items = state.get("parsed_items", [])
    uploaded_files = state.get("uploaded_files", [])
    messages = list(state.get("validation_messages", []))
    blocker_count = state.get("blocker_count", 0)
    trace_id = state.get("trace_id", "")

    if not parsed_items:
        return {"validation_messages": messages, "blocker_count": blocker_count}

    items_map = items_by_drawing(parsed_items)

    for drawing_number, drawing_items in items_map.items():
        rep_item: APDItem = drawing_items[0]
        rev = rep_item.normalised_revision
        part_numbers = [i.position for i in drawing_items if i.position]

        # Find the uploaded drawing attachment for this drawing number.
        # 1. Match by drawing number in filename
        # 2. Any file explicitly tagged as "drawing"
        # 3. Any file with no doc_type set (APD uploads from the frontend)
        uploaded = _find_uploaded_file_for_drawing(drawing_number, uploaded_files)
        if not uploaded:
            uploaded = next((f for f in uploaded_files if f.get("doc_type") == "drawing"), None)
        if not uploaded:
            uploaded = next((f for f in uploaded_files if not f.get("doc_type")), None)

        if not uploaded:
            messages.append(_msg("✗", (
                f"Drawing {drawing_number}: No drawing attachment found. "
                "Please attach the drawing to proceed."
            )))
            blocker_count += 1
            continue

        if settings.gemini_enabled:
            result = _gemini_validate_drawing(
                uploaded["content"], uploaded["mime_type"],
                drawing_number, rev, part_numbers, trace_id,
            )
            messages.extend(result["messages"])
            blocker_count += result["blockers"]
        else:
            messages.append(_msg("⚠", (
                f"Drawing {drawing_number}: Gemini validation skipped (API key not configured). "
                "Please manually verify the drawing number, revision, and part/item numbers "
                "against the item long text."
            )))

    return {
        "validation_messages": messages,
        "blocker_count": blocker_count,
    }


def _gemini_validate_drawing(
    file_bytes: bytes,
    mime_type: str,
    drawing_number: str,
    revision: str,
    part_numbers: List[str] = None,
    trace_id: str = "",
) -> dict:
    from app.services import gemini_service

    prompt, lf_prompt = gemini_service.build_apd_drawing_prompt(
        drawing_number, revision, part_numbers or []
    )
    raw = gemini_service.validate_document(
        file_bytes, mime_type, prompt,
        langfuse_prompt=lf_prompt, trace_id=trace_id,
    )

    try:
        result = gemini_service.parse_json_response(raw)
    except Exception:
        return {
            "messages": [_msg("⚠", f"Drawing {drawing_number}: Could not parse Gemini validation response. Manual review recommended.")],
            "blockers": 0,
        }

    messages = []
    blockers = 0

    has_drawings = result.get("has_drawings", "PASS")
    dn_match     = result.get("drawing_number_match", "UNCLEAR")
    rev_match    = result.get("revision_match", "UNCLEAR")
    pn_match     = result.get("part_number_match", "NOT_CHECKED")
    found_dn     = result.get("found_drawing_number")
    found_rev    = result.get("found_revision")
    found_pn     = result.get("found_part_numbers")
    notes        = result.get("notes", "")

    # Check 1 — Drawing content present
    if has_drawings == "FAIL":
        messages.append(_msg("✗", (
            f"Drawing {drawing_number}: The attached file appears to be blank or contains no "
            "drawing content. Please attach a valid engineering drawing."
        )))
        blockers += 1
        return {"messages": messages, "blockers": blockers}

    # Check 2 — Drawing number
    if dn_match == "PASS":
        messages.append(_msg("✓", f"Drawing {drawing_number}: Drawing number verified on title block."))
    elif dn_match == "FAIL":
        found_str = f" (found: {found_dn})" if found_dn else ""
        messages.append(_msg("✗", (
            f"Drawing {drawing_number}: Drawing number on attachment does not match{found_str}. "
            "Please verify the correct drawing is attached."
        )))
        blockers += 1
    else:
        messages.append(_msg("⚠", f"Drawing {drawing_number}: Drawing number could not be clearly read from title block. Please manually verify."))

    # Check 3 — Revision
    if rev_match == "PASS":
        messages.append(_msg("✓", f"Drawing {drawing_number}: Revision {revision} verified on drawing."))
    elif rev_match == "FAIL":
        found_str = f" (found: REV {found_rev})" if found_rev else ""
        messages.append(_msg("✗", (
            f"Drawing {drawing_number}: Revision mismatch — item long text specifies REV {revision}{found_str}. "
            "Please confirm the correct revision is attached."
        )))
        blockers += 1
    else:
        messages.append(_msg("⚠", f"Drawing {drawing_number}: Revision could not be clearly read. Please manually verify REV {revision}."))

    # Check 4 — Part/item numbers from schedule table
    if pn_match == "PASS":
        messages.append(_msg("✓", f"Drawing {drawing_number}: Part/item number(s) verified in drawing schedule table."))
    elif pn_match == "FAIL":
        found_str = f" (found in table: {found_pn})" if found_pn else ""
        messages.append(_msg("✗", (
            f"Drawing {drawing_number}: Part/item number(s) from item long text not found in "
            f"drawing schedule/BOM table{found_str}. Please verify the drawing covers the correct items."
        )))
        blockers += 1
    elif pn_match == "UNCLEAR":
        messages.append(_msg("⚠", f"Drawing {drawing_number}: Part/item numbers could not be clearly read from schedule table. Please manually verify."))
    # NOT_CHECKED: no schedule table visible — silently skip

    if notes:
        messages.append(_msg("⚠", f"Drawing {drawing_number}: {notes}"))

    return {"messages": messages, "blockers": blockers}


# ---------------------------------------------------------------------------
# Node 3: Secondary validations (multi-position drawing check)
# ---------------------------------------------------------------------------

def secondary_validations(state: PRValidationState) -> dict:
    parsed_items = state.get("parsed_items", [])
    messages = list(state.get("validation_messages", []))
    blocker_count = state.get("blocker_count", 0)

    items_map = items_by_drawing(parsed_items)

    for drawing_number, drawing_items in items_map.items():
        if len(drawing_items) > 1:
            positions = [i.position for i in drawing_items if i.position]
            pos_str = ", ".join(positions) if positions else "(unlabelled)"
            messages.append(_msg("⚠", (
                f"Drawing {drawing_number} covers {len(drawing_items)} items (positions: {pos_str}). "
                "Confirm the drawing schedule table lists all of these positions."
            )))

    return {
        "validation_messages": messages,
        "blocker_count": blocker_count,
    }
