"""
LangGraph nodes for Supply PR (APD — As Per Drawing) validation.

Flow: parse_items → search_drive → validate_documents → secondary_validations
"""
import mimetypes
from typing import Optional

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
# Node 2: Search Google Drive for each unique drawing
# ---------------------------------------------------------------------------

def search_drive(state: PRValidationState) -> dict:
    settings = get_settings()
    parsed_items = state.get("parsed_items", [])

    if not parsed_items:
        return {}

    drive_results: dict = {}

    if not settings.drive_enabled:
        # Drive not configured — mark all as not found (will fall through to attachment check)
        for item in unique_drawings(parsed_items):
            drive_results[item.drawing_number] = {"file": None, "bytes": None}
        return {"drive_results": drive_results}

    from app.services import drive_service

    for item in unique_drawings(parsed_items):
        dn = item.drawing_number
        rev = item.normalised_revision
        try:
            found = drive_service.search_drawing(dn, rev)
            if found:
                file_bytes = drive_service.download_file_bytes(found.id)
                drive_results[dn] = {"file": found, "bytes": file_bytes}
            else:
                drive_results[dn] = {"file": None, "bytes": None}
        except Exception as exc:
            drive_results[dn] = {"file": None, "bytes": None, "error": str(exc)}

    return {"drive_results": drive_results}


# ---------------------------------------------------------------------------
# Node 3: Validate documents (Drive file or uploaded attachment)
# ---------------------------------------------------------------------------

def validate_documents(state: PRValidationState) -> dict:
    settings = get_settings()
    parsed_items = state.get("parsed_items", [])
    drive_results = state.get("drive_results", {})
    uploaded_files = state.get("uploaded_files", [])
    messages = list(state.get("validation_messages", []))
    blocker_count = state.get("blocker_count", 0)

    items_map = items_by_drawing(parsed_items)

    for drawing_number, drawing_items in items_map.items():
        rep_item: APDItem = drawing_items[0]
        rev = rep_item.normalised_revision
        drive_entry = drive_results.get(drawing_number, {"file": None, "bytes": None})
        drive_file = drive_entry.get("file")
        drive_bytes = drive_entry.get("bytes")
        drive_error = drive_entry.get("error")

        if drive_error:
            messages.append(_msg("⚠", f"Drawing {drawing_number}: Drive search encountered an error ({drive_error}). Checking for attachment."))

        if drive_file:
            # Drawing found in Drive
            if settings.gemini_enabled:
                result = _gemini_validate_drawing(drive_bytes, drive_file.mime_type or "application/pdf", drawing_number, rev)
                messages.extend(result["messages"])
                blocker_count += result["blockers"]
            else:
                messages.append(_msg("✓", f"Drawing {drawing_number} found in system (REV {rev}). Gemini validation skipped (API key not configured)."))
        else:
            # Not in Drive — look for uploaded attachment
            uploaded = _find_uploaded_file_for_drawing(drawing_number, uploaded_files)
            if not uploaded:
                # Also accept any file tagged as "drawing"
                uploaded = next((f for f in uploaded_files if f.get("doc_type") == "drawing"), None)

            if not uploaded:
                messages.append(_msg("✗", f"Drawing {drawing_number} not found in the system and no drawing attached. This is a blocker — please attach the drawing to proceed."))
                blocker_count += 1
            else:
                messages.append(_msg("⚠", f"Drawing {drawing_number} not found in system. Validating attached file '{uploaded['name']}'."))
                if settings.gemini_enabled:
                    result = _gemini_validate_drawing(uploaded["content"], uploaded["mime_type"], drawing_number, rev)
                    messages.extend(result["messages"])
                    blocker_count += result["blockers"]
                else:
                    messages.append(_msg("⚠", f"Gemini validation skipped (API key not configured). Please manually verify drawing {drawing_number}."))

    return {
        "validation_messages": messages,
        "blocker_count": blocker_count,
    }


def _gemini_validate_drawing(
    file_bytes: bytes, mime_type: str, drawing_number: str, revision: str
) -> dict:
    from app.services import gemini_service

    prompt = gemini_service.build_apd_drawing_prompt(drawing_number, revision)
    raw = gemini_service.validate_document(file_bytes, mime_type, prompt)

    try:
        result = gemini_service.parse_json_response(raw)
    except Exception:
        return {
            "messages": [_msg("⚠", f"Drawing {drawing_number}: Could not parse Gemini validation response. Manual review recommended.")],
            "blockers": 0,
        }

    messages = []
    blockers = 0

    dn_match = result.get("drawing_number_match", "UNCLEAR")
    rev_match = result.get("revision_match", "UNCLEAR")
    stamp = result.get("approval_stamp", "UNCLEAR")
    legible = result.get("legible", "FAIL")
    notes = result.get("notes", "")

    if legible == "FAIL":
        messages.append(_msg("✗", f"Drawing {drawing_number}: Drawing is not legible. Please attach a clear, readable copy."))
        blockers += 1
        return {"messages": messages, "blockers": blockers}

    if dn_match == "PASS":
        messages.append(_msg("✓", f"Drawing {drawing_number}: Drawing number verified on document."))
    elif dn_match == "FAIL":
        messages.append(_msg("✗", f"Drawing {drawing_number}: Drawing number not visible or does not match. Please verify."))
        blockers += 1
    else:
        messages.append(_msg("⚠", f"Drawing {drawing_number}: Drawing number visibility unclear. Please manually verify."))

    if rev_match == "PASS":
        messages.append(_msg("✓", f"Drawing {drawing_number}: Revision {revision} verified."))
    elif rev_match == "FAIL":
        messages.append(_msg("✗", f"Drawing {drawing_number}: Revision mismatch detected. Item long text specifies REV {revision}. Please confirm correct revision."))
        blockers += 1
    else:
        messages.append(_msg("⚠", f"Drawing {drawing_number}: Revision visibility unclear. Please manually verify REV {revision}."))

    if stamp == "PASS":
        messages.append(_msg("✓", f"Drawing {drawing_number}: Approval stamp or authorising signature found on title block."))
    elif stamp == "FAIL":
        messages.append(_msg("✗", f"Drawing {drawing_number}: No approval stamp or authorising engineer found on title block. This is a blocker."))
        blockers += 1
    else:
        messages.append(_msg("⚠", f"Drawing {drawing_number}: Approval stamp unclear. Please manually verify title block."))

    if notes:
        messages.append(_msg("⚠", f"Drawing {drawing_number}: Additional note — {notes}"))

    return {"messages": messages, "blockers": blockers}


# ---------------------------------------------------------------------------
# Node 4: Secondary validations (material mismatch, multi-position check)
# ---------------------------------------------------------------------------

def secondary_validations(state: PRValidationState) -> dict:
    settings = get_settings()
    parsed_items = state.get("parsed_items", [])
    drive_results = state.get("drive_results", {})
    messages = list(state.get("validation_messages", []))
    blocker_count = state.get("blocker_count", 0)

    items_map = items_by_drawing(parsed_items)

    for drawing_number, drawing_items in items_map.items():
        # Material mismatch check (only if Gemini is available and drawing was found)
        materials = [i.material for i in drawing_items if i.material]
        if materials and settings.gemini_enabled:
            drive_entry = drive_results.get(drawing_number, {})
            file_bytes = drive_entry.get("bytes")
            if file_bytes:
                material_str = ", ".join(set(materials))
                prompt = (
                    f"Does this engineering drawing reference or specify the material '{material_str}'? "
                    f"If a different material is specified on the drawing, flag it. "
                    f"Respond with JSON: {{\"material_match\": \"PASS\"|\"FAIL\"|\"UNCLEAR\", \"drawing_material\": \"<material on drawing or null>\"}}"
                )
                try:
                    from app.services import gemini_service
                    raw = gemini_service.validate_document(file_bytes, "application/pdf", prompt)
                    result = gemini_service.parse_json_response(raw)
                    if result.get("material_match") == "FAIL":
                        drawing_mat = result.get("drawing_material", "unknown")
                        messages.append(_msg("⚠", f"Drawing {drawing_number}: Material mismatch — item long text specifies '{material_str}' but drawing shows '{drawing_mat}'. Please verify."))
                except Exception:
                    pass  # Don't block on secondary check failures

        # Multi-position drawing check
        if len(drawing_items) > 1:
            positions = [i.position for i in drawing_items if i.position]
            pos_str = ", ".join(positions) if positions else "(unlabelled)"
            messages.append(_msg("⚠", f"Drawing {drawing_number} is used for {len(drawing_items)} items (positions: {pos_str}). Confirm the drawing covers all listed positions."))

    return {
        "validation_messages": messages,
        "blocker_count": blocker_count,
    }
