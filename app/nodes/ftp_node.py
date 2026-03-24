"""
LangGraph node for Supply PR (FTP — First Time Purchase) validation.
A costing sheet attachment is required.
"""
from datetime import datetime, timezone
from typing import Optional

from app.state import PRValidationState, UploadedFile
from app.config import get_settings


def handle(state: PRValidationState) -> dict:
    uploaded_files = state.get("uploaded_files", [])
    messages = list(state.get("validation_messages", []))
    blocker_count = state.get("blocker_count", 0)

    # Find the costing sheet attachment
    costing_file: Optional[UploadedFile] = next(
        (f for f in uploaded_files if f.get("doc_type") in ("costing_sheet", "ftp")),
        None,
    )
    # Fallback: accept any uploaded file if only one is present
    if costing_file is None and len(uploaded_files) == 1:
        costing_file = uploaded_files[0]

    if costing_file is None:
        messages.append({
            "icon": "✗",
            "text": "FTP Costing Sheet not attached. This is a blocker — please attach the costing sheet to proceed.",
        })
        return {
            "validation_messages": messages,
            "blocker_count": blocker_count + 1,
        }

    messages.append({"icon": "✓", "text": f"Costing sheet attachment found: '{costing_file['name']}'."})

    settings = get_settings()
    if not settings.gemini_enabled:
        messages.append({"icon": "⚠", "text": "Gemini validation skipped (API key not configured). Please manually verify the costing sheet."})
        return {"validation_messages": messages, "blocker_count": blocker_count}

    from app.services import gemini_service

    prompt, lf_prompt = gemini_service.build_ftp_costing_sheet_prompt()
    raw = gemini_service.validate_document(costing_file["content"], costing_file["mime_type"], prompt,
                                           langfuse_prompt=lf_prompt, trace_id=state.get("trace_id", ""))

    try:
        result = gemini_service.parse_json_response(raw)
    except Exception:
        messages.append({"icon": "⚠", "text": "Could not parse Gemini validation response for costing sheet. Manual review recommended."})
        return {"validation_messages": messages, "blocker_count": blocker_count}

    cost_breakdown = result.get("cost_breakdown_present", "FAIL")
    cost_details = result.get("cost_breakdown_details", "")
    validity_present = result.get("validity_date_present", "FAIL")
    validity_date_str = result.get("validity_date")
    signature = result.get("authorising_signature", "FAIL")

    # Cost breakdown
    if cost_breakdown == "PASS":
        messages.append({"icon": "✓", "text": f"Cost breakdown verified: {cost_details or 'present'}."})
    else:
        messages.append({"icon": "✗", "text": f"Cost breakdown incomplete or missing. Required: material, processing, overhead, total unit cost. {cost_details or ''}"})
        blocker_count += 1

    # Validity date
    if validity_present == "FAIL" or not validity_date_str:
        messages.append({"icon": "✗", "text": "Validity date not found on costing sheet. This is a blocker."})
        blocker_count += 1
    else:
        age_warning = _check_date_age(validity_date_str, max_months=6)
        if age_warning:
            messages.append({"icon": "✗", "text": f"Costing sheet validity date '{validity_date_str}' is {age_warning}. This is a blocker — please obtain an updated costing sheet."})
            blocker_count += 1
        else:
            messages.append({"icon": "✓", "text": f"Validity date '{validity_date_str}' is current (within 6 months)."})

    # Signature
    if signature == "PASS":
        messages.append({"icon": "✓", "text": "Authorising signature or approver name found on costing sheet."})
    elif signature == "FAIL":
        messages.append({"icon": "✗", "text": "Authorising signature or approver name not found on costing sheet. This is a blocker."})
        blocker_count += 1
    else:
        messages.append({"icon": "⚠", "text": "Authorising signature visibility unclear. Please manually verify."})

    notes = result.get("notes", "")
    if notes:
        messages.append({"icon": "⚠", "text": f"Additional note — {notes}"})

    return {"validation_messages": messages, "blocker_count": blocker_count}


def _check_date_age(date_str: str, max_months: int = 6) -> Optional[str]:
    """
    Returns a human-readable warning string if the date is older than max_months,
    or None if the date is recent enough.
    """
    from dateutil import parser as dateparser

    try:
        parsed = dateparser.parse(date_str, dayfirst=False)
        if parsed is None:
            return None  # Can't parse; don't block
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        now = datetime.now(tz=timezone.utc)
        delta_months = (now.year - parsed.year) * 12 + (now.month - parsed.month)
        if delta_months > max_months:
            return f"more than {delta_months} months old"
    except Exception:
        return None  # On parse failure, be lenient

    return None
