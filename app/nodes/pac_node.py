"""
LangGraph node for PAC PR validation.
A PAC Certificate must be attached.
"""
from datetime import datetime, timezone
from typing import Optional

from app.state import PRValidationState, UploadedFile
from app.config import get_settings


_EXPIRY_WARNING_DAYS = 30


def handle(state: PRValidationState) -> dict:
    uploaded_files = state.get("uploaded_files", [])
    messages = list(state.get("validation_messages", []))
    blocker_count = state.get("blocker_count", 0)
    pr_description = state.get("pr_description", "")

    # Find the PAC certificate attachment
    pac_file: Optional[UploadedFile] = next(
        (f for f in uploaded_files if f.get("doc_type") in ("pac_cert", "pac")),
        None,
    )
    if pac_file is None and len(uploaded_files) == 1:
        pac_file = uploaded_files[0]

    if pac_file is None:
        messages.append({
            "icon": "✗",
            "text": "PAC Certificate not attached. This is a blocker — please attach the PAC Certificate to proceed.",
        })
        return {
            "validation_messages": messages,
            "blocker_count": blocker_count + 1,
        }

    messages.append({"icon": "✓", "text": f"PAC Certificate attachment found: '{pac_file['name']}'."})

    settings = get_settings()
    if not settings.gemini_enabled:
        messages.append({"icon": "⚠", "text": "Gemini validation skipped (API key not configured). Please manually verify the PAC Certificate."})
        return {"validation_messages": messages, "blocker_count": blocker_count}

    from app.services import gemini_service

    prompt, lf_prompt = gemini_service.build_pac_cert_prompt(pr_description)
    raw = gemini_service.validate_document(pac_file["content"], pac_file["mime_type"], prompt,
                                           langfuse_prompt=lf_prompt, trace_id=state.get("trace_id", ""))

    try:
        result = gemini_service.parse_json_response(raw)
    except Exception:
        messages.append({"icon": "⚠", "text": "Could not parse Gemini validation response for PAC Certificate. Manual review recommended."})
        return {"validation_messages": messages, "blocker_count": blocker_count}

    cert_number = result.get("pac_cert_number", "FAIL")
    vendor_name = result.get("vendor_name", "FAIL")
    expiry_present = result.get("expiry_date_present", "FAIL")
    expiry_date_str = result.get("expiry_date")
    signature = result.get("authorising_signature", "FAIL")
    scope_match = result.get("scope_match", "NOT_CHECKED")

    if cert_number == "PASS":
        messages.append({"icon": "✓", "text": "PAC Certificate number visible on document."})
    else:
        messages.append({"icon": "✗", "text": "PAC Certificate number not visible. This is a blocker."})
        blocker_count += 1

    if vendor_name == "PASS":
        messages.append({"icon": "✓", "text": "Vendor name visible on PAC Certificate."})
    else:
        messages.append({"icon": "✗", "text": "Vendor name not visible on PAC Certificate. This is a blocker."})
        blocker_count += 1

    if expiry_present == "FAIL" or not expiry_date_str:
        messages.append({"icon": "✗", "text": "Validity/expiry date not found on PAC Certificate. This is a blocker."})
        blocker_count += 1
    else:
        expiry_warning = _check_expiry(expiry_date_str)
        if expiry_warning == "expired":
            messages.append({"icon": "✗", "text": f"PAC Certificate has expired (expiry date: '{expiry_date_str}'). This is a blocker."})
            blocker_count += 1
        elif expiry_warning == "soon":
            messages.append({"icon": "⚠", "text": f"PAC Certificate expires within {_EXPIRY_WARNING_DAYS} days (expiry: '{expiry_date_str}'). Consider renewing."})
        else:
            messages.append({"icon": "✓", "text": f"PAC Certificate expiry date '{expiry_date_str}' is current."})

    if signature == "PASS":
        messages.append({"icon": "✓", "text": "Authorising signature found on PAC Certificate."})
    elif signature == "FAIL":
        messages.append({"icon": "✗", "text": "Authorising signature not found on PAC Certificate. This is a blocker."})
        blocker_count += 1
    else:
        messages.append({"icon": "⚠", "text": "Authorising signature visibility unclear. Please manually verify."})

    if scope_match == "PASS":
        messages.append({"icon": "✓", "text": "Certificate scope broadly matches the PR description."})
    elif scope_match == "FAIL":
        messages.append({"icon": "⚠", "text": "Certificate scope may not match the PR description. Please verify alignment."})

    notes = result.get("notes", "")
    if notes:
        messages.append({"icon": "⚠", "text": f"Additional note — {notes}"})

    return {"validation_messages": messages, "blocker_count": blocker_count}


def _check_expiry(date_str: str) -> Optional[str]:
    """Returns 'expired', 'soon', or None."""
    from dateutil import parser as dateparser

    try:
        parsed = dateparser.parse(date_str, dayfirst=False)
        if parsed is None:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        now = datetime.now(tz=timezone.utc)
        delta = (parsed - now).days
        if delta < 0:
            return "expired"
        if delta <= _EXPIRY_WARNING_DAYS:
            return "soon"
    except Exception:
        return None

    return None
