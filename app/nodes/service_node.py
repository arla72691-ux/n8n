"""
LangGraph node for Service PR validation.
Three documents are required: Scope of Work, JSA, and Technical Skill-Set Document.
"""
from typing import Optional

from app.state import PRValidationState, UploadedFile
from app.config import get_settings


_REQUIRED_DOCS = {
    "scope_of_work": "Scope of Work",
    "jsa": "Job Safety Analysis (JSA)",
    "technical_skill_set": "Technical Skill-Set Document",
}


def handle(state: PRValidationState) -> dict:
    uploaded_files = state.get("uploaded_files", [])
    messages = list(state.get("validation_messages", []))
    blocker_count = state.get("blocker_count", 0)

    # Build a map of doc_type → uploaded file
    doc_map: dict = {}
    for f in uploaded_files:
        doc_type = f.get("doc_type", "")
        if doc_type in _REQUIRED_DOCS:
            doc_map[doc_type] = f

    # Check for missing documents
    missing = [label for key, label in _REQUIRED_DOCS.items() if key not in doc_map]
    if missing:
        for label in missing:
            messages.append({
                "icon": "✗",
                "text": f"Required document missing: '{label}'. This is a blocker.",
            })
            blocker_count += 1
        return {"validation_messages": messages, "blocker_count": blocker_count}

    messages.append({"icon": "✓", "text": "All three required Service PR documents are attached."})

    settings = get_settings()

    if not settings.gemini_enabled:
        messages.append({"icon": "⚠", "text": "Gemini validation skipped (API key not configured). Please manually verify all three service documents."})
        return {"validation_messages": messages, "blocker_count": blocker_count}

    from app.services import gemini_service

    doc_results: dict = {}

    # Validate each document
    for doc_type, label in _REQUIRED_DOCS.items():
        f = doc_map[doc_type]
        messages.append({"icon": "✓", "text": f"Validating '{label}': '{f['name']}'."})

        if doc_type == "scope_of_work":
            prompt = gemini_service.build_scope_of_work_prompt()
        elif doc_type == "jsa":
            prompt = gemini_service.build_jsa_prompt()
        else:
            prompt = gemini_service.build_technical_skillset_prompt()

        try:
            raw = gemini_service.validate_document(f["content"], f["mime_type"], prompt)
            result = gemini_service.parse_json_response(raw)
            doc_results[doc_type] = result
        except Exception as exc:
            messages.append({"icon": "⚠", "text": f"{label}: Could not parse Gemini response ({exc}). Manual review recommended."})
            doc_results[doc_type] = {}
            continue

        result_messages, result_blockers = _evaluate_doc(doc_type, label, result)
        messages.extend(result_messages)
        blocker_count += result_blockers

    # Cross-check: vendor name and work description consistency
    cross_messages = _cross_check(doc_results)
    messages.extend(cross_messages)

    return {"validation_messages": messages, "blocker_count": blocker_count}


def _evaluate_doc(doc_type: str, label: str, result: dict) -> tuple:
    messages = []
    blockers = 0

    if doc_type == "scope_of_work":
        checks = [
            ("deliverables_present", "Deliverables section"),
            ("timeline_present", "Timeline/duration"),
            ("acceptance_criteria_present", "Acceptance criteria/sign-off conditions"),
        ]
    elif doc_type == "jsa":
        checks = [
            ("hazards_identified", "Hazard identification"),
            ("control_measures_present", "Control measures/mitigations"),
            ("safety_officer_signature", "Safety officer/HSE authority signature"),
        ]
    else:  # technical_skill_set
        checks = [
            ("qualifications_listed", "Required qualifications"),
            ("certifications_listed", "Required certifications/licences"),
        ]

    for field, description in checks:
        value = result.get(field, "FAIL")
        if value == "PASS":
            messages.append({"icon": "✓", "text": f"{label}: {description} — found."})
        elif value == "FAIL":
            messages.append({"icon": "✗", "text": f"{label}: {description} — not found. This is a blocker."})
            blockers += 1
        else:
            messages.append({"icon": "⚠", "text": f"{label}: {description} — unclear. Please manually verify."})

    notes = result.get("notes", "")
    if notes:
        messages.append({"icon": "⚠", "text": f"{label}: Additional note — {notes}"})

    return messages, blockers


def _cross_check(doc_results: dict) -> list:
    messages = []

    # Collect vendor names and work description summaries
    vendors = {}
    summaries = {}
    for doc_type, result in doc_results.items():
        if not result:
            continue
        vendor = result.get("vendor_name")
        if vendor:
            vendors[doc_type] = vendor.strip().lower()
        summary = result.get("work_description_summary")
        if summary:
            summaries[doc_type] = summary.strip()

    # Vendor name consistency
    if len(vendors) >= 2:
        unique_vendors = set(vendors.values())
        if len(unique_vendors) > 1:
            vendor_details = "; ".join(f"{_REQUIRED_DOCS[k]}: '{v}'" for k, v in vendors.items())
            messages.append({
                "icon": "⚠",
                "text": f"Cross-check: Vendor names appear inconsistent across documents — {vendor_details}. Please verify all documents are from the same vendor.",
            })

    return messages
